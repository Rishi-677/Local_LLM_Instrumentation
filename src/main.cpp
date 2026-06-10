// Local_LLM_Instrumentation — application entry point + integration.
//
// Wires the whole pipeline together:
//
//   producer (inference OR replay) ──► EventRing ──► consumer thread
//                                                      │
//        cb_eval hook (Capturer)                       ├─ Topology.update
//                                                       ├─ AnomalyDetector.inspect
//                                                       ├─ Recorder.write (optional)
//                                                       └─ UiState setters ──► App (FTXUI, main thread)
//
// Two modes:
//   live   : load a GGUF, attach the non-invasive cb_eval hook, generate tokens.
//   replay : feed a recorded NDJSON session through the same consumer pipeline
//            (no model needed) — proves the schema seam.

#include "llama.h"
#include "ggml.h"

#include "event.hpp"
#include "ring_buffer.hpp"
#include "capture/capturer.hpp"
#include "capture/topology.hpp"
#include "anomaly.hpp"
#include "session/recorder.hpp"
#include "session/replay.hpp"
#include "tui/ui_state.hpp"
#include "tui/app.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string prompt   = "The quick brown fox jumps over the lazy dog.";
    std::string record;          // path to write NDJSON, empty = off
    std::string replay;          // path to read NDJSON, empty = live mode
    int   max_tokens     = 64;
    int   delay_ms       = 0;    // optional pacing so the live run is watchable
    float anomaly_thresh = 1e4f;
    bool  no_flash_attn  = true; // default OFF so attention matrices materialize
    bool  headless       = false;// run to completion without the TUI (verify/CI)
    bool  help           = false;
};

void print_usage(std::FILE* f) {
    std::fprintf(f,
        "Local_LLM_Instrumentation - terminal-native live debugger for local LLMs\n\n"
        "usage:\n"
        "  local_llm_instrumentation <model.gguf> [prompt] [options]\n"
        "  local_llm_instrumentation --replay <session.ndjson> [options]\n\n"
        "options:\n"
        "  --record <file>          record telemetry to NDJSON\n"
        "  --replay <file>          replay a recorded session (no model)\n"
        "  --max-tokens <n>         tokens to generate (default 64)\n"
        "  --delay-ms <n>           pace tokens/events so the run is watchable\n"
        "  --anomaly-threshold <x>  |value| above x flags an outlier (default 1e4)\n"
        "  --no-flash-attn          disable flash attention (default; exposes attention)\n"
        "  --flash-attn             allow auto flash attention (hides attention matrix)\n"
        "  --headless               run to completion without the TUI (verify / CI)\n"
        "  -h, --help               show this help\n");
}

Options parse_args(int argc, char** argv) {
    Options o;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); return ""; }
            return argv[++i];
        };
        if      (a == "--record")            o.record = next("--record");
        else if (a == "--replay")            o.replay = next("--replay");
        else if (a == "--max-tokens")        o.max_tokens = std::atoi(next("--max-tokens").c_str());
        else if (a == "--delay-ms")          o.delay_ms = std::atoi(next("--delay-ms").c_str());
        else if (a == "--anomaly-threshold") o.anomaly_thresh = (float)std::atof(next("--anomaly-threshold").c_str());
        else if (a == "--flash-attn")        o.no_flash_attn = false;
        else if (a == "--no-flash-attn")     o.no_flash_attn = true;
        else if (a == "--headless")          o.headless = true;
        else if (a == "-h" || a == "--help") o.help = true;
        else                                 pos.push_back(a);
    }
    if (!pos.empty()) o.model  = pos[0];
    if (pos.size() > 1) o.prompt = pos[1];
    return o;
}

// Redirect llama.cpp's verbose loader logging away from the TUI screen.
void quiet_log(ggml_log_level level, const char* text, void* /*user*/) {
    if (level >= GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
}

} // namespace

// ---------------------------------------------------------------------------
// Consumer: drains the ring and fans events out to topology / anomaly /
// recorder / UI. Runs until `alive` goes false (the App has quit).
// ---------------------------------------------------------------------------
static void consumer_loop(ts::EventRing& ring,
                          ts::UiState& ui,
                          ts::App* app,
                          ts::Topology& topo,
                          ts::AnomalyDetector& detector,
                          ts::Recorder* recorder,
                          ts::AttentionSink& attn_sink,
                          std::atomic<int>& target_layer,
                          std::atomic<int>& pending_select,
                          std::atomic<bool>& alive) {
    while (alive.load(std::memory_order_acquire)) {
        // Apply a pending capture-target selection (from the UI thread).
        int ps = pending_select.exchange(-2, std::memory_order_acq_rel);
        if (ps != -2) {
            topo.select(ps);
            target_layer.store(ps, std::memory_order_release);
            attn_sink.target_layer.store(ps, std::memory_order_release);
        }

        bool any = false;
        ts::TensorEvent ev;
        while (ring.pop(ev)) {
            any = true;
            topo.update(ev);
            ui.push_event(ev);

            if (ev.layer_idx == target_layer.load(std::memory_order_acquire)) {
                ui.set_selected_metrics(ev);
            }
            if (auto a = detector.inspect(ev)) {
                ui.push_anomaly({a->timestamp_ns, a->severity, a->text});
            }
            if (recorder) recorder->write(ev);
        }

        // Publish the latest captured attention matrix, if any.
        ts::AttentionPayload ap;
        if (attn_sink.take(ap)) ui.set_attention(ap);

        if (any) {
            ui.set_topology(topo.flatten());
            ui.set_dropped(ring.dropped());
            if (app) app->request_redraw();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 fps
    }
}

// ---------------------------------------------------------------------------
// Live producer: load model, attach the hook, generate tokens.
// ---------------------------------------------------------------------------
// Sets a flag true on scope exit so the headless drainer knows the producer
// has finished and the ring can be drained to empty.
struct DoneGuard {
    std::atomic<bool>& done;
    ~DoneGuard() { done.store(true, std::memory_order_release); }
};

static void inference_loop(const Options& opt,
                           ts::EventRing& ring,
                           ts::UiState& ui,
                           ts::AttentionSink* attn_sink,
                           std::atomic<bool>& alive,
                           std::atomic<bool>& done) {
    DoneGuard guard{done};
    ts::Capturer capturer(ring);
    capturer.set_attention_sink(attn_sink);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(opt.model.c_str(), mparams);
    if (!model) {
        ui.set_session_state("LOAD FAILED");
        return;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = 2048;
    cparams.cb_eval           = &ts::Capturer::on_eval;
    cparams.cb_eval_user_data = &capturer;
    cparams.flash_attn_type   = opt.no_flash_attn ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                                                  : LLAMA_FLASH_ATTN_TYPE_AUTO;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        ui.set_session_state("CTX FAILED");
        llama_model_free(model);
        return;
    }

    // Tokenize the prompt.
    std::vector<llama_token> tokens(256);
    int n = llama_tokenize(vocab, opt.prompt.c_str(), (int)opt.prompt.size(),
                           tokens.data(), (int)tokens.size(), true, false);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, opt.prompt.c_str(), (int)opt.prompt.size(),
                           tokens.data(), (int)tokens.size(), true, false);
    }
    tokens.resize(n > 0 ? n : 0);
    ui.set_n_tokens(n);

    llama_sampler* smpl = llama_sampler_init_greedy();

    // Prompt prefill.
    if (n > 0 && alive.load()) {
        capturer.prime();
        llama_batch batch = llama_batch_get_one(tokens.data(), n);
        llama_decode(ctx, batch);
    }

    // Autoregressive generation.
    int generated = 0;
    while (alive.load(std::memory_order_acquire) && generated < opt.max_tokens) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        ++generated;
        ui.set_n_tokens(n + generated);

        capturer.prime();
        llama_batch b = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, b) != 0) break;

        if (opt.delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(opt.delay_ms));
        }
    }

    ui.set_session_state("DONE");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
}

// ---------------------------------------------------------------------------
// Replay producer: feed a recorded session through the ring.
// ---------------------------------------------------------------------------
static void replay_loop(const Options& opt,
                        ts::EventRing& ring,
                        ts::UiState& ui,
                        std::atomic<bool>& alive,
                        std::atomic<bool>& done) {
    DoneGuard guard{done};
    ts::Replay r(opt.replay);
    if (!r.ok()) { ui.set_session_state("REPLAY OPEN FAILED"); return; }

    ts::TensorEvent ev;
    while (alive.load(std::memory_order_acquire) && r.next(ev)) {
        while (!ring.push(ev) && alive.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // backpressure
        }
        if (opt.delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(opt.delay_ms));
        }
    }
    ui.set_session_state("REPLAY DONE");
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

    if (opt.help) { print_usage(stdout); return 0; }

    const bool replay_mode = !opt.replay.empty();
    if (opt.model.empty() && !replay_mode) {
        print_usage(stderr);
        return 1;
    }

    // Pre-flight: fail fast with a clear message if the model file is missing,
    // rather than surfacing it as a "LOAD FAILED" badge inside the TUI.
    if (!replay_mode) {
        if (std::FILE* f = std::fopen(opt.model.c_str(), "rb")) {
            std::fclose(f);
        } else {
            std::fprintf(stderr, "error: model file not found: %s\n", opt.model.c_str());
            return 1;
        }
    }

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    ts::EventRing       ring(1u << 15);
    ts::UiState         ui;
    ts::Topology        topo;
    ts::AnomalyDetector detector(ts::AnomalyDetector::Config{opt.anomaly_thresh, false});

    std::unique_ptr<ts::Recorder> recorder;
    if (!opt.record.empty()) {
        recorder = std::make_unique<ts::Recorder>(opt.record);
        if (!recorder->ok()) {
            std::fprintf(stderr, "warning: could not open record file '%s'\n", opt.record.c_str());
            recorder.reset();
        }
    }

    ui.set_model_name(replay_mode ? opt.replay : opt.model);
    ui.set_session_state(replay_mode ? "REPLAY" : "LIVE");

    ts::AttentionSink attn_sink;
    // Default the attention target to layer 0 so the prompt's self-attention
    // matrix is captured during prefill, before the user selects anything.
    attn_sink.target_layer.store(0, std::memory_order_release);

    std::atomic<bool> alive{true};
    std::atomic<bool> producer_done{false};
    std::atomic<int>  target_layer{0};
    std::atomic<int>  pending_select{-2};

    auto start_producer = [&]() {
        if (replay_mode) {
            return std::thread(replay_loop, std::cref(opt), std::ref(ring), std::ref(ui),
                               std::ref(alive), std::ref(producer_done));
        }
        return std::thread(inference_loop, std::cref(opt), std::ref(ring), std::ref(ui),
                           &attn_sink, std::ref(alive), std::ref(producer_done));
    };

    // ---- Headless mode: run to completion, no TUI (verification / CI) -------
    if (opt.headless) {
        std::thread producer = start_producer();

        uint64_t total = 0, anomalies = 0;
        bool got_attention = false;
        int  att_rows = 0, att_cols = 0;
        ts::TensorEvent ev;
        while (!producer_done.load(std::memory_order_acquire) || ring.size_approx() > 0) {
            while (ring.pop(ev)) {
                ++total;
                topo.update(ev);
                if (auto a = detector.inspect(ev)) ++anomalies;
                if (recorder) recorder->write(ev);
            }
            ts::AttentionPayload ap;
            if (attn_sink.take(ap)) { got_attention = true; att_rows = ap.rows; att_cols = ap.cols; }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        producer.join();

        auto layers = topo.flatten();
        std::printf("Local_LLM_Instrumentation headless run complete.\n");
        std::printf("  tensor ops captured : %llu\n", (unsigned long long)total);
        std::printf("  topology rows       : %zu\n", layers.size());
        std::printf("  anomalies flagged   : %llu\n", (unsigned long long)anomalies);
        if (got_attention) std::printf("  attention captured  : yes (%dx%d)\n", att_rows, att_cols);
        else               std::printf("  attention captured  : no\n");
        std::printf("  events dropped      : %zu\n", ring.dropped());
        if (recorder) { recorder->close(); std::printf("  recorded to         : %s\n", opt.record.c_str()); }
        llama_backend_free();
        return 0;
    }

    // ---- Interactive mode: TUI on the main thread ---------------------------
    ts::App app(ui);

    // Space on a topology row → resolve that row's layer_idx from the snapshot
    // and hand it to the consumer (which owns Topology, single-threaded).
    app.on_select_target = [&](int row) {
        auto snap = ui.snapshot();
        if (row >= 0 && row < (int)snap.topology.size()) {
            pending_select.store(snap.topology[row].layer_idx, std::memory_order_release);
        }
    };

    std::thread consumer(consumer_loop, std::ref(ring), std::ref(ui), &app,
                         std::ref(topo), std::ref(detector), recorder.get(),
                         std::ref(attn_sink), std::ref(target_layer),
                         std::ref(pending_select), std::ref(alive));

    std::thread producer = start_producer();

    int rc = app.run();          // blocks on the FTXUI loop until the user quits

    alive.store(false, std::memory_order_release);
    if (producer.joinable()) producer.join();
    if (consumer.joinable()) consumer.join();

    if (recorder) recorder->close();
    llama_backend_free();
    return rc;
}
