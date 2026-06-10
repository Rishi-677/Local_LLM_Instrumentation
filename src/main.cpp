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
#include "config.hpp"
#include "log.hpp"
#include "capture/capturer.hpp"
#include "capture/topology.hpp"
#include "anomaly.hpp"
#include "session/recorder.hpp"
#include "session/replay.hpp"
#include "export.hpp"
#include "tui/ui_state.hpp"
#include "tui/app.hpp"

#include <array>
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
    bool  preview        = false;// render one TUI frame to stdout and exit
    std::string preview_filter;  // optional stream filter for --preview
    bool  bench          = false;// measure hook overhead (hooked vs baseline)
    std::string export_csv;      // headless: write per-layer summary CSV
    std::string export_json;     // headless: write per-layer summary JSON
    bool  help           = false;

    // Sourced from config (TOML) as a baseline, overridable on the CLI.
    std::string config_path;
    bool        print_config = false;
    size_t      ring_capacity = 32768;
    int         attn_layer = 0;
    int         attn_head  = 0;
    bool        flag_cpu_fallback = false;
    int         stats_sample = 8192;
    // Per-OpClass capture mask, indexed by static_cast<int>(OpClass):
    // {Embed, Attn, MLP, Norm, Output, Other}.
    std::array<bool, 6> capture_mask = {true, true, true, true, true, true};
};

void print_usage(std::FILE* f) {
    std::fprintf(f,
        "Local_LLM_Instrumentation - terminal-native live debugger for local LLMs\n\n"
        "usage:\n"
        "  local_llm_instrumentation <model.gguf> [prompt] [options]\n"
        "  local_llm_instrumentation --replay <session.ndjson> [options]\n\n"
        "options:\n"
        "  --config <file>          load settings from a TOML config (CLI overrides)\n"
        "  --print-config           print the effective config and exit\n"
        "  --ring-capacity <n>      telemetry ring buffer capacity (events)\n"
        "  --record <file>          record telemetry to NDJSON\n"
        "  --replay <file>          replay a recorded session (no model)\n"
        "  --max-tokens <n>         tokens to generate (default 64)\n"
        "  --delay-ms <n>           pace tokens/events so the run is watchable\n"
        "  --anomaly-threshold <x>  |value| above x flags an outlier (default 1e4)\n"
        "  --no-flash-attn          disable flash attention (default; exposes attention)\n"
        "  --flash-attn             allow auto flash attention (hides attention matrix)\n"
        "  --headless               run to completion without the TUI (verify / CI)\n"
        "  --preview                render one TUI frame to stdout and exit\n"
        "  --bench                  measure hook overhead (hooked vs baseline) and exit\n"
        "  --export-csv <file>      write a per-layer stats summary as CSV (headless)\n"
        "  --export-json <file>     write a per-layer stats summary as JSON (headless)\n"
        "  -h, --help               show this help\n");
}

// Apply CLI flags over an Options already seeded from config (CLI wins).
void apply_cli(int argc, char** argv, Options& o) {
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); return ""; }
            return argv[++i];
        };
        if      (a == "--config")            o.config_path = next("--config");
        else if (a == "--print-config")      o.print_config = true;
        else if (a == "--record")            o.record = next("--record");
        else if (a == "--replay")            o.replay = next("--replay");
        else if (a == "--max-tokens")        o.max_tokens = std::atoi(next("--max-tokens").c_str());
        else if (a == "--delay-ms")          o.delay_ms = std::atoi(next("--delay-ms").c_str());
        else if (a == "--ring-capacity")     o.ring_capacity = (size_t)std::strtoull(next("--ring-capacity").c_str(), nullptr, 10);
        else if (a == "--anomaly-threshold") o.anomaly_thresh = (float)std::atof(next("--anomaly-threshold").c_str());
        else if (a == "--flash-attn")        o.no_flash_attn = false;
        else if (a == "--no-flash-attn")     o.no_flash_attn = true;
        else if (a == "--headless")          o.headless = true;
        else if (a == "--preview")           o.preview = true;
        else if (a == "--preview-filter")    { o.preview = true; o.preview_filter = next("--preview-filter"); }
        else if (a == "--bench")             o.bench = true;
        else if (a == "--export-csv")        { o.headless = true; o.export_csv = next("--export-csv"); }
        else if (a == "--export-json")       { o.headless = true; o.export_json = next("--export-json"); }
        else if (a == "-h" || a == "--help") o.help = true;
        else                                 pos.push_back(a);
    }
    if (!pos.empty()) o.model  = pos[0];
    if (pos.size() > 1) o.prompt = pos[1];
}

// Seed Options defaults from a loaded Config (before CLI overrides).
void seed_from_config(Options& o, const ts::Config& c) {
    o.max_tokens       = c.max_tokens;
    o.delay_ms         = c.delay_ms;
    o.anomaly_thresh   = c.anomaly_threshold;
    o.no_flash_attn    = c.no_flash_attn;
    o.ring_capacity    = c.ring_capacity;
    o.attn_layer       = c.attention_layer;
    o.attn_head        = c.attention_head;
    o.flag_cpu_fallback = c.flag_cpu_fallback;
    o.stats_sample      = c.stats_sample;
    o.capture_mask = { c.capture_embed, c.capture_attn, c.capture_mlp,
                       c.capture_norm, /*Output*/ true, c.capture_other };
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
    capturer.set_capture_mask(opt.capture_mask);
    capturer.set_stats_sample(opt.stats_sample);

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

// ---------------------------------------------------------------------------
// Benchmark: time generation throughput with and without the cb_eval hook to
// quantify capture overhead. Surfaces the Phase-2 "keep the hot path cheap"
// target. No TUI, no threads.
// ---------------------------------------------------------------------------
static int run_bench(const Options& opt) {
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(opt.model.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "error: failed to load model: %s\n", opt.model.c_str());
        llama_backend_free();
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens(256);
    int n = llama_tokenize(vocab, opt.prompt.c_str(), (int)opt.prompt.size(),
                           tokens.data(), (int)tokens.size(), true, false);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, opt.prompt.c_str(), (int)opt.prompt.size(),
                           tokens.data(), (int)tokens.size(), true, false);
    }
    tokens.resize(n > 0 ? n : 0);

    // Time the generation of opt.max_tokens tokens (prompt prefill excluded so we
    // measure steady-state per-token cost). Returns tokens/sec and op count.
    auto time_run = [&](bool hook, uint64_t* ops_out) -> double {
        ts::EventRing ring(opt.ring_capacity);
        ts::Capturer  capturer(ring);
        capturer.set_stats_sample(opt.stats_sample);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 2048;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        if (hook) {
            cp.cb_eval = &ts::Capturer::on_eval;
            cp.cb_eval_user_data = &capturer;
        }
        llama_context* ctx = llama_init_from_model(model, cp);
        llama_sampler* smpl = llama_sampler_init_greedy();

        llama_batch pb = llama_batch_get_one(tokens.data(), n);
        llama_decode(ctx, pb);  // prefill (not timed)

        const auto t0 = std::chrono::steady_clock::now();
        int gen = 0;
        for (; gen < opt.max_tokens; ++gen) {
            llama_token tok = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, tok)) break;
            llama_batch b = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, b) != 0) break;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();

        // Drain whatever the hook produced so we can report op throughput.
        if (ops_out) {
            uint64_t c = 0; ts::TensorEvent ev;
            while (ring.pop(ev)) ++c;
            *ops_out = c + ring.dropped();
        }
        llama_sampler_free(smpl);
        llama_free(ctx);
        return gen > 0 && secs > 0 ? gen / secs : 0.0;
    };

    uint64_t dummy = 0, ops = 0;
    const double base   = time_run(false, &dummy);  // baseline, no hook
    const double hooked = time_run(true,  &ops);    // hooked

    const double overhead = base > 0 ? (base - hooked) / base * 100.0 : 0.0;
    std::printf("Local_LLM_Instrumentation hook overhead benchmark (%d gen tokens)\n", opt.max_tokens);
    std::printf("  baseline (no hook) : %.1f tok/s\n", base);
    std::printf("  hooked             : %.1f tok/s\n", hooked);
    std::printf("  overhead           : %.1f%%\n", overhead);
    std::printf("  tensor ops/token   : ~%llu\n",
                (unsigned long long)(opt.max_tokens ? ops / (unsigned)opt.max_tokens : ops));
    std::printf("  note: most overhead is structural — cb_eval runs the graph\n"
                "        node-by-node (no backend fusion). Mask op classes or\n"
                "        narrow the capture to reduce observed nodes.\n");

    llama_model_free(model);
    llama_backend_free();
    return 0;
}

int main(int argc, char** argv) {
    Options opt;

    // Pre-scan for --config so the file can seed defaults before CLI overrides.
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--config") opt.config_path = argv[i + 1];

    std::string cfg_err;
    ts::Config cfg = opt.config_path.empty()
                         ? ts::Config::defaults()
                         : ts::Config::load(opt.config_path, &cfg_err);
    seed_from_config(opt, cfg);
    apply_cli(argc, argv, opt);

    if (opt.help) { print_usage(stdout); return 0; }
    if (opt.print_config) { std::fputs(cfg.to_string().c_str(), stdout); return 0; }

    ts::log::init();
    if (!cfg_err.empty())
        spdlog::warn("config load error ({}): {}", opt.config_path, cfg_err);
    spdlog::info("Local_LLM_Instrumentation starting (theme={}, ring_capacity={})",
                 cfg.theme, opt.ring_capacity);

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

    if (opt.bench) return run_bench(opt);

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    ts::EventRing       ring(opt.ring_capacity);
    ts::UiState         ui;
    ts::Topology        topo;
    ts::AnomalyDetector detector(
        ts::AnomalyDetector::Config{opt.anomaly_thresh, opt.flag_cpu_fallback});

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
    // Default the attention target (from config) so the prompt's self-attention
    // matrix is captured during prefill, before the user selects anything.
    attn_sink.target_layer.store(opt.attn_layer, std::memory_order_release);
    attn_sink.target_head.store(opt.attn_head, std::memory_order_release);

    std::atomic<bool> alive{true};
    std::atomic<bool> producer_done{false};
    std::atomic<int>  target_layer{opt.attn_layer};
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
        ts::Summary summary;
        const bool want_summary = !opt.export_csv.empty() || !opt.export_json.empty();
        ts::TensorEvent ev;
        while (!producer_done.load(std::memory_order_acquire) || ring.size_approx() > 0) {
            while (ring.pop(ev)) {
                ++total;
                topo.update(ev);
                if (auto a = detector.inspect(ev)) ++anomalies;
                if (recorder) recorder->write(ev);
                if (want_summary) summary.add(ev);
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
        if (!opt.export_csv.empty()) {
            bool ok = summary.write_csv(opt.export_csv);
            std::printf("  exported CSV        : %s (%zu rows)%s\n", opt.export_csv.c_str(),
                        summary.rows(), ok ? "" : " [FAILED]");
        }
        if (!opt.export_json.empty()) {
            bool ok = summary.write_json(opt.export_json);
            std::printf("  exported JSON       : %s (%zu rows)%s\n", opt.export_json.c_str(),
                        summary.rows(), ok ? "" : " [FAILED]");
        }
        llama_backend_free();
        return 0;
    }

    // ---- Preview mode: populate the UI, render ONE frame, print, exit -------
    if (opt.preview) {
        std::thread producer = start_producer();
        ts::TensorEvent ev;
        while (!producer_done.load(std::memory_order_acquire) || ring.size_approx() > 0) {
            while (ring.pop(ev)) {
                topo.update(ev);
                ui.push_event(ev);
                if (ev.layer_idx == 0) ui.set_selected_metrics(ev);
                if (auto an = detector.inspect(ev))
                    ui.push_anomaly({an->timestamp_ns, an->severity, an->text});
            }
            ts::AttentionPayload ap;
            if (attn_sink.take(ap)) ui.set_attention(ap);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        producer.join();
        topo.select(0);
        ui.set_topology(topo.flatten());
        ui.set_session_state("LIVE");

        ts::App app(ui);
        if (!opt.preview_filter.empty()) app.set_stream_filter(opt.preview_filter);
        std::string frame = app.preview(150, 46);
        std::fwrite(frame.data(), 1, frame.size(), stdout);
        std::printf("\n");
        if (recorder) recorder->close();
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
