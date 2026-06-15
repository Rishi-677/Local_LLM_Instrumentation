// Local_LLM_Instrumentation — FTXUI TUI shell implementation (C-TUI).
// All six panes live here to keep the public surface (app.hpp) minimal. Each
// pane is a free function taking the per-frame Snapshot plus whatever local
// view state it needs; App::run() wires them into a Container with a single
// CatchEvent that owns global keys (Tab/q) and routes directional keys to the
// focused pane.
#include "app.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "ftxui/component/component.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/dom/node.hpp"
#include "ftxui/screen/color.hpp"
#include "ftxui/screen/screen.hpp"
#include "latency_flamegraph.hpp"

namespace ts {
using namespace ftxui;
namespace {

// Warm gruvbox-ish palette to match the reference dashboard.
const Color C_ACCENT = Color::RGB(234, 147, 130);  // salmon — primary data
const Color C_TEXT = Color::RGB(213, 196, 161);    // cream — normal text
const Color C_HEAD = Color::RGB(235, 219, 178);    // bright cream — headers
const Color C_KEY = Color::RGB(250, 189, 47);      // yellow — keys / focus
const Color C_DIM = Color::RGB(124, 111, 100);     // dim gray
const Color C_BORDER = Color::RGB(102, 92, 84);    // border gray
const Color C_AQUA = Color::RGB(142, 192, 124);    // green
const Color C_ORANGE = Color::RGB(254, 128, 25);   // orange
const Color C_PURPLE = Color::RGB(211, 134, 155);  // purple
const Color C_RED = Color::RGB(251, 73, 52);       // red
const Color C_SELBG = Color::RGB(80, 73, 69);      // selection background

// Pane identity — index used by the focus cycle.
enum Pane : int {
    P_TOPOLOGY = 0,
    P_STREAM = 1,
    P_ATTENTION = 2,
    P_METRICS = 3,
    P_FLAME = 4,
    P_LEDGER = 5,
    P_COUNT = 6,
};

// A keycap chip like [Tab] in the header / hints.
Element keycap(const std::string& k) {
    return text("[" + k + "]") | color(C_KEY) | bold;
}

// Format a steady-clock ns timestamp as hh:mm:ss.mmm (wall-clock-ish; the
// absolute epoch is irrelevant for a live debugger, the cadence is what matters).
std::string fmt_time(uint64_t ts_ns) {
    const uint64_t total_ms = ts_ns / 1'000'000ull;
    const uint64_t ms = total_ms % 1000;
    const uint64_t total_s = total_ms / 1000;
    const uint64_t s = total_s % 60;
    const uint64_t m = (total_s / 60) % 60;
    const uint64_t h = (total_s / 3600) % 24;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02llu:%02llu:%02llu.%03llu", (unsigned long long)h,
                  (unsigned long long)m, (unsigned long long)s, (unsigned long long)ms);
    return std::string(buf);
}

std::string shape_to_string(const std::array<int64_t, kMaxDims>& shape) {
    std::string out = "[";
    bool first = true;
    for (int i = 0; i < kMaxDims; ++i) {
        if (shape[i] == 0 && i > 0)
            continue;  // trim trailing zero dims
        if (!first)
            out += ", ";
        out += std::to_string(shape[i]);
        first = false;
    }
    out += "]";
    return out;
}

// A few ggml_type values mapped to short names; falls back to the raw enum.
std::string dtype_name(int32_t dtype) {
    switch (dtype) {
        case 0:
            return "f32";
        case 1:
            return "f16";
        case 2:
            return "q4_0";
        case 3:
            return "q4_1";
        case 8:
            return "q8_0";
        case 24:
            return "i8";
        case 25:
            return "i16";
        case 26:
            return "i32";
        case 30:
            return "bf16";
        default:
            return "t" + std::to_string(dtype);
    }
}

// Wrap a body in a square-cornered window whose title is embedded in the top
// border, like the reference layout. Focus is shown by a yellow ▌ + label.
Element pane(int num, const std::string& name, bool focused, Element body) {
    const std::string t = " " + std::to_string(num) + ". " + name + " ";
    Element title = focused ? hbox({text("▌") | color(C_KEY), text(t) | bold | color(C_HEAD),
                                    text("(Focus Active) ") | color(C_KEY) | bold})
                            : hbox({text(t) | bold | color(C_HEAD)});
    return window(title, body, LIGHT) | color(C_BORDER) | flex;
}

// A layer bucket (vs. an op-class leaf) is named lowercase: "layers.N",
// "embed", "output", "other". Op-class leaves are Capitalized ("Attn", ...).
bool is_layer_row(const std::string& label) {
    return label.rfind("layers.", 0) == 0 || label == "embed" || label == "output" ||
           label == "other";
}

// Accumulated latency as ms (>=1ms) or µs, for the topology / metrics views.
std::string fmt_latency(uint64_t us) {
    char b[24];
    if (us >= 1000)
        std::snprintf(b, sizeof(b), "%.1f ms", us / 1000.0);
    else
        std::snprintf(b, sizeof(b), "%llu µs", (unsigned long long)us);
    return b;
}

// Pane 1: MODEL TOPOLOGY
Element render_topology(const UiState::Snapshot& s, bool /*focused*/) {
    Elements rows;
    rows.push_back(hbox({text("▼ ") | color(C_KEY), text("model") | bold | color(C_HEAD)}));
    if (s.topology.empty()) {
        rows.push_back(text("   (awaiting forward pass…)") | color(C_DIM));
    }

    for (int i = 0; i < static_cast<int>(s.topology.size()); ++i) {
        const LayerNode& n = s.topology[i];
        const bool cursor = (i == s.selected);
        const bool layer = is_layer_row(n.label);
        Elements parts;
        if (layer) {
            const char* glyph = n.selected ? "▼ " : "▶ ";
            Element lbl = text(n.label) | color(n.selected ? C_KEY : C_TEXT);
            if (n.selected)
                lbl = lbl | bold;
            parts = {text("  "), text(glyph) | color(C_KEY), lbl};
            if (n.selected)
                parts.push_back(text("  [Active Capture Target]") | color(C_AQUA));
        } else {
            parts = {text("     ● ") | color(C_DIM), text(n.label) | color(C_ACCENT)};
        }

        // Right-aligned accumulated latency so the dominant block stands out.
        if (n.total_latency_us > 0) {
            parts.push_back(filler());
            parts.push_back(text(fmt_latency(n.total_latency_us) + " ") | color(C_DIM));
        }
        Element row = hbox(std::move(parts));
        if (cursor)
            row = row | bgcolor(C_SELBG);
        rows.push_back(row);
    }
    return vbox(std::move(rows)) | yframe | flex;
}

// Map an op class to a friendly "layer type" like the reference.
std::string layer_type(const TensorEvent& ev) {
    switch (ev.op_class) {
        case OpClass::Attn:
            return "Attn (Self)";
        case OpClass::MLP:
            return "MLP (SwiGLU)";
        case OpClass::Norm:
            return "LayerNorm";
        case OpClass::Embed:
            return "Embedding";
        case OpClass::Output:
            return "Output";
        default:
            return std::string(ev.op_name);
    }
}

std::string device_label(Device d) {
    switch (d) {
        case Device::CUDA:
            return "CUDA [GPU 0]";
        case Device::Metal:
            return "Metal [GPU]";
        case Device::CPU:
            return "CPU";
        default:
            return "—";
    }
}

std::string to_lower_copy(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Pane 2: LIVE PACKET STREAM
Element render_stream(const UiState::Snapshot& s, const std::string& filter, bool editing) {
    auto sep = [] { return separator() | color(C_BORDER); };
    const std::string needle = to_lower_copy(filter);
    Elements rows;
    rows.push_back(hbox({
        text("ID") | size(WIDTH, EQUAL, 7) | bold | color(C_HEAD),
        sep(),
        text("TIMESTAMP") | size(WIDTH, EQUAL, 14) | bold | color(C_HEAD),
        sep(),
        text("LAYER TYPE") | size(WIDTH, EQUAL, 16) | bold | color(C_HEAD),
        sep(),
        text("COMPUTE DEVICE") | bold | color(C_HEAD),
    }));
    rows.push_back(separator() | color(C_BORDER));

    int shown = 0, total = 0;
    // Newest at bottom; frame keeps the latest in view.
    for (const TensorEvent& ev : s.stream) {
        ++total;
        const std::string lt = layer_type(ev);
        if (!needle.empty()) {
            std::string hay =
                to_lower_copy(lt + " " + std::string(ev.op_name) + " " + device_label(ev.device) +
                              " " + std::string(ev.node_name));
            if (hay.find(needle) == std::string::npos)
                continue;
        }
        ++shown;
        const bool cpu = (ev.device != Device::CUDA && ev.device != Device::Metal);
        Element devc = text(device_label(ev.device)) | color(cpu ? C_KEY : C_AQUA);
        Element row = hbox({
            text(std::to_string(ev.id)) | size(WIDTH, EQUAL, 7) | color(C_ACCENT),
            sep(),
            text(fmt_time(ev.timestamp_ns)) | size(WIDTH, EQUAL, 14) | color(C_ACCENT),
            sep(),
            text(lt) | size(WIDTH, EQUAL, 16) | color(C_TEXT),
            sep(),
            devc,
        });
        if (ev.has_nan || ev.has_inf)
            row = row | color(C_RED) | bold;
        rows.push_back(row);
    }

    Element body = vbox(std::move(rows)) | yframe | flex;
    // Filter line: editing prompt, or a persistent active-filter indicator.
    if (editing) {
        Element f = hbox({text("/") | color(C_KEY) | bold, text(filter) | color(C_HEAD),
                          text("▌") | color(C_KEY), filler(),
                          text("[Enter] apply  [Esc] clear ") | color(C_DIM)});
        return vbox({body, separator() | color(C_BORDER), f});
    }
    if (!filter.empty()) {
        Element f = hbox(
            {text("filter: ") | color(C_DIM), text(filter) | color(C_KEY) | bold,
             text("  (" + std::to_string(shown) + "/" + std::to_string(total) + ")") | color(C_DIM),
             filler(), text("[/] edit  [Esc] clear ") | color(C_DIM)});
        return vbox({body, separator() | color(C_BORDER), f});
    }
    Element hint = hbox({filler(), text("[/] filter ") | color(C_DIM)});
    return vbox({body, hint});
}

// Pane 3: ATTENTION MATRIX VISUALIZER
Element render_attention(const UiState::Snapshot& s, int pan_row, int pan_col, float contrast) {
    if (!s.has_attention || s.attention.weights.empty() || s.attention.rows <= 0 ||
        s.attention.cols <= 0) {
        return vbox({
                   text("No attention captured — select a layer and run "
                        "with --flash-attn off") |
                       dim | hcenter,
               }) |
               flex | center;
    }

    const AttentionPayload& a = s.attention;
    const int rows = a.rows;
    const int cols = a.cols;
    // Normalize against the observed max so the heatmap uses its full range.
    float wmax = 0.0f;
    for (float w : a.weights)
        wmax = std::max(wmax, std::fabs(w));
    if (wmax <= 0.0f)
        wmax = 1.0f;

    static const char* kRamp[] = {"··", "░░", "▒▒", "▓▓", "██"};
    constexpr int kRampN = 5;
    // Viewport: cap to a sensible size; pan offsets clamp inside the matrix.
    const int vp_rows = std::min(rows, 16);
    const int vp_cols = std::min(cols, 40);
    const int r0 = std::clamp(pan_row, 0, std::max(0, rows - vp_rows));
    const int c0 = std::clamp(pan_col, 0, std::max(0, cols - vp_cols));

    Elements grid_lines;
    for (int r = r0; r < r0 + vp_rows && r < rows; ++r) {
        Elements cells;
        cells.push_back(text("q" + std::to_string(r)) | size(WIDTH, EQUAL, 4) | color(C_DIM));
        for (int c = c0; c < c0 + vp_cols && c < cols; ++c) {
            float v = std::fabs(a.weights[static_cast<size_t>(r) * cols + c]) / wmax;
            v = std::clamp(v * contrast, 0.0f, 1.0f);
            int level = std::clamp(static_cast<int>(v * (kRampN - 1) + 0.5f), 0, kRampN - 1);
            Color fg = C_BORDER;
            if (level >= 4)
                fg = C_HEAD;
            else if (level == 3)
                fg = C_ACCENT;
            else if (level == 2)
                fg = C_ORANGE;
            else if (level == 1)
                fg = C_DIM;
            cells.push_back(text(kRamp[level]) | color(fg));
        }
        grid_lines.push_back(hbox(std::move(cells)));
    }

    char cbuf[8];
    std::snprintf(cbuf, sizeof(cbuf), "%.2f", contrast);
    Element hints = vbox({
        text("head " + std::to_string(a.head) + " · layer " + std::to_string(a.layer_idx)) |
            color(C_HEAD) | bold,
        text(std::to_string(rows) + " queries × " + std::to_string(cols) + " keys") | color(C_DIM),
        text("view  q" + std::to_string(r0) + "+ k" + std::to_string(c0)) | color(C_DIM),
        text("contrast x" + std::string(cbuf)) | color(C_DIM),
        text(""),
        hbox({keycap("h/j/k/l"), text(" Pan matrix") | color(C_TEXT)}),
        hbox({keycap("+/-"), text(" Weight contrast") | color(C_TEXT)}),
        hbox({keycap("f"), text(" Fit / reset view") | color(C_TEXT)}),
    });

    return hbox({
               vbox(std::move(grid_lines)) | flex,
               separator() | color(C_BORDER),
               hints | size(WIDTH, EQUAL, 26),
           }) |
           flex;
}

// A label: value row, label in cream, value in salmon.
Element kv(const std::string& k, Element v) {
    return hbox({text(k) | color(C_HEAD) | bold, v});
}

// Discrete colored-square sparsity bar: green fill, orange boundary, purple rest.
Element sparsity_bar(float frac) {
    frac = std::clamp(frac, 0.0f, 1.0f);
    const int N = 14;
    const int filled = static_cast<int>(frac * N + 0.5f);
    Elements sq;
    for (int i = 0; i < N; ++i) {
        Color c = C_PURPLE;
        if (i < filled)
            c = C_AQUA;
        else if (i == filled && filled < N)
            c = C_ORANGE;
        sq.push_back(text("■") | color(c));
    }
    char pct[8];
    std::snprintf(pct, sizeof(pct), "%.1f%%", frac * 100.0f);
    sq.push_back(text(" " + std::string(pct)) | color(C_ACCENT));
    return hbox(std::move(sq));
}

// Pane 4: RUNTIME METRICS INSPECTOR
Element render_metrics(const UiState::Snapshot& s) {
    if (!s.has_metrics) {
        return vbox({text(" (no metrics for selected layer yet) ") | color(C_DIM)}) | flex;
    }
    const TensorEvent& m = s.selected_metrics;
    Element nan_inf = text("clean") | color(C_AQUA);
    if (m.has_nan || m.has_inf) {
        std::string f;
        if (m.has_nan)
            f += "NaN ";
        if (m.has_inf)
            f += "Inf";
        nan_inf = text(f) | color(C_RED) | bold;
    }

    return vbox({
               kv("Tensor Shape : ", hbox({text(shape_to_string(m.shape)) | color(C_ACCENT),
                                           text("   Dtype: ") | color(C_HEAD) | bold,
                                           text(dtype_name(m.dtype)) | color(C_ACCENT)})),
               kv("Node         : ", text(std::string(m.node_name) + "  (" +
                                          std::string(to_string(m.op_class)) + ")") |
                                         color(C_ACCENT)),
               kv("Sparsity Rate: ", sparsity_bar(m.sparsity)),
               kv("Latency Delta: ", text(std::to_string(m.latency_us) + " µs") | color(C_ACCENT)),
               kv("Value Range  : ",
                  text("[" + std::to_string(m.v_min) + ", " + std::to_string(m.v_max) + "]  mean " +
                       std::to_string(m.v_mean)) |
                      color(C_ACCENT)),
               kv("Std / L2 Norm: ",
                  text(std::to_string(m.v_std) + "  /  " + std::to_string(m.l2_norm)) |
                      color(C_ACCENT)),
               kv("NaN / Inf    : ", nan_inf),
           }) |
           flex;
}

// Pane 5: LATENCY FLAMEGRAPH
Element render_flamegraph(const UiState::Snapshot& s) {
    const std::vector<LatencyFlameRow> rows = build_latency_flamegraph(s.topology, 10);
    if (rows.empty()) {
        return vbox({text(" (no latency samples yet) ") | color(C_DIM)}) | flex;
    }
    const uint64_t max_latency = rows.front().total_latency_us;
    const uint64_t total = latency_flamegraph_total(rows);

    Elements lines;
    for (const LatencyFlameRow& row : rows) {
        char pct[16];
        const double share =
            total ? (static_cast<double>(row.total_latency_us) * 100.0 / static_cast<double>(total))
                  : 0.0;
        std::snprintf(pct, sizeof(pct), "%5.1f%%", share);

        lines.push_back(hbox({
            text(row.label) | size(WIDTH, EQUAL, 12) | color(C_HEAD) | bold,
            text(latency_ascii_bar(row.total_latency_us, max_latency, 24)) | color(C_ACCENT),
            text(" " + fmt_latency(row.total_latency_us)) | size(WIDTH, EQUAL, 9) | color(C_TEXT),
            text(pct) | color(C_DIM),
        }));
    }

    return vbox(std::move(lines)) | yframe | flex;
}

// Pane 6: NUMERICAL ANOMALY LEDGER
Element render_ledger(const UiState::Snapshot& s) {
    Elements rows;
    if (s.anomalies.empty()) {
        rows.push_back(hbox({text("✓ ") | color(C_AQUA) | bold,
                             text("no anomalies — values within bounds") | color(C_AQUA)}));
    }
    // Newest last; frame scrolls to keep recent ones visible.
    for (const Anomaly& a : s.anomalies) {
        const bool err = (a.severity >= 2);
        const char* glyph = err ? "✖" : "⚠";
        Color gc = err ? C_RED : C_ORANGE;
        rows.push_back(hbox({
            text(fmt_time(a.ts_ns)) | size(WIDTH, EQUAL, 14) | color(C_DIM),
            text(" "),
            text(glyph) | color(gc) | bold,
            text(" "),
            text(a.text) | color(err ? C_RED : C_ACCENT),
        }));
    }
    return vbox(std::move(rows)) | yframe | flex;
}

// Compose the full dashboard from a snapshot plus presentation state. Shared by
// the live render loop and the headless preview path.
Element build_frame(const UiState::Snapshot& s, int focus, int pan_row, int pan_col, float contrast,
                    const std::string& filter, bool filter_editing) {
    Color state_col = (s.session_state == "LIVE") ? C_AQUA : C_ORANGE;
    // Model basename only (drop path + extension), capped, to keep the header tight.
    std::string model = s.model_name;
    if (auto p = model.find_last_of("/\\"); p != std::string::npos)
        model = model.substr(p + 1);
    if (auto p = model.rfind(".gguf"); p != std::string::npos)
        model = model.substr(0, p);
    if (model.size() > 22)
        model = model.substr(0, 21) + "…";
    if (model.empty())
        model = "(no model)";

    auto bar = [] { return text(" │ ") | color(C_BORDER); };
    Element header = hbox({
        text(" Local_LLM_Instrumentation ") | bold | color(Color::Black) | bgcolor(C_ORANGE),
        text("  "),
        keycap("Tab"),
        text(" Focus") | color(C_TEXT),
        bar(),
        keycap("j/k"),
        text(" Nav") | color(C_TEXT),
        bar(),
        keycap("Spc"),
        text(" Target") | color(C_TEXT),
        bar(),
        keycap("hjkl"),
        text(" Pan") | color(C_TEXT),
        bar(),
        keycap("±"),
        text(" Contrast") | color(C_TEXT),
        bar(),
        keycap("Q"),
        text(" Quit") | color(C_TEXT),
        bar(),
        keycap("P"),
        text(" Pause") | color(C_TEXT),
        bar(),
        keycap("S"),
        text(" Step") | color(C_TEXT),
        filler(),
        text(model) | color(C_DIM),
        text("  tok:") | color(C_HEAD),
        text(std::to_string(s.n_tokens)) | color(C_ACCENT),
        text(" ev:") | color(C_HEAD),
        text(std::to_string(s.total_events)) | color(C_ACCENT),
        text(" drop:") | color(C_HEAD),
        text(std::to_string(s.dropped)) | color(C_ACCENT),
        text(" "),
        text(" " + s.session_state + " ") | bold | color(Color::Black) | bgcolor(state_col),
    });

    Element topo =
        pane(1, "MODEL TOPOLOGY", focus == P_TOPOLOGY, render_topology(s, focus == P_TOPOLOGY));
    Element stream =
        pane(2, "LIVE PACKET STREAM", focus == P_STREAM, render_stream(s, filter, filter_editing));
    Element attention = pane(3, "ATTENTION MATRIX VISUALIZER", focus == P_ATTENTION,
                             render_attention(s, pan_row, pan_col, contrast));
    Element metrics = pane(4, "RUNTIME METRICS INSPECTOR", focus == P_METRICS, render_metrics(s));
    Element flame = pane(5, "LATENCY FLAMEGRAPH", focus == P_FLAME, render_flamegraph(s));
    Element ledger = pane(6, "NUMERICAL ANOMALY LEDGER", focus == P_LEDGER, render_ledger(s));
    Element top_row = hbox({topo | size(WIDTH, GREATER_THAN, 34) | flex, stream | flex}) | flex;
    Element bottom_row = hbox({metrics | flex, flame | flex, ledger | flex});

    return vbox({
        header,
        separator() | color(C_BORDER),
        top_row,
        attention | size(HEIGHT, GREATER_THAN, 8),
        bottom_row | size(HEIGHT, GREATER_THAN, 9),
    });
}

}  // namespace

App::App(UiState& state) : state_(state) {}
void App::request_redraw() {
    if (screen_)
        screen_->PostEvent(Event::Custom);
}

std::string App::preview(int width, int height) {
    Element doc = build_frame(state_.snapshot(), focus_, att_pan_row_, att_pan_col_, att_contrast_,
                              stream_filter_, filter_editing_);
    auto screen = Screen::Create(Dimension::Fixed(width), Dimension::Fixed(height));
    Render(screen, doc);
    return screen.ToString();
}

int App::run() {
    auto screen = ScreenInteractive::Fullscreen();
    screen_ = &screen;

    // One Renderer drawing the whole dashboard from a per-frame snapshot.
    auto dashboard = Renderer([this] {
        return build_frame(state_.snapshot(), focus_, att_pan_row_, att_pan_col_, att_contrast_,
                           stream_filter_, filter_editing_);
    });

    // Global + routed key handling.
    auto root = CatchEvent(dashboard, [this](Event e) {
        // Cross-thread redraw nudge: consume so it doesn't propagate.
        if (e == Event::Custom)
            return true;

        // Modal: editing the live-stream filter (swallows all keys)
        if (filter_editing_) {
            if (e == Event::Return) {
                filter_editing_ = false;
                return true;
            }
            if (e == Event::Escape) {
                filter_editing_ = false;
                stream_filter_.clear();
                return true;
            }
            if (e == Event::Backspace) {
                if (!stream_filter_.empty())
                    stream_filter_.pop_back();
                return true;
            }
            if (e.is_character()) {
                stream_filter_ += e.character();
                return true;
            }
            return true;
        }

        // Global keys
        if (e == Event::q || e == Event::Q) {
            if (screen_)
                screen_->Exit();
            return true;
        }
        if (e == Event::Tab) {
            focus_ = (focus_ + 1) % P_COUNT;
            return true;
        }
        if (e == Event::TabReverse) {
            focus_ = (focus_ + P_COUNT - 1) % P_COUNT;
            return true;
        }

        // Replay scrub keys (p/s)
        if (replay_ctrl_) {
            if (e == Event::Character('p') || e == Event::Character('P')) {
                bool was = replay_ctrl_->paused.exchange(true, std::memory_order_acq_rel);
                if (was)
                    replay_ctrl_->paused.store(false, std::memory_order_release);
                return true;
            }
            if (e == Event::Character('s') || e == Event::Character('S')) {
                replay_ctrl_->step_request.fetch_add(1, std::memory_order_acq_rel);
                return true;
            }
        }

        // Topology pane keys
        if (focus_ == P_TOPOLOGY) {
            if (e == Event::j || e == Event::ArrowDown) {
                state_.set_selected(state_.selected() + 1);
                return true;
            }
            if (e == Event::k || e == Event::ArrowUp) {
                state_.set_selected(state_.selected() - 1);
                return true;
            }
            if (e == Event::Character(' ') || e == Event::Return) {
                const int sel = state_.selected();
                if (on_select_target && sel >= 0 && sel < state_.topology_size()) {
                    on_select_target(sel);
                }
                return true;
            }
        }

        // Attention pane keys
        if (focus_ == P_ATTENTION) {
            if (e == Event::h) {
                att_pan_col_ = std::max(0, att_pan_col_ - 4);
                return true;
            }
            if (e == Event::l) {
                att_pan_col_ += 4;
                return true;
            }
            if (e == Event::k) {
                att_pan_row_ = std::max(0, att_pan_row_ - 2);
                return true;
            }
            if (e == Event::j) {
                att_pan_row_ += 2;
                return true;
            }
            if (e == Event::Character('+') || e == Event::Character('=')) {
                att_contrast_ = std::min(8.0f, att_contrast_ * 1.25f);
                return true;
            }
            if (e == Event::Character('-') || e == Event::Character('_')) {
                att_contrast_ = std::max(0.1f, att_contrast_ / 1.25f);
                return true;
            }
            if (e == Event::F || e == Event::f) {
                // Fit: reset pan + contrast.
                att_pan_row_ = 0;
                att_pan_col_ = 0;
                att_contrast_ = 1.0f;
                return true;
            }
        }

        // Live Packet Stream pane keys
        if (focus_ == P_STREAM) {
            if (e == Event::Character('/')) {
                filter_editing_ = true;
                return true;
            }
            if (e == Event::Escape && !stream_filter_.empty()) {
                stream_filter_.clear();
                return true;
            }
        }
        return false;
    });

    screen.Loop(root);
    screen_ = nullptr;
    return 0;
}

}  // namespace ts
