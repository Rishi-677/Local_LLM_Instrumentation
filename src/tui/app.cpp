// Local_LLM_Instrumentation — FTXUI TUI shell implementation (C-TUI).
//
// All five panes live here to keep the public surface (app.hpp) minimal. Each
// pane is a free function taking the per-frame Snapshot plus whatever local
// view state it needs; App::run() wires them into a Container with a single
// CatchEvent that owns global keys (Tab/q) and routes directional keys to the
// focused pane.

#include "app.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "ftxui/component/component.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/color.hpp"

namespace ts {

using namespace ftxui;

namespace {

// Pane identity — index used by the focus cycle.
enum Pane : int {
    P_TOPOLOGY  = 0,
    P_STREAM    = 1,
    P_ATTENTION = 2,
    P_METRICS   = 3,
    P_LEDGER    = 4,
    P_COUNT     = 5,
};

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
    std::snprintf(buf, sizeof(buf), "%02llu:%02llu:%02llu.%03llu",
                  (unsigned long long)h, (unsigned long long)m,
                  (unsigned long long)s, (unsigned long long)ms);
    return std::string(buf);
}

std::string shape_to_string(const std::array<int64_t, kMaxDims>& shape) {
    std::string out = "[";
    bool first = true;
    for (int i = 0; i < kMaxDims; ++i) {
        if (shape[i] == 0 && i > 0) continue;  // trim trailing zero dims
        if (!first) out += ", ";
        out += std::to_string(shape[i]);
        first = false;
    }
    out += "]";
    return out;
}

// A few ggml_type values mapped to short names; falls back to the raw enum.
std::string dtype_name(int32_t dtype) {
    switch (dtype) {
        case 0:  return "f32";
        case 1:  return "f16";
        case 2:  return "q4_0";
        case 3:  return "q4_1";
        case 8:  return "q8_0";
        case 24: return "i8";
        case 25: return "i16";
        case 26: return "i32";
        case 30: return "bf16";
        default: return "t" + std::to_string(dtype);
    }
}

// Title decorator: bright + bold, with a "(Focus Active)" suffix when focused.
Element pane_title(const std::string& name, bool focused) {
    if (focused) {
        return hbox({text(" " + name + " ") | bold | color(Color::Black) |
                         bgcolor(Color::Cyan),
                     text(" (Focus Active)") | color(Color::CyanLight) | dim});
    }
    return text(" " + name + " ") | bold | color(Color::GrayLight);
}

Decorator pane_border(bool focused) {
    return [focused](Element e) {
        Element b = border(e);
        if (focused) b = b | color(Color::Cyan);
        return b;
    };
}

// ---- Pane 1: MODEL TOPOLOGY ----------------------------------------------
Element render_topology(const UiState::Snapshot& s, bool focused) {
    Elements rows;
    if (s.topology.empty()) {
        rows.push_back(text(" (awaiting topology…) ") | dim);
    }
    for (int i = 0; i < static_cast<int>(s.topology.size()); ++i) {
        const LayerNode& n = s.topology[i];
        const bool sel = (i == s.selected);

        std::string idx = (n.layer_idx >= 0)
                              ? ("L" + std::to_string(n.layer_idx))
                              : "  -";
        std::string line = (n.selected ? "[*] " : "[ ] ") + idx + "  " +
                           n.label + "  <" + to_string(n.op_class) + ">";

        Element row = text(line);
        if (sel) {
            row = row | bold | color(Color::Black) |
                  bgcolor(focused ? Color::CyanLight : Color::GrayLight);
        } else if (n.selected) {
            row = row | color(Color::GreenLight);
        }
        rows.push_back(row);
    }
    return vbox(std::move(rows)) | yframe | flex;
}

// ---- Pane 2: LIVE PACKET STREAM ------------------------------------------
Element render_stream(const UiState::Snapshot& s) {
    Elements rows;
    rows.push_back(hbox({
                       text("ID") | size(WIDTH, EQUAL, 7) | bold,
                       separator(),
                       text("TIME") | size(WIDTH, EQUAL, 13) | bold,
                       separator(),
                       text("OP") | size(WIDTH, EQUAL, 22) | bold,
                       separator(),
                       text("DEV") | bold,
                   }));
    rows.push_back(separator());

    // Newest at bottom; frame keeps the latest in view.
    for (const TensorEvent& ev : s.stream) {
        std::string op = std::string(to_string(ev.op_class)) + "/" +
                         std::string(ev.op_name);
        if (op.size() > 22) op.resize(22);

        Element devc = text(std::string(to_string(ev.device)));
        if (ev.device != Device::CUDA && ev.device != Device::Metal)
            devc = devc | color(Color::Yellow);  // CPU fallback stands out
        else
            devc = devc | color(Color::GreenLight);

        Element row = hbox({
            text(std::to_string(ev.id)) | size(WIDTH, EQUAL, 7),
            separator(),
            text(fmt_time(ev.timestamp_ns)) | size(WIDTH, EQUAL, 13),
            separator(),
            text(op) | size(WIDTH, EQUAL, 22),
            separator(),
            devc,
        });
        if (ev.has_nan || ev.has_inf) row = row | color(Color::Red);
        rows.push_back(row);
    }
    return vbox(std::move(rows)) | yframe | flex;
}

// ---- Pane 3: ATTENTION MATRIX VISUALIZER ---------------------------------
Element render_attention(const UiState::Snapshot& s, int pan_row, int pan_col,
                         float contrast) {
    if (!s.has_attention || s.attention.weights.empty() ||
        s.attention.rows <= 0 || s.attention.cols <= 0) {
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
    for (float w : a.weights) wmax = std::max(wmax, std::fabs(w));
    if (wmax <= 0.0f) wmax = 1.0f;

    static const char* kRamp[] = {" ", "░", "▒", "▓", "█"};
    constexpr int kRampN = 5;

    // Viewport: cap to a sensible size; pan offsets clamp inside the matrix.
    const int vp_rows = std::min(rows, 24);
    const int vp_cols = std::min(cols, 64);
    const int r0 = std::clamp(pan_row, 0, std::max(0, rows - vp_rows));
    const int c0 = std::clamp(pan_col, 0, std::max(0, cols - vp_cols));

    Elements lines;
    {
        std::string hdr = "head " + std::to_string(a.head) + "  layer " +
                          std::to_string(a.layer_idx) + "  [" +
                          std::to_string(rows) + "x" + std::to_string(cols) +
                          "]  view r" + std::to_string(r0) + " c" +
                          std::to_string(c0) + "  x" +
                          std::to_string(contrast).substr(0, 4);
        lines.push_back(text(hdr) | dim);
    }

    for (int r = r0; r < r0 + vp_rows && r < rows; ++r) {
        Elements cells;
        for (int c = c0; c < c0 + vp_cols && c < cols; ++c) {
            float v = std::fabs(a.weights[static_cast<size_t>(r) * cols + c]) /
                      wmax;
            v = std::clamp(v * contrast, 0.0f, 1.0f);
            int level = static_cast<int>(v * (kRampN - 1) + 0.5f);
            level = std::clamp(level, 0, kRampN - 1);

            // Color by intensity for readability beyond the glyph ramp.
            Color fg = Color::GrayDark;
            if (level >= 4) fg = Color::Red1;
            else if (level == 3) fg = Color::Orange1;
            else if (level == 2) fg = Color::Yellow1;
            else if (level == 1) fg = Color::GreenLight;
            cells.push_back(text(kRamp[level]) | color(fg));
        }
        lines.push_back(hbox(std::move(cells)));
    }
    return vbox(std::move(lines)) | flex;
}

// ---- Pane 4: RUNTIME METRICS INSPECTOR -----------------------------------
Element render_metrics(const UiState::Snapshot& s) {
    if (!s.has_metrics) {
        return vbox({text(" (no metrics for selected layer yet) ") | dim}) |
               flex;
    }
    const TensorEvent& m = s.selected_metrics;

    Element nan_inf = text("clean") | color(Color::GreenLight);
    if (m.has_nan || m.has_inf) {
        std::string f = "";
        if (m.has_nan) f += "NaN ";
        if (m.has_inf) f += "Inf";
        nan_inf = text(f) | color(Color::Red) | bold;
    }

    return vbox({
               hbox({text("node    : ") | bold,
                     text(std::string(m.node_name))}),
               hbox({text("op      : ") | bold,
                     text(std::string(to_string(m.op_class)) + " / " +
                          std::string(m.op_name))}),
               hbox({text("shape   : ") | bold, text(shape_to_string(m.shape))}),
               hbox({text("dtype   : ") | bold,
                     text(dtype_name(m.dtype) + "  (" +
                          std::to_string(m.dtype_size) + "B/elem)")}),
               hbox({text("device  : ") | bold, text(to_string(m.device))}),
               hbox({text("latency : ") | bold,
                     text(std::to_string(m.latency_us) + " us")}),
               hbox({text("range   : ") | bold,
                     text("[" + std::to_string(m.v_min) + ", " +
                          std::to_string(m.v_max) + "]  mean " +
                          std::to_string(m.v_mean))}),
               hbox({text("nan/inf : ") | bold, nan_inf}),
               separator(),
               hbox({text("sparsity ") | bold,
                     gauge(std::clamp(m.sparsity, 0.0f, 1.0f)) | flex,
                     text(" " +
                          std::to_string(static_cast<int>(m.sparsity * 100)) +
                          "%")}),
           }) |
           flex;
}

// ---- Pane 5: NUMERICAL ANOMALY LEDGER ------------------------------------
Element render_ledger(const UiState::Snapshot& s) {
    Elements rows;
    if (s.anomalies.empty()) {
        rows.push_back(text(" (no anomalies) ") | color(Color::GreenLight));
    }
    // Newest last; frame scrolls to keep recent ones visible.
    for (const Anomaly& a : s.anomalies) {
        const char* glyph = (a.severity >= 2) ? "✖" : "⚠";
        Color gc = (a.severity >= 2) ? Color::Red : Color::Yellow;
        rows.push_back(hbox({
            text(fmt_time(a.ts_ns)) | size(WIDTH, EQUAL, 13) | dim,
            text(" "),
            text(glyph) | color(gc) | bold,
            text(" "),
            text(a.text),
        }));
    }
    return vbox(std::move(rows)) | yframe | flex;
}

}  // namespace

// ---------------------------------------------------------------------------

App::App(UiState& state) : state_(state) {}

void App::request_redraw() {
    if (screen_) screen_->PostEvent(Event::Custom);
}

int App::run() {
    auto screen = ScreenInteractive::Fullscreen();
    screen_ = &screen;

    // One Renderer drawing the whole dashboard from a per-frame snapshot.
    auto dashboard = Renderer([this] {
        UiState::Snapshot s = state_.snapshot();

        // Header: shortcuts + session state.
        Color state_col = Color::GreenLight;
        if (s.session_state != "LIVE") state_col = Color::Yellow;

        Element header = hbox({
            text(" Local_LLM_Instrumentation ") | bold | color(Color::Black) |
                bgcolor(Color::Cyan),
            text("  " + (s.model_name.empty() ? std::string("(no model)")
                                              : s.model_name) +
                 "  "),
            separator(),
            text("  Tab focus · j/k select · space target · h/j/k/l pan · "
                 "+/- contrast · q quit  ") |
                dim,
            filler(),
            text(" tok:" + std::to_string(s.n_tokens) + " ev:" +
                 std::to_string(s.total_events) + " drop:" +
                 std::to_string(s.dropped) + " "),
            text(" " + s.session_state + " ") | bold | color(Color::Black) |
                bgcolor(state_col),
        });

        // Pane bodies.
        Element topo = vbox({pane_title("MODEL TOPOLOGY", focus_ == P_TOPOLOGY),
                             separator(),
                             render_topology(s, focus_ == P_TOPOLOGY)}) |
                       pane_border(focus_ == P_TOPOLOGY) | flex;

        Element stream = vbox({pane_title("LIVE PACKET STREAM",
                                          focus_ == P_STREAM),
                               separator(), render_stream(s)}) |
                         pane_border(focus_ == P_STREAM) | flex;

        Element attention =
            vbox({pane_title("ATTENTION MATRIX VISUALIZER",
                             focus_ == P_ATTENTION),
                  separator(),
                  render_attention(s, att_pan_row_, att_pan_col_,
                                   att_contrast_)}) |
            pane_border(focus_ == P_ATTENTION) | flex;

        Element metrics = vbox({pane_title("RUNTIME METRICS INSPECTOR",
                                           focus_ == P_METRICS),
                                separator(), render_metrics(s)}) |
                          pane_border(focus_ == P_METRICS) | flex;

        Element ledger = vbox({pane_title("NUMERICAL ANOMALY LEDGER",
                                          focus_ == P_LEDGER),
                               separator(), render_ledger(s)}) |
                         pane_border(focus_ == P_LEDGER) | flex;

        // Layout:
        //   ┌ header ───────────────────────────────────┐
        //   │ topology (tall) │ stream (tall)            │
        //   │ attention (wide, full row)                 │
        //   │ metrics         │ ledger                   │
        Element top_row =
            hbox({topo | size(WIDTH, GREATER_THAN, 34) | flex, stream | flex}) |
            flex;
        Element bottom_row = hbox({metrics | flex, ledger | flex});

        return vbox({
            header,
            separator(),
            top_row,
            attention | size(HEIGHT, GREATER_THAN, 8),
            bottom_row | size(HEIGHT, GREATER_THAN, 9),
        });
    });

    // Global + routed key handling.
    auto root = CatchEvent(dashboard, [this](Event e) {
        // Cross-thread redraw nudge: consume so it doesn't propagate.
        if (e == Event::Custom) return true;

        // ---- Global keys ----
        if (e == Event::q || e == Event::Q) {
            if (screen_) screen_->Exit();
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

        // ---- Topology pane keys ----
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
                if (on_select_target && sel >= 0 &&
                    sel < state_.topology_size()) {
                    on_select_target(sel);
                }
                return true;
            }
        }

        // ---- Attention pane keys ----
        if (focus_ == P_ATTENTION) {
            if (e == Event::h) { att_pan_col_ = std::max(0, att_pan_col_ - 4); return true; }
            if (e == Event::l) { att_pan_col_ += 4; return true; }
            if (e == Event::k) { att_pan_row_ = std::max(0, att_pan_row_ - 2); return true; }
            if (e == Event::j) { att_pan_row_ += 2; return true; }
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
        return false;
    });

    screen.Loop(root);
    screen_ = nullptr;
    return 0;
}

}  // namespace ts
