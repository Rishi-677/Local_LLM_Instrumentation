// Local_LLM_Instrumentation — FTXUI TUI shell (C-TUI).
//
// `App` owns the 5-pane live dashboard. It reads a `UiState` snapshot once per
// frame and renders; it never blocks the producer. Input is keyboard-driven:
//   - Tab            cycle focus between the 5 panes
//   - j / k          move topology selection (when topology pane focused)
//   - space          set the current layer as capture target (on_select_target)
//   - h/j/k/l + - F  pan / contrast / fit the attention heatmap (when focused)
//   - q / Q          quit
//
// Cross-thread refresh: the lead's consumer thread calls `request_redraw()`
// after mutating the shared UiState; that posts an Event::Custom to the FTXUI
// screen so the loop redraws from the latest snapshot.

#pragma once

#include <functional>
#include <string>

#include "ui_state.hpp"

namespace ftxui {
class ScreenInteractive;
}

namespace ts {

class App {
public:
    explicit App(UiState& state);

    // Blocks on the FTXUI event loop until the user quits. Returns 0.
    // Must be called on the main thread.
    int run();

    // Thread-safe: nudges the render loop to redraw from the latest snapshot.
    // Safe to call before run() (no-op until the screen exists).
    void request_redraw();

    // Render a single frame of the current UiState to a fixed-size text buffer
    // (ANSI-colored). Used for headless layout preview / docs — no event loop.
    std::string preview(int width, int height);

    // Apply a live-stream filter (also used by --preview-filter for docs/tests).
    void set_stream_filter(const std::string& f) { stream_filter_ = f; }

    // Invoked with the layer index when the user presses space on a topology
    // row. Set this before run(). May be called from the render thread.
    std::function<void(int /*layer_idx*/)> on_select_target;

private:
    UiState&                  state_;
    ftxui::ScreenInteractive* screen_ = nullptr;  // valid only during run()

    // Focus: which of the 5 panes currently receives directional input.
    int  focus_ = 0;   // 0..4

    // Attention pane local view state (not shared — purely presentation).
    int   att_pan_row_  = 0;
    int   att_pan_col_  = 0;
    float att_contrast_ = 1.0f;

    // Live packet-stream filter ('/' to edit; matches layer type / op / device).
    std::string stream_filter_;
    bool        filter_editing_ = false;
};

} // namespace ts
