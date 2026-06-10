# Local_LLM_Instrumentation — Roadmap

> A terminal-native, non-invasive live debugger for local transformer LLMs.
> MVP target: **llama.cpp / GGUF**, single in-process **C++** binary, **FTXUI** dashboard.

This roadmap breaks the project into small, independently-buildable components. Work top-to-bottom; each component ends in a verifiable state. Check items off as they land.

---

## North star

Hook llama.cpp's built-in `ggml_backend_sched_eval_callback` (`cb_eval`) to observe **every tensor of every layer** during inference — non-invasively — compute lightweight stats, push them through a fixed-size lock-free ring buffer, and render a 5-pane vim-keybound TUI (Topology · Live Stream · Attention · Metrics · Anomaly).

```
┌─────────────────────── one C++ process ───────────────────────┐
│  [inference thread]                  [TUI render thread]       │
│  llama.cpp eval loop                 FTXUI app (~30–60 fps)    │
│    └─ cb_eval(tensor, ask, ud) ──►   drains ring buffer        │
│         compute lightweight stats    updates pane state        │
│         push TensorEvent ───────►  [lock-free SPSC ring buf]   │
│                                      keyboard: vim + Tab focus │
└───────────────────────────────────────────────────────────────┘
```

---

## Phase 0 — Scaffolding

- [x] **C0 · Build skeleton.** `CMakeLists.txt` vendoring `llama.cpp` + `FTXUI` via CMake `FetchContent`. Hello-world that loads a small GGUF and decodes one token. Verify it builds on Windows 11 / MSVC.
- [x] **C1 · Core schema** — `src/event.hpp`. `TensorEvent` { id, timestamp, layer_idx, node_name, op_type, class (Embed/Attn/MLP/Norm/Other), device, shape[4], dtype, latency_us, min, max, mean, sparsity, has_nan, has_inf, payload handle } and `LayerNode` topology types. **This struct is the contract for everything downstream and the seam for a future sidecar.**
- [x] **C2 · Ring buffer** — `src/ring_buffer.hpp`. Lock-free fixed-size SPSC queue of `TensorEvent` + dropped-count. Single producer = callback, single consumer = TUI.

## Phase 1 — Capture backbone

- [x] **C3 · Wire the hook.** Register `cb_eval` on `llama_context_params`; minimal version logs op/name/shape/dtype to prove non-invasive capture (mirror the upstream `eval-callback` example).
- [x] **C4 · Stats in the callback** — `src/capture/eval_callback.cpp`. Compute shape, dtype, min/max/mean, sparsity (fraction ≈ 0), NaN/Inf, per-op latency (timestamp deltas). `ggml_backend_tensor_get` for GPU tensors only when needed. Push `TensorEvent`. Keep it allocation-free and cheap.
- [x] **C5 · Topology builder** — `src/capture/topology.cpp`. Parse tensor names (`inp_embd`, `attn_norm-0`, `ffn_out-12`, `l_out-N`) → (class, layer_idx) → assemble/update the `LayerNode` tree incrementally. No separate model introspection needed.

## Phase 2 — TUI shell

- [x] **C6 · FTXUI layout** — `src/tui/app.cpp` + panes. 5-pane layout from the PS mockup + header (shortcuts + session state). `Tab` cycles focus, `Q` quits. Static data first.
- [x] **C7 · Live stream wired** — `src/tui/pane_stream.cpp`. Render thread drains the ring buffer → chronological scrolling stream with fixed history; show dropped-count when sampling kicks in.
- [x] **C8 · Topology navigation** — `src/tui/pane_topology.cpp`. Navigable tree: `j/k` move, `space` selects the **capture target**, active layer highlighted; selection drives other panes.

## Phase 3 — Signature views

- [x] **C9 · Metrics inspector** — `src/tui/pane_metrics.cpp`. Shape / dtype / sparsity bar / latency delta for the selected layer/event.
- [x] **C10 · Anomaly ledger** — `src/anomaly.cpp` + `src/tui/pane_anomaly.cpp`. Rules: NaN/Inf, extreme values (user threshold), CPU-fallback / unexpected device transfer. Timestamped, severity-tagged, scrolling.
- [x] **C11 · Attention heatmap** — `src/tui/pane_attention.cpp`. Capture the KQ `SOFT_MAX` tensor for the selected layer/head (**requires `--flash-attn off`**). Unicode block heatmap (`██▒▒░░`); `h/j/k/l` pan, `+/-` contrast, head/layer/token-window select, `F` fullscreen. Graceful "unavailable" hint when flash-attn is on.

## Phase 4 — Polish & robustness

- [x] **C12 · Performance.** `--bench` measures hook overhead (hooked vs un-hooked throughput). Finding: ~50% overhead is *structural* — `cb_eval` runs the graph node-by-node, defeating fusion (not our stats). `capture.stats_sample` makes the per-tensor scan tunable.
- [x] **C13 · Search & feel.** `/` live search/filter on the packet stream (layer type / op / device / node), warm gruvbox theme, focus marker, per-layer latency in topology.
- [x] **C14 · Record & replay** — `src/session/recorder.cpp` + `replay.cpp`. Record to NDJSON; replay feeds the *same* TUI from a file instead of the live callback (proves the schema seam, de-risks the future sidecar).

## Phase 0 — Foundation infrastructure  ✅ done

- [x] Core data model split: `TensorMeta` / `ActivationStats` / `LayerEvent` (`src/event.hpp`).
- [x] Typed event bus with subscribe/publish + priority (`src/event_bus.hpp`).
- [x] TOML config loader via toml++ (`src/config.*`, `local_llm_instrumentation.toml`) — buffer size, capture mask, anomaly, attention, theme.
- [x] spdlog file logging; CMake ASAN/UBSAN/TSAN presets; clang-format/clang-tidy; Catch2 tests (24); GitHub Actions CI.

## Phase 7 — Export (partial)

- [x] Per-layer stats summary → CSV / JSON (`src/export.hpp`, `--export-csv/--export-json`).
- [ ] HTML self-contained report with embedded SVG flamegraph.

## Backlog — next candidates

- [ ] **Latency flamegraph panel** (Phase 5) — ASCII bars per layer sorted by compute time (export data already supports it).
- [ ] **Crash/panic handler** (Phase 3) — restore the terminal on signal/exception.
- [ ] **Replay scrubber** (Phase 6) — step / pause / speed over a recorded session.
- [ ] **Mask-in-`ask` overhead cut** — return false in the cb_eval ask phase for masked classes so they skip the callback entirely.

## Phase 5 — Platform (post-MVP, not built now)

- [ ] PyTorch sidecar (`register_forward_hook` + `output_attentions` → FlatBuffers over loopback socket → same TUI).
- [ ] TF/Keras adapter; FlatBuffers `.llmtrace` sessions; packaging.

---

## File layout (greenfield)

```
roadmap.md
CMakeLists.txt                  # FetchContent: llama.cpp, FTXUI
src/
  main.cpp                      # arg parse, load GGUF, spawn threads
  event.hpp                     # TensorEvent + LayerNode (schema seam)
  ring_buffer.hpp               # lock-free SPSC ring
  capture/
    eval_callback.cpp           # cb_eval + inline stats
    topology.cpp                # name parser → tree
  anomaly.cpp                   # anomaly rules + ledger
  tui/
    app.cpp                     # FTXUI app, focus/keymap
    pane_topology.cpp
    pane_stream.cpp
    pane_attention.cpp
    pane_metrics.cpp
    pane_anomaly.cpp
  session/
    recorder.cpp / replay.cpp   # NDJSON record + replay
```

## Risks / constraints

- **Flash attention hides the attention matrix** → attention view requires `--flash-attn off`; degrade gracefully.
- **GPU tensor read-back costs** → only copy selected-target payloads; sample the rest.
- **Callback is hot** (fires per-op) → keep it allocation-free; heavy work on the TUI thread.
- **Windows build** of llama.cpp + FTXUI under MSVC — verify early in C0.
- **High event volume** can swamp the TUI → bounded ring + visible drop counter, never silent truncation.

## Verification (end-to-end)

1. Build on Windows via CMake; llama.cpp + FTXUI link and a GGUF loads (C0).
2. Run against a small GGUF; Live Stream scrolls and the Topology tree auto-populates.
3. `j/k` + `space` on a layer updates Metrics + Attention; `Tab` cycles focus; `Q` quits.
4. With `--flash-attn off`, the heatmap renders/pans; with it on, shows the graceful hint.
5. Crafted anomaly case is flagged with timestamp + severity.
6. Record to NDJSON, replay into the same TUI, panes match.
7. Hooked vs un-hooked tokens/sec shows bounded overhead; drop counter behaves under load.
