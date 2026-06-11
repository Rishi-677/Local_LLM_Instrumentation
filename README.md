# Local_LLM_Instrumentation

A terminal-native, **non-invasive live debugger for local transformer LLMs**.

Local_LLM_Instrumentation attaches to a running [llama.cpp](https://github.com/ggml-org/llama.cpp)
inference via ggml's built-in `cb_eval` hook and streams layer-level, tensor-level,
and attention telemetry into a keyboard-driven terminal dashboard — without
modifying the model or its code. Think `btop`/`lazygit`, but for a model's forward pass.

> Status: MVP. Targets llama.cpp / GGUF as a single in-process C++ binary.
> See [`docs/implementation-roadmap.md`](docs/implementation-roadmap.md) for the phased component breakdown.

## What it shows

A 5-pane dashboard (vim-style keys, `Tab` to cycle focus):

1. **Model Topology** — the forward pass as a navigable tree (`j`/`k` to move, `space` to set the capture target).
2. **Live Packet Stream** — every tensor op in chronological order (id, time, op, device).
3. **Attention Matrix** — the selected layer/head `kq_soft_max` heatmap (`h/j/k/l` pan, `+`/`-` contrast).
4. **Runtime Metrics** — shape, dtype, sparsity, latency for the selected layer.
5. **Anomaly Ledger** — NaN / Inf / extreme-value flags with timestamps.

## How it works

```
┌─────────────────────── one C++ process ───────────────────────┐
│  inference thread                    TUI render thread         │
│  llama.cpp eval loop                 FTXUI app (~30 fps)       │
│    └─ cb_eval(tensor) ───────────►   drains ring buffer        │
│         compute cheap stats          updates 5 panes          │
│         push TensorEvent ───────►  [lock-free SPSC ring buf]   │
└───────────────────────────────────────────────────────────────┘
```

The capture hook computes only lightweight stats inline (shape, dtype, min/max/mean,
sparsity, NaN/Inf, latency) and pushes a POD `TensorEvent` through a fixed-size
lock-free ring; the heavy attention matrix is captured out-of-band for the selected
layer only. The event struct is also the seam for a future PyTorch/TF sidecar.

## Build

Requires CMake ≥ 3.20, a C++20 compiler, and Git. llama.cpp (b9587) and FTXUI (v6.1.9)
are fetched automatically.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target local_llm_instrumentation
```

On MinGW/Windows the link is static (`-fopenmp -static`) so the binary is
self-contained.

## Run

```sh
# Live: attach to a local GGUF model and watch it think
./build/local_llm_instrumentation model.gguf "The quick brown fox"

# Settings from a TOML file (CLI flags override); print the effective config
./build/local_llm_instrumentation model.gguf --config local_llm_instrumentation.toml
./build/local_llm_instrumentation --print-config

# Record a session to NDJSON; replay it later (no model needed)
./build/local_llm_instrumentation model.gguf "Hello" --record session.ndjson
./build/local_llm_instrumentation --replay session.ndjson

# Headless run + analysis exports (per-layer stats summary)
./build/local_llm_instrumentation model.gguf "Hello" --headless --export-csv out.csv --export-json out.json

# Measure the capture overhead (hooked vs un-hooked throughput)
./build/local_llm_instrumentation model.gguf --bench --max-tokens 32

# Render one dashboard frame to stdout (docs / layout check)
./build/local_llm_instrumentation model.gguf --preview
```

In the TUI: `Tab` cycles panes, `j/k` + `Space` pick a capture target, `h/j/k/l`
+ `+/-` pan/contrast the attention heatmap, `/` filters the packet stream, `q` quits.

Key options: `--config <file>`, `--max-tokens N`, `--delay-ms N`, `--ring-capacity N`,
`--anomaly-threshold X`, `--no-flash-attn` (default — needed for the attention matrix),
`--record/--replay <file>`, `--headless`, `--bench`, `--export-csv/--export-json <file>`.

## Notes & limitations

- **Attention requires flash attention OFF** (the default here). With flash attention on,
  llama.cpp never materializes the `kq_soft_max` matrix, so the attention pane shows a hint.
- The full N×N attention matrix appears during prompt **prefill**; single-token generation
  steps produce a 1×N_kv attention row.
- MVP is CPU-first; GPU tensors are detected but their stats are sampled host-side only.
- Attention is captured live and is **not** part of the recorded event stream (it is out-of-band).
