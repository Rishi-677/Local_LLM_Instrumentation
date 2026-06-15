# Local_LLM_Instrumentation

A terminal based, non invasive live debugger for local transformer language models.

When a language model runs, it pushes your text through many layers of math, and normally
you cannot watch what happens inside while it runs. This tool lets you watch. It attaches to
a running [llama.cpp](https://github.com/ggml-org/llama.cpp) inference through ggml's built in
`cb_eval` callback, measures a few cheap numbers for every tensor of every layer, and streams
that telemetry into a keyboard driven terminal dashboard. It never changes the model or its
code, which is what "non invasive" means. Think of it as btop or lazygit, but for a model's
forward pass.

It also has a sidecar mode that does the same thing for PyTorch and TensorFlow models running
in Python, by sending the telemetry over a local network socket into the very same dashboard.

Status: working MVP. The main target is llama.cpp and GGUF models, as a single C++ program.
See [`docs/implementation-roadmap.md`](docs/implementation-roadmap.md) for the full component
breakdown, and [`docs/product-requirements.md`](docs/product-requirements.md) for the goals.

## Why it exists

Debugging a model today usually means one of two painful options. You either add a lot of
logging directly into the model code, which is intrusive and slow, or you use heavy separate
tools after the run is finished, which cannot show you what happened live. This project gives
you a third option: a fast, live, read only view of the forward pass, with almost no setup and
without touching the model.

## What it shows

The dashboard has six panes. Press Tab to move focus between them, and use vim style keys
inside the focused pane.

1. Model Topology. The structure of the model as a tree, built live just from the tensor
   names with no hard coding. Move with j and k, and press Space to set a layer as the capture
   target. Each layer shows the total time spent in it, so the slowest block stands out.
2. Live Packet Stream. Every tensor operation in the order it happened, with an id, a
   timestamp, the operation type, and the device. Press the forward slash key to filter the
   list live, and Escape to clear the filter.
3. Attention Matrix. The attention heatmap for the selected layer and head. Move around it
   with h, j, k, and l, change the contrast with plus and minus, and press capital F to make it
   fill the screen.
4. Runtime Metrics. For the selected tensor: shape, data type, a sparsity bar, latency, the
   value range, the mean, the standard deviation, the L2 norm, and a NaN or infinity flag.
5. Latency Flamegraph. Bars that rank the layers by how much compute time they used.
6. Anomaly Ledger. A running log of problems such as NaN or infinity values, unusually large
   numbers, and operations that fell back to the CPU, each with a timestamp and a severity.

## How it works

Everything runs inside one program with two threads. The first thread runs the model. Every
time the engine computes a tensor, it calls the hook, which measures a few cheap numbers and
drops a small fixed size record into a queue. The second thread is the dashboard, which empties
the queue about thirty times per second and redraws. The queue has a fixed size, so if the
model produces records faster than the screen can show them, the queue drops the extra records
and counts how many it dropped. This way the model is never slowed down waiting for the screen,
and memory never grows without limit.

```mermaid
flowchart LR
    subgraph one["one process, two threads"]
        direction LR
        L["llama.cpp runs the model"] -->|"hook fires for every tensor"| C["compute cheap stats: shape, min, max, mean, std, L2, sparsity, NaN or Inf, latency"]
        C -->|"push one small record"| R[["lock free queue, fixed size, counts drops"]]
        R -->|"drain about 30 times per second"| D["dashboard draws the panes"]
    end
    K["keyboard: vim keys and Tab"] --> D
```

The hook computes only lightweight stats inline and pushes a plain old data record called a
`TensorEvent` through the queue. The heavy attention matrix is captured on the side, and only
for the selected layer, so the hot path stays cheap. That same `TensorEvent` record is the one
contract used everywhere: it travels through the queue, it is written to disk for recording,
and it is sent across the network for the sidecar. Because it is the same structure in all
three places, the dashboard code is reused without changes.

## Build

You need CMake version 3.20 or newer, a C++20 compiler, and Git. The dependencies, which are
llama.cpp (tag b9587) and FTXUI (version 6.1.9), are downloaded automatically by CMake.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target local_llm_instrumentation
```

On Windows with the MinGW toolchain the program is linked statically (using the `-fopenmp` and
`-static` flags), so the resulting binary is self contained. Linux with GCC also builds and is
covered in continuous integration.

## Run

In the examples below, replace `model.gguf` with the path to a real GGUF model file. On Linux
or macOS, drop the `.exe` from the program name.

```sh
# Live mode: attach to a local model and watch it think in the dashboard
./build/local_llm_instrumentation.exe model.gguf "The quick brown fox"

# Load settings from a TOML file, then print the settings that are actually in effect
./build/local_llm_instrumentation.exe model.gguf --config local_llm_instrumentation.toml
./build/local_llm_instrumentation.exe --print-config

# Record a session to a file, then replay it later with no model loaded
./build/local_llm_instrumentation.exe model.gguf "Hello" --record session.ndjson
./build/local_llm_instrumentation.exe --replay session.ndjson

# Run without the dashboard and write analysis files (a per layer summary)
./build/local_llm_instrumentation.exe model.gguf "Hello" --headless --export-csv out.csv --export-json out.json --export-html report.html

# Measure how much the hook costs (with the hook versus without it)
./build/local_llm_instrumentation.exe model.gguf --bench --max-tokens 32

# Draw one full dashboard frame to the screen and exit (useful for a screenshot)
./build/local_llm_instrumentation.exe model.gguf --preview
```

## Keyboard reference

| Key | What it does |
| --- | --- |
| Tab, or Shift and Tab | move focus between panes |
| j or k | move the selection in the topology pane |
| Space | set the selected layer as the capture target |
| h, j, k, l | move around the attention matrix |
| plus or minus | change the attention contrast |
| f | reset the attention view |
| capital F | toggle attention fullscreen |
| forward slash, then text, then Escape | filter the packet stream, then clear it |
| p or s | during replay, pause or step one event at a time |
| q | quit |

## Recording and replay (the persistence layer)

There is no traditional database. The persistence layer is the session format. The `--record`
option writes the full telemetry stream to a file, one JSON object per line (the NDJSON format,
or a compact `.llmtrace` format). The `--replay` option reads that file back into the exact
same dashboard, with no model loaded at all. This works because the format on disk is the same
`TensorEvent` structure the program uses in memory. During replay you can pause with p and step
one event at a time with s, like a debugger.

## Sidecar mode (the network API)

The same dashboard can watch a PyTorch or TensorFlow model. The C++ side opens a network socket
and acts as a server. A small Python program attaches hooks to the model and streams telemetry
to that socket. The messages are line delimited JSON: first a `ready` message, then many
`event` messages, then a `done` message.

```sh
# Terminal A: start the dashboard as a receiver (the server)
./build/local_llm_instrumentation.exe --sidecar --sidecar-port 9876

# Terminal B: stream a small PyTorch model's tensors to the server (the client)
python py_sidecar/sidecar.py --model hf-internal-testing/tiny-random-gpt2 --port 9876 --max-tokens 6 --output-attentions
```

There is also an automated end to end test that starts the server, runs the client against a
tiny model, and checks that events actually arrived. It is a good way to confirm the whole
pipeline works:

```sh
python tests/test_sidecar_e2e.py
```

The Python sidecars need a few packages, listed in
[`py_sidecar/requirements.txt`](py_sidecar/requirements.txt). The PyTorch sidecar lives in
`py_sidecar/sidecar.py` and the TensorFlow and Keras version lives in `py_sidecar/sidecar_tf.py`.

## Common options

| Option | Meaning |
| --- | --- |
| `--config <file>` | load settings from a TOML file (command line flags still override) |
| `--max-tokens N` | how many tokens to generate |
| `--delay-ms N` | slow the run down so it is easy to watch on screen |
| `--ring-capacity N` | the size of the telemetry queue |
| `--anomaly-threshold X` | flag a value as an outlier when its size goes above X |
| `--no-flash-attn` | turn off flash attention (this is the default, and it is needed for the attention pane) |
| `--flash-attn` | allow flash attention (this hides the attention matrix) |
| `--record <file>` or `--replay <file>` | record a session, or replay one |
| `--headless` | run with no dashboard, for scripts and continuous integration |
| `--export-csv`, `--export-json`, `--export-html <file>` | write a per layer summary |
| `--bench` | measure the cost of the hook and exit |
| `--sidecar`, `--sidecar-port N`, `--sidecar-host ADDR` | receive telemetry from a Python sidecar |

## Tests

The project has a unit test suite (using Catch2) and the end to end sidecar test described
above. To build and run the unit tests:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLOCAL_LLM_INSTRUMENTATION_BUILD_TESTS=ON
cmake --build build --target unit_tests
cd build && ctest --output-on-failure
```

## Notes and limitations

1. The attention pane needs flash attention to be off, which is the default here. When flash
   attention is on, llama.cpp never builds the attention matrix, so the pane shows a short
   message instead of a heatmap.
2. The full attention matrix, of size queries by keys, appears while the prompt is being read
   in. During single token generation you instead see one row of attention.
3. The MVP is CPU first. Tensors on a GPU are detected, but their statistics are sampled on the
   host side only.
4. Attention is captured live and is not part of the recorded session file, because it is
   handled on a side channel rather than through the main event stream.
