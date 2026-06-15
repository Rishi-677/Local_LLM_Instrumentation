# Local_LLM_Instrumentation

## Clone

```sh
git clone https://github.com/Rishi-677/Local_LLM_Instrumentation.git
cd Local_LLM_Instrumentation
```

## Build

You need CMake version 3.20 or newer, a C++20 compiler, and Git. The dependencies, which are
llama.cpp (tag b9587) and FTXUI (version 6.1.9), are downloaded automatically by CMake.

The C++ program depends on:
- **llama.cpp** (tag b9587) — GGUF model loading and inference
- **FTXUI** (version 6.1.9) — terminal UI framework (fullscreen, keyboard, rendering)
- **spdlog** — asynchronous logging
- **toml++** — TOML config parsing
- **Catch2** — unit tests

All dependencies are fetched automatically by CMake's `FetchContent`. No manual installation
is needed beyond CMake 3.20+, a C++20 compiler, and Git.

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
