## Local LLM Instrumentation, Tracing, and Replay Platform

---

### Overview

Ever had problem messing around with the concept of Transformers and Attention? Or wondered how data propagates through the layers? This project serves as a lightweight telemetry and diagnostic tool for local transformer models

### Expected Tech Stack

C++

### Team Size

2 Members

### Problem Statement

- The system should hook non-invasively (without altering the model’s code) into the model's execution pipeline
- After that it captures real-time intermediate states (shapes, activation stats, and layer latencies) as tokens progress through the architecture
- Finally, the data is presented as an interactive TUI interface (can use keyboard-based or vim-like keybindings to increase interactivity)
- Intercept the forward pass to extract metadata and activations from submodules (embeddings, attention, MLP) without modifying model source code
- Expected metrics to be analysed
    - Layer-by-layer execution latency to determine which transformer block dominates computing
    - Attention matrix visualization
    - Runtime metrics inspection (tensor shape, sparsity rate, timestamps)
- Optional - track sparsity, mean, max to flag numerical anomalies or clipping risks or outlier features
- Example of visualisation method
    - Store upto a fixed number of layer information in a fixed size ring buffer (so that the RAM consumption doesn’t shoot up)
    - Tab key cycles focus between different sections of the TUI
    - One section lists out the layers in sequence, where they can cycled by using j/k keys and space to select
    - Other sections update accordingly, displaying other information

 [Tab]: Cycle Focus  |  [Q]: Quit App
╔══ █ 1. MODEL TOPOLOGY (Focus Active) ═══════════════╗┌── 2. LIVE PACKET STREAM ──────────────────────────────┐
║ ▼ llama-3-8b                                        ║│  ID  │ TIMESTAMP    │ LAYER TYPE   │ COMPUTE DEVICE   │
║   ► embed_tokens                                    ║├──────┼──────────────┼──────────────┼──────────────────┤
║   ▼ layers                                          ║│  104 │ 21:14:02.110 │ Attn (Self)  │ CUDA [GPU 0]     │
║    ▶ layers.0                                       ║│  105 │ 21:14:02.114 │ MLP (SwiGLU) │ CUDA [GPU 0]     │
║    ▼ layers.1  [Active Capture Target]              ║│  106 │ 21:14:02.119 │ Attn (Self)  │ CUDA [GPU 0]     │
║      ● layers.1.attn ───────────────────────────────╢│  107 │ 21:14:02.122 │ MLP (SwiGLU) │ CUDA [GPU 0]     │
║      ● layers.1.mlp                                 ║│  108 │ 21:14:02.128 │ LayerNorm    │ CPU (Fallback)   │
║    ► layers.2                                       ║│      │              │              │                  │ 
╚═════════════════════════ [Press j/k to Navigate] ═══╝└──────┴──────────────┴──────────────┴──────────────────┘
┌── 3. ATTENTION MATRIX VISUALIZER (HEAD 0) ───────────────────────────────────────────────────────────────────┐
│ Tokens:  [I]   [want]  [it]   [to]   [be]   [keyboard] [driven]               Viewport Window: [0-7] x [0-7] │
│ [I]       ██     ░░     ░░     ░░     ░░        ░░        ░░         ────────────────────────────────────────│
│ [want]    ▒▒     ██     ░░     ░░     ░░        ░░        ░░            [Focus + F]: Open Fullscreen         │
│ [it]      ░░     ▒▒     ██     ░░     ░░        ░░        ░░            [Arrows/(h,j,k,l)]: Pan Matrix       │
│ [to]      ░░     ░░     ▒▒     ██     ░░        ░░        ░░            [+/-]: Change Weight Contrast        │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
┌── 4. RUNTIME METRICS INSPECTOR ──────────────────────┐┌── 5. NUMERICAL ANOMALY LEDGER ───────────────────────┐
│ Tensor Shape : [1, 32, 4096]   Dtype: float16        ││ 21:14:02.114 ⚠ Outlier Feature Layer 0: Max > 6.0    │
│ Sparsity Rate: 🟩🟩🟩🟩🟩🟩🟩🟨⬜⬜⬜⬜⬜ 54.2%     ││ 21:14:02.128 ✖ CUDA OOM Fallback: Processing         │
│ Latency Delta: 1.142 ms (Within Normal Bounds)       ││ LayerNorm on CPU Host Memory.                        │
└──────────────────────────────────────────────────────┘└──────────────────────────────────────────────────────┘

(this serves only as an example, feel free to change the layout according to you)

**Other Notes**

- inspiration for TUI can be taken from lazygit or btop and using NerdFonts can help to get the clean but better symbols