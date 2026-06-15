// Local_LLM_Instrumentation — core telemetry schema (C1).
// `TensorEvent` is the single contract that flows from the capture layer
// (cb_eval hook) through the ring buffer into every TUI pane. It is a plain
// trivially-copyable struct on purpose: this is the seam where a future
// out-of-process sidecar (PyTorch/TF) will serialize the same shape over a
// socket. Keep it POD — no heap-owning members on the hot path.

#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace ts {
// Coarse classification derived from a tensor/op name. Drives pane grouping
// and the topology tree. Kept small and stable.
enum class OpClass : uint8_t {
    Embed,   // token / position embeddings
    Attn,    // attention (qkv, scores, softmax, output proj)
    MLP,     // feed-forward / MLP / SwiGLU
    Norm,    // layer norm / rms norm
    Output,  // final projection / logits
    Other,
};

// Where the op executed. Used by the anomaly layer to flag unexpected
// CPU fallbacks / device transfers.
enum class Device : uint8_t {
    CPU,
    CUDA,
    Metal,
    Other,
};

// Serialization format for recorded sessions.
enum class RecordFormat : uint8_t {
    NDJSON,    // human-readable newline-delimited JSON (default)
    LLMTRACE,  // compact binary .llmtrace — raw LayerEvent structs
};

constexpr int kMaxDims = 4;   // ggml tensors are up to 4D
constexpr int kNameLen = 64;  // matches ggml's GGML_MAX_NAME

// The core data model is split into three locked-down shapes. Everything
// downstream depends on these — keep them trivially copyable (the sidecar /
// session seam) and additive-only.
//
//   TensorMeta  — static description of one tensor (name, shape, dtype, device)
//   ActivationStats — cheap numeric summary computed from its values
//   LayerEvent — one observed op in a forward pass = meta + stats + timing
//
// LayerEvent composes the two via (public) inheritance so callers can read flat
// fields (ev.shape, ev.v_min) AND pass the sub-shapes (TensorMeta, ActivationStats)
// around independently, e.g. to a stats engine or serializer.

// Static description of a single tensor.
struct TensorMeta {
    char node_name[kNameLen] = {};  // ggml tensor name, e.g. "ffn_out-12"
    char op_name[kNameLen] = {};    // ggml op name, e.g. "MUL_MAT"
    std::array<int64_t, kMaxDims> shape = {0, 0, 0, 0};
    int32_t dtype = 0;       // ggml_type enum value
    int32_t dtype_size = 0;  // bytes per element
    Device device = Device::CPU;
};

// Lightweight numeric summary of a tensor's values (computed in the callback).
struct ActivationStats {
    float v_min = 0.0f;
    float v_max = 0.0f;
    float v_mean = 0.0f;
    float v_std = 0.0f;     // population standard deviation
    float l2_norm = 0.0f;   // sqrt(sum of squares) over the scan
    float sparsity = 0.0f;  // fraction of elements ≈ 0
    bool has_nan = false;
    bool has_inf = false;
    bool stats_valid = false;  // false if data wasn't read (sampled out)
};

// One observed tensor op in a single forward pass: tensor metadata + activation
// stats + event-level identity/timing.
struct LayerEvent : TensorMeta, ActivationStats {
    uint64_t id = 0;            // monotonic sequence number
    uint64_t timestamp_ns = 0;  // capture time (steady clock)
    int32_t layer_idx = -1;     // -1 = not layer-scoped (embed/output)
    OpClass op_class = OpClass::Other;
    uint64_t latency_us = 0;  // time since previous op on this thread
    // Heavy payload (e.g. an attention matrix slice) is captured ONLY for the
    // user-selected target, out-of-band, and referenced by index. 0 = none.
    uint32_t payload_id = 0;
};

// Back-compat alias: the codebase grew up calling this TensorEvent.
using TensorEvent = LayerEvent;
static_assert(std::is_trivially_copyable<LayerEvent>::value,
              "LayerEvent must stay trivially copyable (session / sidecar seam)");
static_assert(std::is_trivially_copyable<TensorMeta>::value, "");
static_assert(std::is_trivially_copyable<ActivationStats>::value, "");

// Topology: assembled incrementally from observed node names. Not on the hot
// path — lives on the TUI side and is updated as events arrive.
struct LayerNode {
    int32_t layer_idx = -1;
    std::string label;  // display name, e.g. "layers.12"
    OpClass op_class = OpClass::Other;
    bool selected = false;  // current capture target
    uint64_t last_seen_id = 0;
    uint64_t total_latency_us = 0;  // accumulated across this block
    std::vector<LayerNode> children;
};

// Out-of-band heavy payload (attention matrix slice for the selected target).
struct AttentionPayload {
    uint32_t payload_id = 0;
    int32_t layer_idx = -1;
    int32_t head = 0;
    int32_t rows = 0;            // query tokens
    int32_t cols = 0;            // key tokens
    std::vector<float> weights;  // row-major, rows*cols
};

inline const char* to_string(OpClass c) {
    switch (c) {
        case OpClass::Embed:
            return "Embed";
        case OpClass::Attn:
            return "Attn";
        case OpClass::MLP:
            return "MLP";
        case OpClass::Norm:
            return "Norm";
        case OpClass::Output:
            return "Output";
        default:
            return "Other";
    }
}

inline const char* to_string(Device d) {
    switch (d) {
        case Device::CPU:
            return "CPU";
        case Device::CUDA:
            return "CUDA";
        case Device::Metal:
            return "Metal";
        default:
            return "Other";
    }
}

}  // namespace ts
