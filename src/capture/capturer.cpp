#include "capturer.hpp"

#include <chrono>
#include <cmath>
#include <cstring>

#include "ggml.h"
#include "ggml-backend.h"

#include "topology.hpp"

namespace ts {

namespace {

constexpr size_t kMaxScan = 65536; // cap element scan in the hot path

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

// Best-effort device from the tensor's backend buffer name ("CPU", "CUDA0",
// "Metal", ...). CPU when unknown — this milestone is CPU-only anyway.
Device device_of(const ggml_tensor * t) {
    if (!t->buffer) return Device::CPU;
    const char * bn = ggml_backend_buffer_name(t->buffer);
    if (!bn) return Device::CPU;
    if (std::strstr(bn, "CUDA")) return Device::CUDA;
    if (std::strstr(bn, "Metal")) return Device::Metal;
    if (std::strstr(bn, "CPU")) return Device::CPU;
    return Device::Other;
}

// Cheap F32 numeric summary over (a sample of) the contiguous tensor data.
void fill_stats(const ggml_tensor * t, TensorEvent & e) {
    const int64_t n = ggml_nelements(t);
    if (n <= 0) return;

    const float * data = static_cast<const float *>(t->data);

    // Sample with a stride if the tensor is larger than the scan cap.
    int64_t stride = 1;
    if (static_cast<size_t>(n) > kMaxScan) {
        stride = n / static_cast<int64_t>(kMaxScan);
        if (stride < 1) stride = 1;
    }

    float    vmin = data[0];
    float    vmax = data[0];
    double   sum  = 0.0;
    uint64_t count = 0;
    uint64_t zeros = 0;
    bool     nan = false;
    bool     inf = false;

    for (int64_t i = 0; i < n; i += stride) {
        const float v = data[i];
        if (std::isnan(v)) { nan = true; continue; }
        if (std::isinf(v)) { inf = true; continue; }
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        sum += v;
        if (std::fabs(v) < 1e-6f) ++zeros;
        ++count;
    }

    e.v_min    = vmin;
    e.v_max    = vmax;
    e.v_mean   = count ? static_cast<float>(sum / static_cast<double>(count)) : 0.0f;
    e.sparsity = count ? static_cast<float>(zeros) / static_cast<float>(count) : 0.0f;
    e.has_nan  = nan;
    e.has_inf  = inf;
    e.stats_valid = true;
}

} // namespace

bool Capturer::on_eval(ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return true; // observe every node
    }
    auto * self = static_cast<Capturer *>(user_data);
    if (!self || !t) return true;
    return self->capture(t);
}

bool Capturer::capture(ggml_tensor * t) {
    TensorEvent e;
    e.id           = counter_.fetch_add(1, std::memory_order_relaxed) + 1;
    e.timestamp_ns = now_ns();

    // Latency delta vs. the previous captured op.
    const uint64_t prev = prev_ns_.exchange(e.timestamp_ns, std::memory_order_relaxed);
    e.latency_us = (prev && e.timestamp_ns > prev)
                       ? (e.timestamp_ns - prev) / 1000u
                       : 0u;

    // Names (bounded copies into the fixed buffers).
    std::strncpy(e.node_name, t->name ? t->name : "", kNameLen - 1);
    const char * opn = ggml_op_name(t->op);
    std::strncpy(e.op_name, opn ? opn : "", kNameLen - 1);

    // Shape / dtype.
    for (int i = 0; i < kMaxDims; ++i) e.shape[i] = t->ne[i];
    e.dtype      = static_cast<int32_t>(t->type);
    e.dtype_size = static_cast<int32_t>(ggml_type_size(t->type));

    // Classification (class + layer index from the name).
    auto [cls, layer] = classify(t->name);
    e.op_class  = cls;
    e.layer_idx = layer;

    e.device = device_of(t);

    // Cheap stats only when we can safely read contiguous F32 host data.
    // Guard against device (e.g. CUDA) pointers, which are non-null but not
    // host-dereferenceable.
    const bool host_readable =
        t->data != nullptr &&
        (t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer));
    if (t->type == GGML_TYPE_F32 && host_readable && ggml_is_contiguous(t)) {
        fill_stats(t, e);
    }

    ring_.push(e); // SPSC; drops + counts on overflow, never blocks
    return true;
}

void Capturer::prime() {
    prev_ns_.store(0, std::memory_order_relaxed);
}

} // namespace ts
