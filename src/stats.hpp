// Local_LLM_Instrumentation — activation statistics (Phase 2).
//
// A pure, testable function that computes an ActivationStats summary from a
// contiguous float buffer. Extracted from the capture hot path so it can be
// unit-tested in isolation and reused by any future backend. Large buffers are
// sampled with a stride to keep the per-op cost bounded.

#pragma once

#include <cmath>
#include <cstdint>

#include "event.hpp"

namespace ts {

// Compute min / max / mean / std / L2 norm / sparsity over `data[0..n)`.
// NaN/Inf are flagged and excluded from the numeric reductions. If n exceeds
// `max_scan`, elements are sampled with a stride so the scan stays bounded.
inline ActivationStats compute_stats(const float* data, int64_t n,
                                     int64_t max_scan = 65536) {
    ActivationStats s;
    if (!data || n <= 0 || max_scan <= 0) return s;  // max_scan<=0 disables stats

    int64_t stride = (n > max_scan) ? (n / max_scan) : 1;
    if (stride < 1) stride = 1;

    float    vmin = data[0];
    float    vmax = data[0];
    double   sum   = 0.0;
    double   sumsq = 0.0;
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
        sum   += v;
        sumsq += static_cast<double>(v) * v;
        if (std::fabs(v) < 1e-6f) ++zeros;
        ++count;
    }

    if (count > 0) {
        const double mean = sum / static_cast<double>(count);
        const double var  = sumsq / static_cast<double>(count) - mean * mean;
        s.v_min    = vmin;
        s.v_max    = vmax;
        s.v_mean   = static_cast<float>(mean);
        s.v_std    = static_cast<float>(std::sqrt(var > 0.0 ? var : 0.0));
        s.l2_norm  = static_cast<float>(std::sqrt(sumsq));
        s.sparsity = static_cast<float>(zeros) / static_cast<float>(count);
    }
    s.has_nan     = nan;
    s.has_inf     = inf;
    s.stats_valid = true;
    return s;
}

} // namespace ts
