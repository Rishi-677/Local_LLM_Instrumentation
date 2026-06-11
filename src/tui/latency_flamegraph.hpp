// Local_LLM_Instrumentation - latency flamegraph model.
//
// Builds a small, sorted view model from the topology rows the TUI already
// owns. The renderer can turn these rows into ASCII bars without knowing how
// topology nodes are accumulated.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include "../event.hpp"

namespace ts {

struct LatencyFlameRow {
    int32_t     layer_idx = -1;
    std::string label;
    uint64_t    total_latency_us = 0;
};

inline bool is_latency_flame_bucket(const LayerNode& n) {
    return n.label.rfind("layers.", 0) == 0 || n.label == "embed" ||
           n.label == "output" || n.label == "other";
}

inline std::vector<LatencyFlameRow> build_latency_flamegraph(
    const std::vector<LayerNode>& topology, std::size_t limit = 10) {
    std::vector<LatencyFlameRow> rows;
    rows.reserve(topology.size());

    for (const LayerNode& n : topology) {
        if (!is_latency_flame_bucket(n) || n.total_latency_us == 0) continue;
        rows.push_back(LatencyFlameRow{n.layer_idx, n.label, n.total_latency_us});
    }

    std::stable_sort(rows.begin(), rows.end(), [](const LatencyFlameRow& a,
                                                  const LatencyFlameRow& b) {
        if (a.total_latency_us != b.total_latency_us) {
            return a.total_latency_us > b.total_latency_us;
        }
        return a.label < b.label;
    });

    if (limit > 0 && rows.size() > limit) rows.resize(limit);
    return rows;
}

inline uint64_t latency_flamegraph_total(const std::vector<LatencyFlameRow>& rows) {
    uint64_t total = 0;
    for (const LatencyFlameRow& row : rows) total += row.total_latency_us;
    return total;
}

inline std::string latency_ascii_bar(uint64_t value, uint64_t max_value,
                                     std::size_t width = 24) {
    if (width == 0) return {};
    if (value == 0 || max_value == 0) return std::string(width, '.');

    const long double ratio =
        static_cast<long double>(value) / static_cast<long double>(max_value);
    std::size_t filled = static_cast<std::size_t>(ratio * width + 0.999L);
    if (filled == 0) filled = 1;
    if (filled > width) filled = width;

    return std::string(filled, '#') + std::string(width - filled, '.');
}

} // namespace ts
