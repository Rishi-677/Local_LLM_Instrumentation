// Local_LLM_Instrumentation — analysis export (Phase 7).
//
// Aggregates the telemetry stream into a compact per-(layer, op-class) summary
// and writes it as CSV or JSON for offline analysis (spreadsheets, plots). This
// is distinct from the raw NDJSON session log: it's a rollup, not every event.

#pragma once

#include <algorithm>
#include <fstream>
#include <map>
#include <string>
#include <utility>

#include "event.hpp"

namespace ts {

class Summary {
public:
    // Fold one event into the rollup keyed by (layer_idx, op_class).
    void add(const TensorEvent& e) {
        auto& s = rows_[{e.layer_idx, static_cast<int>(e.op_class)}];
        s.count += 1;
        s.total_latency_us += e.latency_us;
        if (e.stats_valid) {
            float a = std::max(e.v_max, -e.v_min);
            if (a > s.max_abs) s.max_abs = a;
            s.sparsity_sum += e.sparsity;
            s.stat_count += 1;
        }
        s.nan = s.nan || e.has_nan;
        s.inf = s.inf || e.has_inf;
    }

    bool write_csv(const std::string& path) const {
        std::ofstream f(path);
        if (!f) return false;
        f << "layer_idx,op_class,count,total_latency_us,avg_latency_us,"
             "max_abs,avg_sparsity,nan,inf\n";
        for (const auto& [key, s] : rows_) {
            f << key.first << "," << to_string(static_cast<OpClass>(key.second))
              << "," << s.count << "," << s.total_latency_us << ","
              << (s.count ? s.total_latency_us / s.count : 0) << ","
              << s.max_abs << ","
              << (s.stat_count ? s.sparsity_sum / s.stat_count : 0.0) << ","
              << (s.nan ? 1 : 0) << "," << (s.inf ? 1 : 0) << "\n";
        }
        return true;
    }

    bool write_json(const std::string& path) const {
        std::ofstream f(path);
        if (!f) return false;
        f << "[\n";
        bool first = true;
        for (const auto& [key, s] : rows_) {
            if (!first) f << ",\n";
            first = false;
            f << "  {\"layer_idx\":" << key.first
              << ",\"op_class\":\"" << to_string(static_cast<OpClass>(key.second))
              << "\",\"count\":" << s.count
              << ",\"total_latency_us\":" << s.total_latency_us
              << ",\"avg_latency_us\":" << (s.count ? s.total_latency_us / s.count : 0)
              << ",\"max_abs\":" << s.max_abs
              << ",\"avg_sparsity\":" << (s.stat_count ? s.sparsity_sum / s.stat_count : 0.0)
              << ",\"nan\":" << (s.nan ? "true" : "false")
              << ",\"inf\":" << (s.inf ? "true" : "false") << "}";
        }
        f << "\n]\n";
        return true;
    }

    size_t rows() const { return rows_.size(); }

private:
    struct Row {
        uint64_t count = 0;
        uint64_t total_latency_us = 0;
        float    max_abs = 0.0f;
        double   sparsity_sum = 0.0;
        uint64_t stat_count = 0;
        bool     nan = false;
        bool     inf = false;
    };
    std::map<std::pair<int32_t, int>, Row> rows_;
};

} // namespace ts
