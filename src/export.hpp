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
#include <vector>

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

    bool write_html(const std::string& path) const {
        // Sort rows by total_latency_us descending for the flamegraph.
        std::vector<std::pair<std::pair<int32_t, int>, Row>> sorted(rows_.begin(), rows_.end());
        std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) {
            return a.second.total_latency_us > b.second.total_latency_us;
        });

        uint64_t total_events = 0, total_latency = 0, max_latency = 0;
        for (auto& [k, s] : sorted) {
            total_events += s.count;
            total_latency += s.total_latency_us;
            if (s.total_latency_us > max_latency) max_latency = s.total_latency_us;
        }

        const int bar_area_w = 600;
        const char* colors[6] = {
            "#83a598", "#8ec07c", "#fabd2f", "#d3869b", "#fe8019", "#928374"
        };
        const char* opclass_label[6] = {
            "Embed", "Attn", "MLP", "Norm", "Output", "Other"
        };

        std::ofstream f(path);
        if (!f) return false;

        f << "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n"
          << "<title>Local_LLM_Instrumentation — Layer Summary</title>\n"
          << "<style>\n"
          << "body{background:#1d2021;color:#ebdbb2;font-family:monospace;padding:20px;max-width:960px;margin:0 auto}\n"
          << "h1{color:#fabd2f;border-bottom:2px solid #504945;padding-bottom:8px}\n"
          << "h2{color:#8ec07c;margin-top:32px}\n"
          << ".summary{background:#282828;padding:12px 16px;border-radius:4px;margin:16px 0}\n"
          << ".summary span{margin-right:24px}\n"
          << ".num{color:#83a598}\n"
          << "table{border-collapse:collapse;width:100%;margin-top:8px}\n"
          << "th,td{border:1px solid #504945;padding:4px 10px;text-align:right}\n"
          << "th{background:#3c3836;color:#fabd2f;position:sticky;top:0}\n"
          << "td:first-child{text-align:left}\n"
          << "tr:hover{background:#3c3836}\n"
          << ".bar{height:18px;border-radius:2px}\n"
          << ".legend{display:flex;gap:16px;margin:12px 0}\n"
          << ".legend span{display:flex;align-items:center;gap:4px}\n"
          << ".swatch{width:12px;height:12px;border-radius:2px;display:inline-block}\n"
          << "</style></head><body>\n"
          << "<h1>Local_LLM_Instrumentation — Layer Summary</h1>\n"
          << "<div class='summary'>\n"
          << "  <span>Events: <span class='num'>" << total_events << "</span></span>\n"
          << "  <span>Layers: <span class='num'>" << rows_.size() << "</span></span>\n"
          << "  <span>Total latency: <span class='num'>" << total_latency
          << "</span> \u00b5s</span>\n"
          << "  <span>Max bar: <span class='num'>" << max_latency
          << "</span> \u00b5s</span>\n"
          << "</div>\n";

        // Legend.
        f << "<div class='legend'>\n";
        for (int ci = 0; ci < 6; ++ci)
            f << "  <span><span class='swatch' style='background:" << colors[ci]
              << "'></span>" << opclass_label[ci] << "</span>\n";
        f << "</div>\n";

        // SVG flamegraph.
        int svg_h = static_cast<int>(sorted.size() * 22 + 20);
        f << "<svg width='100%' height='" << svg_h << "' viewBox='0 0 800 " << svg_h
          << "' style='background:#282828;border-radius:4px'>\n";

        for (size_t i = 0; i < sorted.size(); ++i) {
            auto& [key, s] = sorted[i];
            int ci = (key.second >= 0 && key.second < 6) ? key.second : 5;
            int bw = (max_latency > 0)
                         ? static_cast<int>(bar_area_w *
                                            static_cast<double>(s.total_latency_us) /
                                            max_latency)
                         : 1;
            int y = static_cast<int>(i * 22) + 2;

            char label[80];
            std::snprintf(label, sizeof(label), "L%d %s  %llu us  %llu ops",
                          key.first, opclass_label[ci],
                          (unsigned long long)s.total_latency_us,
                          (unsigned long long)s.count);

            f << "  <rect class='bar' x='180' y='" << y << "' width='" << bw
              << "' height='18' fill='" << colors[ci] << "' opacity='0.9'/>\n"
              << "  <text x='5' y='" << (y + 14)
              << "' fill='#ebdbb2' font-size='12' font-family='monospace'>"
              << label << "</text>\n";
        }
        f << "</svg>\n";

        // Data table.
        f << "<h2>Per-Layer Breakdown</h2>\n<table>\n"
          << "<tr><th>Layer</th><th>Op Class</th><th>Count</th>"
          << "<th>Total (\u00b5s)</th><th>Avg (\u00b5s)</th><th>Max|Abs</th>"
          << "<th>Avg Sparsity</th><th>NaN</th><th>Inf</th></tr>\n";

        for (auto& [key, s] : sorted) {
            f << "<tr><td>" << key.first << "</td>"
              << "<td>" << opclass_label[(key.second >= 0 && key.second < 6) ? key.second : 5]
              << "</td><td>" << s.count << "</td><td>" << s.total_latency_us << "</td><td>"
              << (s.count ? s.total_latency_us / s.count : 0) << "</td><td>" << s.max_abs
              << "</td><td>"
              << (s.stat_count ? s.sparsity_sum / s.stat_count : 0.0) << "</td>"
              << "<td>" << (s.nan ? "\u26a0" : "") << "</td>"
              << "<td>" << (s.inf ? "\u26a0" : "") << "</td></tr>\n";
        }

        f << "</table>\n</body></html>\n";
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
