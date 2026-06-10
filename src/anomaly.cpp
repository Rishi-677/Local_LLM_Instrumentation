// Local_LLM_Instrumentation — anomaly detection implementation (C3).

#include "anomaly.hpp"

#include <cmath>
#include <string>

namespace ts {

namespace {

// node_name / op_name are fixed char[64] buffers that may or may not be
// NUL-terminated when full; build a bounded std::string safely.
std::string field_str(const char (&buf)[kNameLen]) {
    size_t n = 0;
    while (n < kNameLen && buf[n] != '\0') ++n;
    return std::string(buf, n);
}

} // namespace

AnomalyDetector::AnomalyDetector(Config cfg) : cfg_(cfg) {}

void AnomalyDetector::set_threshold(float extreme_abs) {
    cfg_.extreme_abs = extreme_abs;
}

AnomalyRecord AnomalyDetector::push(uint64_t ts, int severity, std::string text) {
    AnomalyRecord rec{ts, severity, std::move(text)};
    ledger_.push_back(rec);
    // Cap growth, dropping oldest. erase(begin) is fine: anomalies are rare
    // relative to events, so this is far off the hot path.
    if (ledger_.size() > kLedgerCap) {
        ledger_.erase(ledger_.begin(),
                      ledger_.begin() + (ledger_.size() - kLedgerCap));
    }
    return rec;
}

std::optional<AnomalyRecord> AnomalyDetector::inspect(const ts::TensorEvent & e) {
    const std::string node = field_str(e.node_name);

    // Rule 1: NaN — highest priority, hard error.
    if (e.has_nan) {
        return push(e.timestamp_ns, 2,
                    "NaN in " + node + " (layer " +
                        std::to_string(e.layer_idx) + ")");
    }

    // Rule 2: Inf — hard error.
    if (e.has_inf) {
        return push(e.timestamp_ns, 2, "Inf in " + node);
    }

    // Rule 3: extreme magnitude outlier (only if stats were actually read).
    // Deduped once per node: many tensors (e.g. pre-softmax kq scores) are
    // legitimately large on every layer/token, and flagging each occurrence
    // would bury everything else. We report the first time each distinct node
    // crosses the threshold.
    if (e.stats_valid &&
        (std::fabs(e.v_max) > cfg_.extreme_abs ||
         std::fabs(e.v_min) > cfg_.extreme_abs)) {
        if (outlier_seen_.insert(node).second) {
            return push(e.timestamp_ns, 1,
                        "Outlier Feature " + node + ": Max " +
                            std::to_string(e.v_max));
        }
        return std::nullopt;
    }

    // Rule 4: unexpected CPU fallback (opt-in; deduped once per node).
    if (cfg_.flag_cpu_fallback && e.device == ts::Device::CPU &&
        e.layer_idx >= 0 && e.op_class != ts::OpClass::Embed) {
        if (cpu_seen_.insert(node).second) {
            return push(e.timestamp_ns, 1,
                        "CPU fallback: " + field_str(e.op_name) +
                            " on host memory");
        }
    }

    return std::nullopt;
}

} // namespace ts
