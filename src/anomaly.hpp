// Local_LLM_Instrumentation — anomaly detection (C3).
//
// Stateless-per-event rule engine that inspects each TensorEvent as it drains
// from the ring and surfaces a single, highest-priority AnomalyRecord. The
// detector keeps a small amount of state: a capped ledger of everything it has
// flagged (for the "anomalies" pane / session summary) and a dedupe set so a
// repeated condition (e.g. CPU fallback on every node) is reported once per
// node rather than once per forward pass.
//
// Depends only on event.hpp + the standard library.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "event.hpp"

namespace ts {

struct AnomalyRecord {
    uint64_t    timestamp_ns = 0;
    int         severity     = 0;   // 0 = info, 1 = warn, 2 = error
    std::string text;
};

class AnomalyDetector {
public:
    struct Config {
        float extreme_abs = 1e4f;       // |v_min|/|v_max| above this → outlier warn

        // CPU-fallback flagging is OFF by default. Rationale: the MVP target is
        // a fully-CPU llama.cpp run, where *every* op executes on the host —
        // flagging each one (even deduped per node) would bury the genuinely
        // useful NaN/Inf/outlier signals under hundreds of "CPU fallback"
        // warnings. It is opt-in for mixed CPU/GPU runs, where a stray CPU op
        // among GPU ops is a real, actionable finding.
        bool  flag_cpu_fallback = false;
    };

    // Two-form construction instead of a single `Config cfg = {}` default
    // argument: a nested aggregate's default member initializers are not yet
    // available inside its own enclosing class, so `= {}` / `= Config{}` as a
    // default argument fails to compile (GCC 15). The no-arg ctor delegates
    // with a default-constructed Config, giving callers the same ergonomics:
    //   AnomalyDetector det;                 // defaults
    //   AnomalyDetector det(AnomalyDetector::Config{1e5f, true});
    AnomalyDetector() : AnomalyDetector(Config{}) {}
    explicit AnomalyDetector(Config cfg);

    // Apply the rules in priority order and return the highest-priority match,
    // or std::nullopt if the event is clean. Any emitted record is appended to
    // the ledger() before being returned.
    std::optional<AnomalyRecord> inspect(const ts::TensorEvent & e);

    void set_threshold(float extreme_abs);

    const std::vector<AnomalyRecord> & ledger() const { return ledger_; }

private:
    AnomalyRecord push(uint64_t ts, int severity, std::string text);

    Config                          cfg_;
    std::vector<AnomalyRecord>      ledger_;
    std::unordered_set<std::string> cpu_seen_;       // dedupe CPU-fallback per node
    std::unordered_set<std::string> outlier_seen_;   // dedupe outliers per node

    static constexpr size_t kLedgerCap = 1000;
};

} // namespace ts
