// Local_LLM_Instrumentation — thread-shared UI snapshot (C-TUI).
// `UiState` is the single seam between the lead's consumer thread (which drains
// the SPSC ring and writes here) and the FTXUI render thread (which reads here
// under the same lock). It owns ONLY plain copies derived from `event.hpp` plus
// std types — no dependency on any other Local_LLM_Instrumentation module, so the TUI stays
// decoupled and independently testable.
// Threading contract:
//   - Every member is guarded by `mu`. Writers use the guarded setters.
//   - The render thread calls `snapshot()` once per frame to grab a cheap copy
//     of exactly what it draws, minimizing lock hold time. Never hold the lock
//     across FTXUI rendering.

#pragma once
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>
#include "../event.hpp"

namespace ts {

// A single entry in the numerical anomaly ledger (pane 5). Assembled on the UI
// side from events whose stats flagged NaN/Inf/unexpected device fallbacks.
struct Anomaly {
    uint64_t ts_ns = 0;
    int severity = 0;  // 1 = warn (⚠), 2 = error (✖)
    std::string text;
};

constexpr size_t kStreamCap = 500;
constexpr size_t kAnomalyCap = 200;

class UiState {
public:
    struct Snapshot {
        std::deque<TensorEvent> stream;
        std::vector<LayerNode> topology;
        int selected = 0;
        TensorEvent selected_metrics{};
        bool has_metrics = false;
        AttentionPayload attention{};
        bool has_attention = false;
        std::vector<Anomaly> anomalies;
        std::string model_name;
        std::string session_state = "LIVE";
        uint64_t dropped = 0;
        uint64_t total_events = 0;
        int n_tokens = 0;
    };

    // Append one event to the live stream, trimming to the last kStreamCap.
    // Also bumps total_events. Does NOT touch selected_metrics — call
    // set_selected_metrics() explicitly for the focused layer.
    void push_event(const TensorEvent& ev) {
        std::lock_guard<std::mutex> lk(mu_);
        stream_.push_back(ev);
        while (stream_.size() > kStreamCap)
            stream_.pop_front();
        ++total_events_;
    }

    void set_topology(std::vector<LayerNode> topo) {
        std::lock_guard<std::mutex> lk(mu_);
        topology_ = std::move(topo);
        if (selected_ >= static_cast<int>(topology_.size())) {
            selected_ = topology_.empty() ? 0 : static_cast<int>(topology_.size()) - 1;
        }
    }

    void set_selected_metrics(const TensorEvent& ev) {
        std::lock_guard<std::mutex> lk(mu_);
        selected_metrics_ = ev;
        has_metrics_ = true;
    }

    void set_attention(const AttentionPayload& att) {
        std::lock_guard<std::mutex> lk(mu_);
        attention_ = att;
        has_attention_ = true;
    }

    void clear_attention() {
        std::lock_guard<std::mutex> lk(mu_);
        attention_ = AttentionPayload{};
        has_attention_ = false;
    }

    void push_anomaly(const Anomaly& a) {
        std::lock_guard<std::mutex> lk(mu_);
        anomalies_.push_back(a);
        while (anomalies_.size() > kAnomalyCap)
            anomalies_.erase(anomalies_.begin());
    }

    void set_model_name(std::string name) {
        std::lock_guard<std::mutex> lk(mu_);
        model_name_ = std::move(name);
    }

    void set_session_state(std::string state) {
        std::lock_guard<std::mutex> lk(mu_);
        session_state_ = std::move(state);
    }

    void set_dropped(uint64_t n) {
        std::lock_guard<std::mutex> lk(mu_);
        dropped_ = n;
    }

    void set_n_tokens(int n) {
        std::lock_guard<std::mutex> lk(mu_);
        n_tokens_ = n;
    }

    // Selection is owned by the UI but exposed so the App can move it on j/k.
    void set_selected(int idx) {
        std::lock_guard<std::mutex> lk(mu_);
        const int n = static_cast<int>(topology_.size());
        if (n == 0) {
            selected_ = 0;
            return;
        }
        if (idx < 0)
            idx = 0;
        if (idx >= n)
            idx = n - 1;
        selected_ = idx;
    }

    int selected() const {
        std::lock_guard<std::mutex> lk(mu_);
        return selected_;
    }

    int topology_size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<int>(topology_.size());
    }

    // Single cheap copy of everything a frame needs. Lock is held only for the
    // duration of the copies, never across rendering.
    Snapshot snapshot() const {
        std::lock_guard<std::mutex> lk(mu_);
        Snapshot s;
        s.stream = stream_;
        s.topology = topology_;
        s.selected = selected_;
        s.selected_metrics = selected_metrics_;
        s.has_metrics = has_metrics_;
        s.attention = attention_;
        s.has_attention = has_attention_;
        s.anomalies = anomalies_;
        s.model_name = model_name_;
        s.session_state = session_state_;
        s.dropped = dropped_;
        s.total_events = total_events_;
        s.n_tokens = n_tokens_;
        return s;
    }

private:
    mutable std::mutex mu_;
    std::deque<TensorEvent> stream_;
    std::vector<LayerNode> topology_;
    int selected_ = 0;
    TensorEvent selected_metrics_{};
    bool has_metrics_ = false;
    AttentionPayload attention_{};
    bool has_attention_ = false;
    std::vector<Anomaly> anomalies_;
    std::string model_name_;
    std::string session_state_ = "LIVE";
    uint64_t dropped_ = 0;
    uint64_t total_events_ = 0;
    int n_tokens_ = 0;
};

}  // namespace ts
