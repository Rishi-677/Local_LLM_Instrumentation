// Local_LLM_Instrumentation — sidecar receiver (PyTorch sidecar).
// Listens on a TCP socket for NDJSON telemetry events from the Python
// sidecar process, parses them into TensorEvent structs, and pushes
// them into the ring buffer — the same pipeline that the live cb_eval
// hook uses. This is the "seam" that event.hpp was designed for:
// out-of-process sidecar → socket → receiver → ring → TUI
// Thread safety: the receiver runs on its own thread (the producer side
// of the ring). It owns the socket and parsing state.

#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include "../event.hpp"
#include "../ring_buffer.hpp"

namespace ts {
class AttentionSink;
// Flat topology entry sent from the sidecar (mirrors LayerNode for the
// Python side, kept minimal).
struct SidecarTopologyEntry {
    int32_t layer_idx = -1;
    std::string label;
    int op_class = 5;  // OpClass::Other
    uint64_t total_latency_us = 0;
};

class SidecarReceiver {
public:
    struct Config {
        std::string host = "127.0.0.1";
        int port = 9876;
    };
    explicit SidecarReceiver(EventRing& ring, Config cfg);
    ~SidecarReceiver();

    // Run the receiver loop: accept a connection, read NDJSON lines,
    // parse into TensorEvent, push to ring. Blocks until the connection
    // drops or `alive` becomes false. Intended to run on a dedicated thread.
    void run(std::atomic<bool>& alive);

    // Attach an attention sink so attention-weight lines populate the
    // heatmap pipeline. Optional; no attention capture when unset.
    void set_attention_sink(AttentionSink& sink) {
        attn_sink_ = &sink;
    }

    // Access the last-received topology snapshot (populated by the
    // receiver as it processes "ready" / topology-type events).
    std::vector<SidecarTopologyEntry> topology_snapshot() const;

    // Number of events received.
    uint64_t events_received() const {
        return received_.load();
    }

private:
    // Parse a single NDJSON line into a TensorEvent. Returns false on
    // malformed input (line is silently skipped).
    bool parse_line(const std::string& line, TensorEvent& out);

    // Low-level NDJSON scanner helpers (mirror replay.cpp internals).
    static void skip_ws(const std::string& s, size_t& i);
    static bool parse_string(const std::string& s, size_t& i, std::string& out);
    static std::string read_token(const std::string& s, size_t& i);
    static void copy_into(char (&dst)[kNameLen], const std::string& src);
    static bool parse_float_array(const std::string& s, size_t& i, std::vector<float>& out);

    // Extract an attention payload from an already-parsed TensorEvent and
    // the raw NDJSON line. Returns false if the event has no attention data.
    static bool extract_attention(const std::string& line, const TensorEvent& ev,
                                  AttentionPayload& out);

    EventRing& ring_;
    Config cfg_;
    std::atomic<uint64_t> received_{0};
    std::atomic<int> fd_{-1};
    mutable std::mutex topo_mu_;
    std::vector<SidecarTopologyEntry> topo_;
    AttentionSink* attn_sink_ = nullptr;
};

}  // namespace ts
