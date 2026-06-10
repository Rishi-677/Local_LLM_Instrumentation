// Local_LLM_Instrumentation — capture backbone (C4).
//
// Bridges llama.cpp's ggml_backend_sched eval callback to our TensorEvent
// stream. The callback runs on the inference thread (the producer side of the
// SPSC ring); it must stay cheap and allocation-free so it never stalls the
// forward pass.
//
// Wiring (lead integrates):
//   ts::Capturer cap(ring);
//   ctx_params.cb_eval            = &ts::Capturer::on_eval;
//   ctx_params.cb_eval_user_data  = &cap;
// The Capturer must outlive the llama_context.

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>

#include "../event.hpp"
#include "../ring_buffer.hpp"

struct ggml_tensor; // fwd-decl; ggml.h pulled in by the .cpp

namespace ts {

// Out-of-band channel for the heavy attention payload. The SPSC ring carries
// only POD TensorEvents; the 2D attention matrix (a slice of the kq_soft_max
// tensor for the selected layer/head) is published here and picked up by the
// consumer thread. `target_layer`/`target_head` are set from the UI thread.
class AttentionSink {
public:
    std::atomic<int> target_layer{-1};   // -1 = capture nothing
    std::atomic<int> target_head{0};

    void publish(const AttentionPayload & p) {
        std::lock_guard<std::mutex> lk(mu_);
        latest_ = p;
        has_    = true;
    }
    // Move the latest payload out (consume-once). Returns false if none pending.
    bool take(AttentionPayload & out) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!has_) return false;
        out  = latest_;
        has_ = false;
        return true;
    }

private:
    std::mutex       mu_;
    AttentionPayload latest_;
    bool             has_ = false;
};

class Capturer {
public:
    explicit Capturer(EventRing & ring) : ring_(ring) {}

    // ggml_backend_sched_eval_callback. ask==true: return true to observe the
    // node. ask==false: the node is computed -> build a TensorEvent and push.
    // `user_data` must be the owning Capturer*.
    static bool on_eval(ggml_tensor * t, bool ask, void * user_data);

    // Reset the previous-op timestamp at the start of a forward pass so the
    // first op's latency_us isn't polluted by inter-pass idle time. Optional.
    void prime();

    // Attach an attention sink so the target layer's kq_soft_max matrix is
    // captured out-of-band. Optional; null disables attention capture.
    void set_attention_sink(AttentionSink * sink) { attn_ = sink; }

    // Per-OpClass capture mask (indexed by static_cast<int>(OpClass)). Events
    // whose class is masked off are skipped before the ring enqueue, keeping
    // them off the hot path entirely.
    void set_capture_mask(const std::array<bool, 6> & mask) { mask_ = mask; }

private:
    // Per-instance hot path, invoked from the static trampoline.
    bool capture(ggml_tensor * t);

    EventRing &           ring_;
    AttentionSink *       attn_ = nullptr;
    std::array<bool, 6>   mask_ = {true, true, true, true, true, true};
    std::atomic<uint64_t> counter_{ 0 };
    std::atomic<uint64_t> prev_ns_{ 0 }; // steady_clock ns of the previous op

    // Prefer the richer (square, prefill) attention matrix: don't let thin
    // single-token generation rows overwrite it unless the target layer changed.
    int last_attn_layer_ = -999;
    int last_attn_rows_  = 0;
};

} // namespace ts
