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

#include <atomic>
#include <cstdint>

#include "../event.hpp"
#include "../ring_buffer.hpp"

struct ggml_tensor; // fwd-decl; ggml.h pulled in by the .cpp

namespace ts {

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

private:
    // Per-instance hot path, invoked from the static trampoline.
    bool capture(ggml_tensor * t);

    EventRing &           ring_;
    std::atomic<uint64_t> counter_{ 0 };
    std::atomic<uint64_t> prev_ns_{ 0 }; // steady_clock ns of the previous op
};

} // namespace ts
