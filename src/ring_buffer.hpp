// Local_LLM_Instrumentation — lock-free SPSC ring buffer (C2).
// Single producer  = the cb_eval hook (inference thread).
// Single consumer  = the TUI render thread.
// The producer must never block inference, so on overflow we DROP the newest
// event and bump a counter the UI surfaces ("dropped N") — honest sampling,
// never a silent stall. Capacity is fixed (matches the PS ring-buffer idea).

#pragma once
#include <atomic>
#include <cstddef>
#include <vector>
#include "event.hpp"

namespace ts {

template <typename T>
class SpscRing {
public:
    // capacity is rounded up to a power of two so head/tail wrap with a mask.
    explicit SpscRing(size_t capacity = 1u << 14) {
        size_t cap = 1;
        while (cap < capacity)
            cap <<= 1;
        cap_ = cap;
        mask_ = cap - 1;
        buf_.resize(cap_);
    }

    // Producer side. Returns false (and increments dropped_) if full.
    bool push(const T& item) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next = head + 1;
        if (next - tail_.load(std::memory_order_acquire) > cap_) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        buf_[head & mask_] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Consumer side. Returns false if empty.
    bool pop(T& out) {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return false;  // empty
        }
        out = buf_[tail & mask_];
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    // Consumer side: drain up to `max` items into `sink`. Returns count drained.
    template <typename Sink>
    size_t drain(Sink&& sink, size_t max = SIZE_MAX) {
        size_t count = 0;
        T item;
        while (count < max && pop(item)) {
            sink(item);
            ++count;
        }
        return count;
    }

    size_t dropped() const {
        return dropped_.load(std::memory_order_relaxed);
    }
    size_t capacity() const {
        return cap_;
    }
    size_t size_approx() const {
        return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire);
    }

private:
    size_t cap_ = 0;
    size_t mask_ = 0;
    std::vector<T> buf_;
    alignas(64) std::atomic<size_t> head_{0};  // producer writes
    alignas(64) std::atomic<size_t> tail_{0};  // consumer writes
    alignas(64) std::atomic<size_t> dropped_{0};
};

using EventRing = SpscRing<TensorEvent>;

}  // namespace ts
