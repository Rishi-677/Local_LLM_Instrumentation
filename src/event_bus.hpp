// Local_LLM_Instrumentation — typed publish/subscribe event bus (C-bus).
//
// A small, header-only, thread-safe fan-out for in-process signalling between
// the TUI panes, the capture/control layer, and the anomaly engine. Unlike the
// SPSC ring buffer (the hot inference path), this bus is for low-frequency
// control-plane events ("target selected", "pane focused", "session paused")
// where ergonomics and ordering matter more than nanoseconds.
//
// Two delivery modes share the same priority-ordered fan-out:
//   publish(e)       — synchronous: invoke every current handler right now.
//   enqueue(e, p)    — deferred: stage onto an internal queue; the owner later
//                      calls dispatch_pending() (e.g. once per render frame) to
//                      drain it on a single, controlled thread.
//
// Priority (High → Normal → Low) governs handler invocation order, and — for
// the deferred path — the order in which queued events themselves are drained.
// Within one priority level, ordering is stable (subscription / enqueue order).
//
// Re-entrancy / deadlock safety: user handlers may subscribe, unsubscribe, or
// publish from inside a handler. So we NEVER hold the mutex while invoking a
// handler — we snapshot the handler list under the lock, release it, then call.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace ts {

// Relative handler importance. Higher value = invoked earlier. The integral
// values are load-bearing: we sort by them (descending) for the fan-out.
enum class Priority : int {
    Low    = 0,
    Normal = 1,
    High   = 2,
};

template <typename Event>
class EventBus {
public:
    using Handler = std::function<void(const Event&)>;
    using Token   = std::uint64_t;

    EventBus() = default;

    // Non-copyable / non-movable: a bus has identity (subscribers hold tokens
    // against *this* instance) and owns a mutex.
    EventBus(const EventBus&)            = delete;
    EventBus& operator=(const EventBus&) = delete;

    // Register a handler. Returns an opaque, never-reused token for unsubscribe.
    // Handlers at the same priority are invoked in subscription order.
    Token subscribe(Handler h, Priority p = Priority::Normal) {
        std::lock_guard<std::mutex> lk(mu_);
        const Token t = next_token_++;
        // seq_ is a monotonic tiebreaker so we can keep a stable, priority-major
        // order even though subscribe/unsubscribe churn the vector over time.
        subs_.push_back(Sub{t, p, seq_++, std::move(h)});
        return t;
    }

    // Remove a handler by token. Returns true if a matching handler existed.
    bool unsubscribe(Token t) {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = std::find_if(subs_.begin(), subs_.end(),
                                     [t](const Sub& s) { return s.token == t; });
        if (it == subs_.end()) return false;
        subs_.erase(it);
        return true;
    }

    // Synchronously invoke every current handler, in priority order (High →
    // Normal → Low; stable within a level). The handler list is snapshotted
    // under the lock and invoked *after* releasing it, so handlers may safely
    // re-enter the bus (subscribe / unsubscribe / publish).
    void publish(const Event& e) const {
        std::vector<Handler> ordered = snapshot_ordered();
        for (const Handler& h : ordered) {
            h(e);
        }
    }

    // Stage an event for deferred delivery. Cheap and lock-guarded; does NOT
    // invoke any handler. Drained later by dispatch_pending().
    void enqueue(const Event& e, Priority p = Priority::Normal) {
        std::lock_guard<std::mutex> lk(mu_);
        pending_.push_back(Pending{p, eseq_++, e});
    }

    // Drain the internal queue: deliver all queued events, High-priority events
    // first (stable by enqueue order within a level), each fanned out through
    // the same priority-ordered handler invocation as publish(). Returns the
    // number of events dispatched.
    //
    // We take a single snapshot of the events AND the ordered handler list up
    // front, then invoke outside the lock. Events enqueued by a handler during
    // this drain are left for the next dispatch_pending() — bounded, predictable
    // work per call (no live-lock from self-feeding handlers).
    std::size_t dispatch_pending() {
        std::vector<Pending> batch;
        std::vector<Handler> ordered;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (pending_.empty()) return 0;
            batch.swap(pending_);          // take the whole queue; leave it empty
            ordered = ordered_handlers_locked();
        }

        // Priority-major drain, stable by enqueue sequence within each level.
        std::sort(batch.begin(), batch.end(), [](const Pending& a, const Pending& b) {
            if (a.prio != b.prio)
                return static_cast<int>(a.prio) > static_cast<int>(b.prio);
            return a.seq < b.seq;
        });

        for (const Pending& item : batch) {
            for (const Handler& h : ordered) {
                h(item.event);
            }
        }
        return batch.size();
    }

    // Number of currently registered handlers.
    std::size_t subscriber_count() const {
        std::lock_guard<std::mutex> lk(mu_);
        return subs_.size();
    }

    // Number of events staged via enqueue() and not yet dispatched.
    std::size_t pending_count() const {
        std::lock_guard<std::mutex> lk(mu_);
        return pending_.size();
    }

    // Drop all handlers and all pending events. Tokens are not reused.
    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        subs_.clear();
        pending_.clear();
    }

private:
    // One registered subscriber. `seq` preserves subscription order as a stable
    // tiebreaker behind `prio`.
    struct Sub {
        Token       token;
        Priority    prio;
        std::uint64_t seq;
        Handler     handler;
    };

    // One staged event awaiting deferred delivery.
    struct Pending {
        Priority      prio;
        std::uint64_t seq;
        Event         event;
    };

    // Build the priority-ordered handler list. Caller must hold mu_.
    std::vector<Handler> ordered_handlers_locked() const {
        std::vector<const Sub*> view;
        view.reserve(subs_.size());
        for (const Sub& s : subs_) view.push_back(&s);

        // High → Low by priority; subscription order within a priority. A stable
        // sort isn't required since `seq` already breaks every tie uniquely.
        std::sort(view.begin(), view.end(), [](const Sub* a, const Sub* b) {
            if (a->prio != b->prio)
                return static_cast<int>(a->prio) > static_cast<int>(b->prio);
            return a->seq < b->seq;
        });

        std::vector<Handler> out;
        out.reserve(view.size());
        for (const Sub* s : view) out.push_back(s->handler);
        return out;
    }

    // Lock + snapshot helper for publish() (which is const).
    std::vector<Handler> snapshot_ordered() const {
        std::lock_guard<std::mutex> lk(mu_);
        return ordered_handlers_locked();
    }

    mutable std::mutex   mu_;
    std::vector<Sub>     subs_;
    std::vector<Pending> pending_;
    Token                next_token_ = 1;  // 0 reserved as an "invalid" sentinel
    std::uint64_t        seq_  = 0;        // subscription-order counter
    std::uint64_t        eseq_ = 0;        // enqueue-order counter
};

} // namespace ts
