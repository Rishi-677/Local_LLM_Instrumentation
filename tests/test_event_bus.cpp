// Local_LLM_Instrumentation — unit tests for the typed event bus (src/event_bus.hpp).
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <vector>
#include "event_bus.hpp"
using ts::EventBus;
using ts::Priority;

namespace {
// Simple POD event used throughout these tests.
struct PingEvent {
    int id = 0;
    int value = 0;
};

}  // namespace

TEST_CASE("subscribe + publish delivers to all handlers", "[event_bus]") {
    EventBus<PingEvent> bus;
    int a = 0, b = 0, c = 0;
    bus.subscribe([&](const PingEvent& e) { a += e.value; });
    bus.subscribe([&](const PingEvent& e) { b += e.value; });
    bus.subscribe([&](const PingEvent& e) { c += e.value; });
    bus.publish(PingEvent{1, 5});
    REQUIRE(a == 5);
    REQUIRE(b == 5);
    REQUIRE(c == 5);
    REQUIRE(bus.subscriber_count() == 3);
}

TEST_CASE("priority ordering: High runs before Normal/Low", "[event_bus]") {
    EventBus<PingEvent> bus;
    std::vector<std::string> order;
    // Intentionally subscribe out of priority order to prove the bus reorders.
    bus.subscribe([&](const PingEvent&) { order.push_back("normal"); }, Priority::Normal);
    bus.subscribe([&](const PingEvent&) { order.push_back("low"); }, Priority::Low);
    bus.subscribe([&](const PingEvent&) { order.push_back("high"); }, Priority::High);
    bus.publish(PingEvent{1, 0});
    REQUIRE(order == std::vector<std::string>{"high", "normal", "low"});
}

TEST_CASE("priority ties are stable in subscription order", "[event_bus]") {
    EventBus<PingEvent> bus;
    std::vector<int> order;
    bus.subscribe([&](const PingEvent&) { order.push_back(1); }, Priority::Normal);
    bus.subscribe([&](const PingEvent&) { order.push_back(2); }, Priority::Normal);
    bus.subscribe([&](const PingEvent&) { order.push_back(3); }, Priority::Normal);
    bus.publish(PingEvent{1, 0});
    REQUIRE(order == std::vector<int>{1, 2, 3});
}

TEST_CASE("unsubscribe removes a handler and reports existence", "[event_bus]") {
    EventBus<PingEvent> bus;
    int a = 0, b = 0;
    auto ta = bus.subscribe([&](const PingEvent& e) { a += e.value; });
    bus.subscribe([&](const PingEvent& e) { b += e.value; });
    REQUIRE(bus.subscriber_count() == 2);
    REQUIRE(bus.unsubscribe(ta) == true);
    REQUIRE(bus.subscriber_count() == 1);
    // Unknown / already-removed token returns false.
    REQUIRE(bus.unsubscribe(ta) == false);
    REQUIRE(bus.unsubscribe(999999) == false);
    bus.publish(PingEvent{1, 7});
    REQUIRE(a == 0);  // removed handler never fired
    REQUIRE(b == 7);
}

TEST_CASE("enqueue + dispatch_pending drains in priority order", "[event_bus]") {
    EventBus<PingEvent> bus;
    std::vector<int> seen;
    bus.subscribe([&](const PingEvent& e) { seen.push_back(e.id); });

    // Staged out of order; dispatch must deliver High-priority events first,
    // then Normal, then Low — stable by enqueue order within each level.
    bus.enqueue(PingEvent{1, 0}, Priority::Low);
    bus.enqueue(PingEvent{2, 0}, Priority::High);
    bus.enqueue(PingEvent{3, 0}, Priority::Normal);
    bus.enqueue(PingEvent{4, 0}, Priority::High);

    REQUIRE(bus.pending_count() == 4);
    const std::size_t n = bus.dispatch_pending();
    REQUIRE(n == 4);
    REQUIRE(bus.pending_count() == 0);
    // High (2, then 4 in enqueue order), Normal (3), Low (1).
    REQUIRE(seen == std::vector<int>{2, 4, 3, 1});
    // Draining an empty queue is a no-op returning 0.
    REQUIRE(bus.dispatch_pending() == 0);
}

TEST_CASE("subscriber_count reflects subscribe/unsubscribe and clear", "[event_bus]") {
    EventBus<PingEvent> bus;
    REQUIRE(bus.subscriber_count() == 0);
    auto t1 = bus.subscribe([](const PingEvent&) {});
    auto t2 = bus.subscribe([](const PingEvent&) {});
    REQUIRE(bus.subscriber_count() == 2);
    bus.unsubscribe(t1);
    REQUIRE(bus.subscriber_count() == 1);
    bus.subscribe([](const PingEvent&) {});
    REQUIRE(bus.subscriber_count() == 2);
    (void)t2;
    bus.clear();
    REQUIRE(bus.subscriber_count() == 0);
    REQUIRE(bus.pending_count() == 0);
}

TEST_CASE("concurrency smoke: many threads publish while a subscriber counts", "[event_bus]") {
    EventBus<PingEvent> bus;
    std::atomic<long> count{0};
    bus.subscribe([&](const PingEvent& e) { count.fetch_add(e.value, std::memory_order_relaxed); });
    constexpr int kThreads = 4;
    constexpr int kEventsPerThread = 10000;

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < kEventsPerThread; ++i) {
                bus.publish(PingEvent{i, 1});
            }
        });
    }
    for (auto& th : threads)
        th.join();
    REQUIRE(count.load(std::memory_order_relaxed) ==
            static_cast<long>(kThreads) * kEventsPerThread);
}
