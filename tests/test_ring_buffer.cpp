// Tests for the lock-free SPSC ring buffer.

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <thread>

#include "ring_buffer.hpp"

using ts::SpscRing;

TEST_CASE("capacity is rounded up to a power of two", "[ring]") {
    SpscRing<int> r(1000);
    REQUIRE(r.capacity() == 1024);
    SpscRing<int> r2(1024);
    REQUIRE(r2.capacity() == 1024);
}

TEST_CASE("push then pop preserves FIFO order", "[ring]") {
    SpscRing<int> r(8);
    for (int i = 0; i < 5; ++i) REQUIRE(r.push(i));
    int out = -1;
    for (int i = 0; i < 5; ++i) {
        REQUIRE(r.pop(out));
        REQUIRE(out == i);
    }
    REQUIRE_FALSE(r.pop(out));  // empty
}

TEST_CASE("overflow drops and counts, never corrupts", "[ring]") {
    SpscRing<int> r(4);  // capacity 4
    int pushed = 0;
    for (int i = 0; i < 100; ++i)
        if (r.push(i)) ++pushed;
    REQUIRE(pushed == 4);
    REQUIRE(r.dropped() == 96);

    // The 4 retained are the first 4 (drop-newest policy).
    int out = -1;
    for (int i = 0; i < 4; ++i) {
        REQUIRE(r.pop(out));
        REQUIRE(out == i);
    }
}

TEST_CASE("drain delivers all available items", "[ring]") {
    SpscRing<int> r(16);
    for (int i = 0; i < 10; ++i) r.push(i);
    int sum = 0, count = 0;
    size_t n = r.drain([&](int v) { sum += v; ++count; });
    REQUIRE(n == 10);
    REQUIRE(count == 10);
    REQUIRE(sum == 45);
    REQUIRE(r.size_approx() == 0);
}

TEST_CASE("single-producer / single-consumer integrity under threads", "[ring]") {
    SpscRing<uint64_t> r(1024);
    constexpr uint64_t N = 200000;

    std::thread producer([&] {
        for (uint64_t i = 0; i < N; ++i) {
            while (!r.push(i)) { /* spin until the consumer drains */ }
        }
    });

    uint64_t expected = 0;
    uint64_t got = 0;
    std::thread consumer([&] {
        uint64_t v;
        while (got < N) {
            if (r.pop(v)) {
                REQUIRE(v == expected);  // strict FIFO, no gaps/dupes
                ++expected;
                ++got;
            }
        }
    });

    producer.join();
    consumer.join();
    // The real guarantee: every item arrived exactly once, in order (checked
    // above). dropped() is NOT asserted here — it counts failed push *attempts*,
    // which the producer's back-pressure spin inflates without losing any data.
    REQUIRE(got == N);
    REQUIRE(expected == N);
}
