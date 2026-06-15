// Tests for the latency flamegraph view model.
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <utility>
#include <vector>
#include "tui/latency_flamegraph.hpp"

namespace {

ts::LayerNode node(std::string label, int32_t layer, uint64_t latency) {
    ts::LayerNode n;
    n.label = std::move(label);
    n.layer_idx = layer;
    n.total_latency_us = latency;
    return n;
}

}  // namespace

TEST_CASE("latency flamegraph keeps layer buckets and sorts by latency", "[latency]") {
    std::vector<ts::LayerNode> topology{
        node("layers.0", 0, 120), node("Attn", 0, 999),     node("layers.2", 2, 40),
        node("embed", -1, 20),    node("layers.1", 1, 300),
    };
    const auto rows = ts::build_latency_flamegraph(topology);
    REQUIRE(rows.size() == 4);
    REQUIRE(rows[0].label == "layers.1");
    REQUIRE(rows[1].label == "layers.0");
    REQUIRE(rows[2].label == "layers.2");
    REQUIRE(rows[3].label == "embed");
}

TEST_CASE("latency flamegraph limit trims after sorting", "[latency]") {
    std::vector<ts::LayerNode> topology{
        node("layers.0", 0, 10),
        node("layers.1", 1, 30),
        node("layers.2", 2, 20),
    };

    const auto rows = ts::build_latency_flamegraph(topology, 2);
    REQUIRE(rows.size() == 2);
    REQUIRE(rows[0].label == "layers.1");
    REQUIRE(rows[1].label == "layers.2");
}

TEST_CASE("latency ASCII bars scale against the slowest layer", "[latency]") {
    REQUIRE(ts::latency_ascii_bar(10, 40, 8) == "##......");
    REQUIRE(ts::latency_ascii_bar(40, 40, 8) == "########");
    REQUIRE(ts::latency_ascii_bar(0, 40, 8) == "........");
    REQUIRE(ts::latency_ascii_bar(10, 0, 8) == "........");
}
