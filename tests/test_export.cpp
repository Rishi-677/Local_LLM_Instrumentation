// Tests for the per-layer summary aggregator / CSV-JSON export.

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "export.hpp"

namespace {
ts::TensorEvent make_ev(int32_t layer, ts::OpClass cls, uint64_t lat,
                        float vmax, float sparsity) {
    ts::TensorEvent e;
    e.layer_idx = layer;
    e.op_class  = cls;
    e.latency_us = lat;
    e.v_max = vmax;
    e.v_min = 0.0f;
    e.sparsity = sparsity;
    e.stats_valid = true;
    return e;
}

std::string slurp(const std::string& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
}  // namespace

TEST_CASE("summary aggregates by (layer, op-class)", "[export]") {
    ts::Summary s;
    // Two Attn events on layer 0, one MLP on layer 0, one Attn on layer 1.
    s.add(make_ev(0, ts::OpClass::Attn, 100, 5.0f, 0.2f));
    s.add(make_ev(0, ts::OpClass::Attn, 300, 9.0f, 0.4f));
    s.add(make_ev(0, ts::OpClass::MLP, 50, 1.0f, 0.0f));
    s.add(make_ev(1, ts::OpClass::Attn, 20, 2.0f, 0.1f));

    REQUIRE(s.rows() == 3);  // (0,Attn) (0,MLP) (1,Attn)
}

TEST_CASE("CSV export has header, row count, and aggregated values", "[export]") {
    ts::Summary s;
    s.add(make_ev(0, ts::OpClass::Attn, 100, 5.0f, 0.2f));
    s.add(make_ev(0, ts::OpClass::Attn, 300, 9.0f, 0.4f));  // same key

    auto path = (std::filesystem::temp_directory_path() / "ts_export_test.csv").string();
    REQUIRE(s.write_csv(path));
    std::string csv = slurp(path);
    std::filesystem::remove(path);

    REQUIRE(csv.find("layer_idx,op_class,count") != std::string::npos);
    // count=2, total_latency=400, avg=200, max_abs=9
    REQUIRE(csv.find("0,Attn,2,400,200,9") != std::string::npos);
}

TEST_CASE("JSON export is a well-formed array", "[export]") {
    ts::Summary s;
    s.add(make_ev(2, ts::OpClass::Norm, 10, 3.0f, 0.0f));

    auto path = (std::filesystem::temp_directory_path() / "ts_export_test.json").string();
    REQUIRE(s.write_json(path));
    std::string js = slurp(path);
    std::filesystem::remove(path);

    REQUIRE(js.front() == '[');
    REQUIRE(js.find("\"op_class\":\"Norm\"") != std::string::npos);
    REQUIRE(js.find("\"count\":1") != std::string::npos);
}
