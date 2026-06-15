// Tests for the pure activation-statistics function.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>
#include <vector>
#include "stats.hpp"
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("empty / null input yields invalid stats", "[stats]") {
    ts::ActivationStats s = ts::compute_stats(nullptr, 0);
    REQUIRE_FALSE(s.stats_valid);
    std::vector<float> v;
    REQUIRE_FALSE(ts::compute_stats(v.data(), 0).stats_valid);
}

TEST_CASE("basic reductions over a known vector", "[stats]") {
    std::vector<float> v{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    ts::ActivationStats s = ts::compute_stats(v.data(), (int64_t)v.size());
    REQUIRE(s.stats_valid);
    REQUIRE_THAT(s.v_min, WithinAbs(1.0f, 1e-5));
    REQUIRE_THAT(s.v_max, WithinAbs(5.0f, 1e-5));
    REQUIRE_THAT(s.v_mean, WithinAbs(3.0f, 1e-5));
    REQUIRE_THAT(s.v_std, WithinAbs(std::sqrt(2.0f), 1e-4));
    REQUIRE_THAT(s.l2_norm, WithinAbs(std::sqrt(55.0f), 1e-3));
    REQUIRE_THAT(s.sparsity, WithinAbs(0.0f, 1e-6));
    REQUIRE_FALSE(s.has_nan);
    REQUIRE_FALSE(s.has_inf);
}

TEST_CASE("sparsity counts near-zero elements", "[stats]") {
    std::vector<float> v{0.0f, 0.0f, 0.0f, 1.0f};  // 75% zeros
    ts::ActivationStats s = ts::compute_stats(v.data(), (int64_t)v.size());
    REQUIRE_THAT(s.sparsity, WithinAbs(0.75f, 1e-6));
}

TEST_CASE("NaN and Inf are flagged and excluded from reductions", "[stats]") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<float> v{2.0f, nan, 4.0f, inf};
    ts::ActivationStats s = ts::compute_stats(v.data(), (int64_t)v.size());
    REQUIRE(s.has_nan);
    REQUIRE(s.has_inf);
    REQUIRE_THAT(s.v_mean, WithinAbs(3.0f, 1e-5));
    REQUIRE_THAT(s.v_max, WithinAbs(4.0f, 1e-5));
}

TEST_CASE("stride sampling stays bounded on large inputs", "[stats]") {
    std::vector<float> v(1 << 20, 2.0f);
    ts::ActivationStats s = ts::compute_stats(v.data(), (int64_t)v.size(), 1024);
    REQUIRE(s.stats_valid);
    REQUIRE_THAT(s.v_mean, WithinAbs(2.0f, 1e-5));
    REQUIRE_THAT(s.v_min, WithinAbs(2.0f, 1e-5));
    REQUIRE_THAT(s.v_max, WithinAbs(2.0f, 1e-5));
}
