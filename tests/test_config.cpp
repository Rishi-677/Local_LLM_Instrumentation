// Local_LLM_Instrumentation — Config (TOML loader) tests.
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include "config.hpp"

namespace {
// A per-binary-run unique seed without pulling in process headers: the address
// of a static is plenty unique within one test binary run.
inline unsigned long run_seed() {
    static int anchor = 0;
    return static_cast<unsigned long>(reinterpret_cast<std::uintptr_t>(&anchor) & 0xffffu);
}

// RAII temp .toml file under the OS temp dir; cleaned up on scope exit.
struct TempToml {
    std::filesystem::path path;

    explicit TempToml(const std::string& contents) {
        static int counter = 0;
        path = std::filesystem::temp_directory_path() /
               ("local_llm_instrumentation_test_" + std::to_string(run_seed()) + "_" +
                std::to_string(counter++) + ".toml");
        std::ofstream(path) << contents;
    }

    ~TempToml() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    std::string str() const {
        return path.string();
    }
};

}  // namespace

TEST_CASE("Config::defaults has the documented defaults", "[config]") {
    const ts::Config c = ts::Config::defaults();
    CHECK(c.ring_capacity == 32768u);
    CHECK(c.max_tokens == 64);
    CHECK(c.delay_ms == 0);
    CHECK(c.capture_embed);
    CHECK(c.capture_attn);
    CHECK(c.capture_mlp);
    CHECK(c.capture_norm);
    CHECK(c.capture_other);
    CHECK(c.no_flash_attn);
    CHECK(c.anomaly_threshold == 1e4f);
    CHECK_FALSE(c.flag_cpu_fallback);
    CHECK(c.attention_layer == 0);
    CHECK(c.attention_head == 0);
    CHECK(c.theme == "gruvbox");
    CHECK(c.nerd_fonts);
}

TEST_CASE("Config::load applies overrides and keeps untouched defaults", "[config]") {
    // Override a handful of keys across several tables; leave the rest absent.
    TempToml f(R"(
[buffer]
ring_capacity = 1024

[session]
max_tokens = 200

[capture]
attn = false
no_flash_attn = false

[anomaly]
threshold = 250.5
flag_cpu_fallback = true

[attention]
layer = 7

[theme]
name = "nord"
)");

    std::string error = "sentinel";
    const ts::Config c = ts::Config::load(f.str(), &error);
    CHECK(error.empty());  // success clears the error string
    // Overridden.
    CHECK(c.ring_capacity == 1024u);
    CHECK(c.max_tokens == 200);
    CHECK_FALSE(c.capture_attn);
    CHECK_FALSE(c.no_flash_attn);
    CHECK(c.anomaly_threshold == 250.5f);
    CHECK(c.flag_cpu_fallback);
    CHECK(c.attention_layer == 7);
    CHECK(c.theme == "nord");
    // Untouched keys keep their defaults.
    CHECK(c.delay_ms == 0);
    CHECK(c.capture_embed);
    CHECK(c.capture_mlp);
    CHECK(c.capture_norm);
    CHECK(c.capture_other);
    CHECK(c.attention_head == 0);
    CHECK(c.nerd_fonts);
}

TEST_CASE("Config::load on malformed TOML sets error and returns defaults", "[config]") {
    // Unterminated string / dangling assignment — a hard parse error.
    TempToml f("[buffer]\nring_capacity = = =\nthis is not valid toml\n");
    std::string error;
    ts::Config c;
    REQUIRE_NOTHROW(c = ts::Config::load(f.str(), &error));
    CHECK_FALSE(error.empty());        // error populated
    CHECK(c.ring_capacity == 32768u);  // fell back to defaults
    CHECK(c.theme == "gruvbox");
    CHECK(c.max_tokens == 64);
}

TEST_CASE("Config::load on a missing file does not throw and returns defaults", "[config]") {
    const std::string missing =
        (std::filesystem::temp_directory_path() / "local_llm_instrumentation_does_not_exist.toml")
            .string();

    std::string error;
    ts::Config c;
    REQUIRE_NOTHROW(c = ts::Config::load(missing, &error));
    CHECK_FALSE(error.empty());
    CHECK(c.ring_capacity == 32768u);
    CHECK(c.theme == "gruvbox");
}
