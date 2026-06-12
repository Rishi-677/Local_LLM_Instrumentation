// Local_LLM_Instrumentation — runtime configuration (TOML-backed).
//
// `Config` is the single bag of runtime-tunable knobs, populated from a TOML
// file (see local_llm_instrumentation.toml at the repo root) and/or CLI overrides. Every
// field has a sane default, so a missing file or a partial file is never an
// error: load() returns a fully-formed Config either way.
//
// Layout mirrors the TOML tables one-to-one:
//   [buffer]    -> ring_capacity
//   [session]   -> max_tokens, delay_ms
//   [capture]   -> capture_embed/attn/mlp/norm/other, no_flash_attn
//   [anomaly]   -> anomaly_threshold, flag_cpu_fallback
//   [attention] -> attention_layer, attention_head
//   [theme]     -> theme, nerd_fonts
//
// Depends only on toml++ (in the .cpp) + the standard library.

#pragma once

#include <cstddef>
#include <string>

namespace ts {

struct Config {
    // [buffer] — ring buffer sizing.
    size_t ring_capacity = 32768;   // events held in the lock-free ring

    // [session] — generation / pacing.
    int max_tokens = 64;            // tokens to generate before stopping
    int delay_ms   = 0;             // artificial per-token delay (demo / debug)

    // [capture] — op-class capture mask: which classes get recorded.
    bool capture_embed = true;
    bool capture_attn  = true;
    bool capture_mlp   = true;
    bool capture_norm  = true;
    bool capture_other = true;
    bool no_flash_attn = true;      // disable flash-attn so attn scores are visible
    int  stats_sample  = 8192;      // max elements scanned per tensor for stats

    // [anomaly] — anomaly-detection tuning.
    float anomaly_threshold = 1e4f; // |value| above this is flagged as an outlier
    bool  flag_cpu_fallback = false;

    // [attention] — default attention payload target.
    int attention_layer = 0;
    int attention_head  = 0;

    // [sidecar] — PyTorch sidecar receiver.
    bool        sidecar_enabled = false;
    int         sidecar_port    = 9876;
    std::string sidecar_host    = "127.0.0.1";

    // [theme] — TUI appearance.
    std::string theme      = "gruvbox";
    bool        nerd_fonts = true;

    // A Config with every field at its documented default.
    static Config defaults();

    // Parse the TOML file at `path`. On a parse/IO error, fill *error (if
    // non-null) with a human-readable message and return defaults(). Missing
    // keys keep their default; unknown keys are ignored. Never throws.
    static Config load(const std::string& path, std::string* error = nullptr);

    // Short, human-readable one-block dump for logging / --print-config.
    std::string to_string() const;
};

} // namespace ts
