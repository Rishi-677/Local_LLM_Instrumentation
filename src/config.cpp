// Local_LLM_Instrumentation — TOML config loader implementation.

#include "config.hpp"

#include <sstream>

#include <toml++/toml.hpp>

namespace ts {

Config Config::defaults() {
    return Config{};
}

namespace {

// Pull a typed scalar out of a table into `dst` if (and only if) the key exists
// and has the expected type. Anything else (missing key, wrong type) leaves the
// default untouched — tolerant by design.
template <typename T>
void assign(const toml::table& tbl, std::string_view key, T& dst) {
    if (auto node = tbl[key].value<T>()) {
        dst = *node;
    }
}

// size_t needs widening through int64_t (toml++ stores integers as int64_t).
void assign_size(const toml::table& tbl, std::string_view key, size_t& dst) {
    if (auto node = tbl[key].value<int64_t>()) {
        if (*node >= 0) {
            dst = static_cast<size_t>(*node);
        }
    }
}

// float comes through as double in toml++.
void assign_float(const toml::table& tbl, std::string_view key, float& dst) {
    if (auto node = tbl[key].value<double>()) {
        dst = static_cast<float>(*node);
    }
}

} // namespace

Config Config::load(const std::string& path, std::string* error) {
    Config cfg = defaults();

    toml::table root;
    try {
        root = toml::parse_file(path);
    } catch (const toml::parse_error& err) {
        if (error) {
            std::ostringstream os;
            os << "config parse error in " << path << ": " << err.description();
            const auto& src = err.source();
            os << " (line " << src.begin.line << ", column " << src.begin.column << ")";
            *error = os.str();
        }
        return cfg;
    } catch (const std::exception& err) {
        if (error) {
            *error = "config load error in " + path + ": " + err.what();
        }
        return cfg;
    }

    if (auto* buffer = root["buffer"].as_table()) {
        assign_size(*buffer, "ring_capacity", cfg.ring_capacity);
    }

    if (auto* session = root["session"].as_table()) {
        assign<int>(*session, "max_tokens", cfg.max_tokens);
        assign<int>(*session, "delay_ms",   cfg.delay_ms);
    }

    if (auto* capture = root["capture"].as_table()) {
        assign<bool>(*capture, "embed",         cfg.capture_embed);
        assign<bool>(*capture, "attn",          cfg.capture_attn);
        assign<bool>(*capture, "mlp",           cfg.capture_mlp);
        assign<bool>(*capture, "norm",          cfg.capture_norm);
        assign<bool>(*capture, "other",         cfg.capture_other);
        assign<bool>(*capture, "no_flash_attn", cfg.no_flash_attn);
    }

    if (auto* anomaly = root["anomaly"].as_table()) {
        assign_float(*anomaly, "threshold",         cfg.anomaly_threshold);
        assign<bool>(*anomaly, "flag_cpu_fallback", cfg.flag_cpu_fallback);
    }

    if (auto* attention = root["attention"].as_table()) {
        assign<int>(*attention, "layer", cfg.attention_layer);
        assign<int>(*attention, "head",  cfg.attention_head);
    }

    if (auto* theme = root["theme"].as_table()) {
        assign<std::string>(*theme, "name", cfg.theme);
        assign<bool>(*theme, "nerd_fonts",  cfg.nerd_fonts);
    }

    if (error) {
        error->clear();
    }
    return cfg;
}

std::string Config::to_string() const {
    std::ostringstream os;
    os << "Config {\n"
       << "  buffer.ring_capacity = " << ring_capacity << "\n"
       << "  session.max_tokens   = " << max_tokens << "\n"
       << "  session.delay_ms     = " << delay_ms << "\n"
       << "  capture: embed=" << (capture_embed ? "on" : "off")
       << " attn="  << (capture_attn  ? "on" : "off")
       << " mlp="   << (capture_mlp   ? "on" : "off")
       << " norm="  << (capture_norm  ? "on" : "off")
       << " other=" << (capture_other ? "on" : "off")
       << " no_flash_attn=" << (no_flash_attn ? "on" : "off") << "\n"
       << "  anomaly.threshold    = " << anomaly_threshold
       << " flag_cpu_fallback=" << (flag_cpu_fallback ? "on" : "off") << "\n"
       << "  attention.layer/head = " << attention_layer << "/" << attention_head << "\n"
       << "  theme.name           = " << theme
       << " nerd_fonts=" << (nerd_fonts ? "on" : "off") << "\n"
       << "}";
    return os.str();
}

} // namespace ts
