// Local_LLM_Instrumentation — session replay implementation (C4).
//
// Minimal line-oriented parser for the flat NDJSON that Recorder emits. We scan
// the line for "key": tokens and read the value that follows, dispatching by
// the known key set. Values are: quoted strings (with escape handling), numbers,
// booleans, and a single bracketed shape array. Field order is not assumed.

#include "replay.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace ts {

namespace {

// --- low-level scanners over a std::string + cursor -----------------------

void skip_ws(const std::string & s, size_t & i) {
    while (i < s.size() &&
           (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) {
        ++i;
    }
}

// At a '"', parse a JSON string (handling escapes) into `out`, advancing past
// the closing quote. Returns false if not positioned on a string.
bool parse_string(const std::string & s, size_t & i, std::string & out) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '"') return false;
    ++i; // opening quote
    out.clear();
    while (i < s.size()) {
        char c = s[i++];
        if (c == '"') return true;
        if (c == '\\' && i < s.size()) {
            char e = s[i++];
            switch (e) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    if (i + 4 <= s.size()) {
                        unsigned cp = static_cast<unsigned>(
                            std::strtoul(s.substr(i, 4).c_str(), nullptr, 16));
                        i += 4;
                        // Recorder only ever emits \u00XX for control chars.
                        out.push_back(static_cast<char>(cp & 0xff));
                    }
                    break;
                }
                default: out.push_back(e); break;
            }
        } else {
            out.push_back(c);
        }
    }
    return false; // unterminated
}

// Read a bare token (number / bool) up to the next delimiter.
std::string read_token(const std::string & s, size_t & i) {
    skip_ws(s, i);
    size_t start = i;
    while (i < s.size()) {
        char c = s[i];
        if (c == ',' || c == '}' || c == ']' || c == ' ' || c == '\t' ||
            c == '\r' || c == '\n') {
            break;
        }
        ++i;
    }
    return s.substr(start, i - start);
}

void copy_into(char (&dst)[kNameLen], const std::string & src) {
    std::memset(dst, 0, kNameLen);
    size_t n = src.size() < kNameLen ? src.size() : (kNameLen - 1);
    std::memcpy(dst, src.data(), n);
}

} // namespace

Replay::Replay(const std::string & path)
    : in_(path, std::ios::in | std::ios::binary) {}

std::vector<ts::TensorEvent> Replay::load_all() {
    std::vector<ts::TensorEvent> v;
    ts::TensorEvent e;
    while (next(e)) v.push_back(e);
    return v;
}

bool Replay::next(ts::TensorEvent & out) {
    std::string line;
    // Skip blank lines / lines with no object brace.
    while (std::getline(in_, line)) {
        if (line.find('{') != std::string::npos) break;
    }
    if (line.find('{') == std::string::npos) return false;

    ts::TensorEvent e{}; // value-initialized defaults match the schema

    size_t i = line.find('{') + 1;
    while (i < line.size()) {
        skip_ws(line, i);
        if (i >= line.size() || line[i] == '}') break;

        std::string key;
        if (!parse_string(line, i, key)) break; // malformed: stop gracefully

        skip_ws(line, i);
        if (i < line.size() && line[i] == ':') ++i;
        skip_ws(line, i);

        if (key == "node_name") {
            std::string v;
            parse_string(line, i, v);
            copy_into(e.node_name, v);
        } else if (key == "op_name") {
            std::string v;
            parse_string(line, i, v);
            copy_into(e.op_name, v);
        } else if (key == "shape") {
            if (i < line.size() && line[i] == '[') ++i;
            for (int d = 0; d < kMaxDims; ++d) {
                std::string t = read_token(line, i);
                if (!t.empty()) {
                    e.shape[d] = static_cast<int64_t>(
                        std::strtoll(t.c_str(), nullptr, 10));
                }
                skip_ws(line, i);
                if (i < line.size() && line[i] == ',') ++i;
            }
            skip_ws(line, i);
            if (i < line.size() && line[i] == ']') ++i;
        } else {
            std::string t = read_token(line, i);
            if (key == "id") {
                e.id = static_cast<uint64_t>(
                    std::strtoull(t.c_str(), nullptr, 10));
            } else if (key == "timestamp_ns") {
                e.timestamp_ns = static_cast<uint64_t>(
                    std::strtoull(t.c_str(), nullptr, 10));
            } else if (key == "layer_idx") {
                e.layer_idx = static_cast<int32_t>(
                    std::strtol(t.c_str(), nullptr, 10));
            } else if (key == "op_class") {
                e.op_class = static_cast<OpClass>(
                    std::strtol(t.c_str(), nullptr, 10));
            } else if (key == "device") {
                e.device = static_cast<Device>(
                    std::strtol(t.c_str(), nullptr, 10));
            } else if (key == "dtype") {
                e.dtype = static_cast<int32_t>(
                    std::strtol(t.c_str(), nullptr, 10));
            } else if (key == "dtype_size") {
                e.dtype_size = static_cast<int32_t>(
                    std::strtol(t.c_str(), nullptr, 10));
            } else if (key == "latency_us") {
                e.latency_us = static_cast<uint64_t>(
                    std::strtoull(t.c_str(), nullptr, 10));
            } else if (key == "v_min") {
                e.v_min = std::strtof(t.c_str(), nullptr);
            } else if (key == "v_max") {
                e.v_max = std::strtof(t.c_str(), nullptr);
            } else if (key == "v_mean") {
                e.v_mean = std::strtof(t.c_str(), nullptr);
            } else if (key == "sparsity") {
                e.sparsity = std::strtof(t.c_str(), nullptr);
            } else if (key == "has_nan") {
                e.has_nan = (t == "true" || t == "1");
            } else if (key == "has_inf") {
                e.has_inf = (t == "true" || t == "1");
            } else if (key == "stats_valid") {
                e.stats_valid = (t == "true" || t == "1");
            } else if (key == "payload_id") {
                e.payload_id = static_cast<uint32_t>(
                    std::strtoul(t.c_str(), nullptr, 10));
            }
            // Unknown keys: token already consumed, ignored.
        }

        skip_ws(line, i);
        if (i < line.size() && line[i] == ',') ++i;
    }

    out = e;
    return true;
}

} // namespace ts
