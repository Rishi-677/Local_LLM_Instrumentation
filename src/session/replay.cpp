// Local_LLM_Instrumentation — session replay implementation (C4).
//
// Supports two formats:
//   NDJSON     — human-readable newline-delimited JSON (default)
//   LLMTRACE   — compact binary (.llmtrace), raw LayerEvent records
//
// Format is auto-detected from the file extension (.llmtrace → binary).
// The NDJSON parser is a minimal line-oriented scanner that handles exactly
// the flat object shape Recorder emits (string / number / bool values, plus
// one fixed 4-element shape array). It is tolerant of field order and
// whitespace, so a hand-edited or differently-ordered file still loads.

#include "replay.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace ts {

namespace {

constexpr uint32_t kLLMTMagic = 0x544d4c4c; // "LLMT" little-endian

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

RecordFormat detect_format(const std::string & path) {
    if (path.size() >= 9 &&
        path.compare(path.size() - 9, 9, ".llmtrace") == 0)
        return RecordFormat::LLMTRACE;
    return RecordFormat::NDJSON;
}

} // namespace

Replay::Replay(const std::string & path)
    : in_(path, std::ios::in | std::ios::binary),
      format_(detect_format(path)) {
    if (format_ == RecordFormat::LLMTRACE) {
        // Skip the 16-byte header: magic + version + reserved.
        uint32_t header[4];
        in_.read(reinterpret_cast<char*>(header), sizeof(header));
        if (header[0] != kLLMTMagic) {
            // Invalid magic — reset stream and fall back to best-effort.
            in_.clear();
            in_.seekg(0, std::ios::beg);
        }
        bin_pos_ = sizeof(header);
    }
}

std::vector<ts::TensorEvent> Replay::load_all() {
    std::vector<ts::TensorEvent> v;
    ts::TensorEvent e;
    while (next(e)) v.push_back(e);
    return v;
}

bool Replay::next(ts::TensorEvent & out) {
    if (format_ == RecordFormat::LLMTRACE)
        return next_llmtrace(out);
    return next_ndjson(out);
}

bool Replay::next_ndjson(ts::TensorEvent & out) {
    std::string line;
    while (std::getline(in_, line)) {
        if (line.find('{') != std::string::npos) break;
    }
    if (line.find('{') == std::string::npos) return false;

    ts::TensorEvent e{};

    size_t i = line.find('{') + 1;
    while (i < line.size()) {
        skip_ws(line, i);
        if (i >= line.size() || line[i] == '}') break;

        std::string key;
        if (!parse_string(line, i, key)) break;

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
            } else if (key == "v_std") {
                e.v_std = std::strtof(t.c_str(), nullptr);
            } else if (key == "l2_norm") {
                e.l2_norm = std::strtof(t.c_str(), nullptr);
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
        }

        skip_ws(line, i);
        if (i < line.size() && line[i] == ',') ++i;
    }

    out = e;
    return true;
}

bool Replay::next_llmtrace(ts::TensorEvent & out) {
    static_assert(std::is_trivially_copyable<ts::TensorEvent>::value,
                  "TensorEvent must be trivially copyable for binary deserialization");
    ts::TensorEvent e{};
    if (!in_.read(reinterpret_cast<char*>(&e), sizeof(e))) return false;
    bin_pos_ += sizeof(e);
    out = e;
    return true;
}

} // namespace ts
