// Local_LLM_Instrumentation — session recorder implementation (C4).

#include "recorder.hpp"

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

namespace ts {

namespace {

// JSON-escape a bounded char buffer (node_name / op_name are char[64]).
std::string esc(const char (&buf)[kNameLen]) {
    std::string out;
    out.reserve(kNameLen + 2);
    out.push_back('"');
    for (size_t i = 0; i < kNameLen && buf[i] != '\0'; ++i) {
        char c = buf[i];
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char u[8];
                    std::snprintf(u, sizeof(u), "\\u%04x",
                                  static_cast<unsigned>(c) & 0xff);
                    out += u;
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
    return out;
}

// Emit a float with enough precision to round-trip exactly (max_digits10).
std::string fnum(float v) {
    std::ostringstream s;
    s.precision(std::numeric_limits<float>::max_digits10);
    s << v;
    return s.str();
}

} // namespace

Recorder::Recorder(const std::string & path)
    : out_(path, std::ios::out | std::ios::trunc | std::ios::binary) {}

Recorder::~Recorder() { close(); }

void Recorder::close() {
    if (out_.is_open()) {
        out_.flush();
        out_.close();
    }
}

void Recorder::write(const ts::TensorEvent & e) {
    if (!ok()) return;

    std::string line;
    line.reserve(512);
    line += '{';

    line += "\"id\":";          line += std::to_string(e.id);
    line += ",\"timestamp_ns\":"; line += std::to_string(e.timestamp_ns);
    line += ",\"layer_idx\":";  line += std::to_string(e.layer_idx);
    line += ",\"op_class\":";   line += std::to_string(static_cast<int>(e.op_class));
    line += ",\"device\":";     line += std::to_string(static_cast<int>(e.device));
    line += ",\"node_name\":";  line += esc(e.node_name);
    line += ",\"op_name\":";    line += esc(e.op_name);

    line += ",\"shape\":[";
    line += std::to_string(e.shape[0]); line += ',';
    line += std::to_string(e.shape[1]); line += ',';
    line += std::to_string(e.shape[2]); line += ',';
    line += std::to_string(e.shape[3]); line += ']';

    line += ",\"dtype\":";      line += std::to_string(e.dtype);
    line += ",\"dtype_size\":"; line += std::to_string(e.dtype_size);
    line += ",\"latency_us\":"; line += std::to_string(e.latency_us);

    line += ",\"v_min\":";      line += fnum(e.v_min);
    line += ",\"v_max\":";      line += fnum(e.v_max);
    line += ",\"v_mean\":";     line += fnum(e.v_mean);
    line += ",\"sparsity\":";   line += fnum(e.sparsity);

    line += ",\"has_nan\":";    line += (e.has_nan ? "true" : "false");
    line += ",\"has_inf\":";    line += (e.has_inf ? "true" : "false");
    line += ",\"stats_valid\":"; line += (e.stats_valid ? "true" : "false");

    line += ",\"payload_id\":"; line += std::to_string(e.payload_id);

    line += "}\n";

    out_ << line;
}

} // namespace ts
