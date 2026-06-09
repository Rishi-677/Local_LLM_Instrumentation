// Local_LLM_Instrumentation — session recorder (C4).
//
// Serializes the live TensorEvent stream to a newline-delimited JSON file
// (one flat object per line, NDJSON). This is the on-disk format the Replay
// reader consumes for offline analysis / regression diffing. We hand-roll the
// JSON (no external dependency): every TensorEvent is flat with only string,
// number, bool, and a single fixed-length shape array, so a tiny writer/parser
// pair round-trips it exactly.
//
// Depends only on event.hpp + the standard library.

#pragma once

#include <fstream>
#include <string>

#include "../event.hpp"

namespace ts {

class Recorder {
public:
    explicit Recorder(const std::string & path);
    ~Recorder();

    bool ok() const { return static_cast<bool>(out_) && out_.is_open(); }

    // Append one event as a single NDJSON line. No-op if the stream is bad.
    void write(const ts::TensorEvent & e);

    // Flush and close. Safe to call multiple times; the destructor also calls it.
    void close();

private:
    std::ofstream out_;
};

} // namespace ts
