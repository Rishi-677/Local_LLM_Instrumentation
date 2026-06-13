// Local_LLM_Instrumentation — session replay (C4).
//
// Reads back the NDJSON written by Recorder, reconstructing one TensorEvent per
// line. The parser is deliberately *not* a general JSON parser: it understands
// exactly the flat object shape Recorder emits (string / number / bool values,
// plus one fixed 4-element shape array). It is tolerant of field order and
// whitespace, so a hand-edited or differently-ordered file still loads.
//
// Depends only on event.hpp + the standard library.

#pragma once

#include <atomic>
#include <fstream>
#include <string>
#include <vector>

#include "../event.hpp"

namespace ts {

// Shared scrub control for replay mode. Written by the UI thread,
// read by the replay producer thread.
struct ReplayControl {
    std::atomic<bool> paused{false};
    std::atomic<int>  step_request{0}; // >0: advance N events then re-pause
};

class Replay {
public:
    explicit Replay(const std::string & path);

    bool ok() const { return static_cast<bool>(in_) && in_.is_open(); }

    bool next(ts::TensorEvent & out);
    std::vector<ts::TensorEvent> load_all();

private:
    bool next_ndjson(ts::TensorEvent & out);
    bool next_llmtrace(ts::TensorEvent & out);

    std::ifstream in_;
    RecordFormat  format_ = RecordFormat::NDJSON;
    uint64_t      bin_pos_ = 0;
};

} // namespace ts
