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

#include <fstream>
#include <string>
#include <vector>

#include "../event.hpp"

namespace ts {

class Replay {
public:
    explicit Replay(const std::string & path);

    bool ok() const { return static_cast<bool>(in_) && in_.is_open(); }

    // Parse the next non-empty line into `out`. Returns false at EOF (or on a
    // line that contains no recognizable object).
    bool next(ts::TensorEvent & out);

    // Convenience: drain the whole file from the current position.
    std::vector<ts::TensorEvent> load_all();

private:
    std::ifstream in_;
};

} // namespace ts
