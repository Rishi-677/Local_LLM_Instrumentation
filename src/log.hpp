// Local_LLM_Instrumentation — logging (Phase 0).
//
// Diagnostic logging goes to a FILE, never stdout/stderr: the TUI owns the
// terminal, so any stray write would corrupt the dashboard. Backed by spdlog
// (which bundles fmtlib). Call ts::log::init() once at startup; thereafter use
// the spdlog macros (SPDLOG_INFO / SPDLOG_WARN / ...) or spdlog::info(...).
//
// Logging is best-effort: if the sink can't be created we fall back silently so
// the app still runs.

#pragma once

#include <exception>
#include <memory>
#include <string>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

namespace ts::log {

inline void init(const std::string& path = "local_llm_instrumentation.log", bool verbose = false) {
    try {
        auto logger = spdlog::basic_logger_mt("local_llm_instrumentation", path, /*truncate=*/true);
        logger->set_level(verbose ? spdlog::level::debug : spdlog::level::info);
        logger->flush_on(spdlog::level::info);
        logger->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
        spdlog::set_default_logger(std::move(logger));
    } catch (const std::exception&) {
        // Best-effort: keep the default (no-op-ish) logger if the file sink fails.
    }
}

} // namespace ts::log
