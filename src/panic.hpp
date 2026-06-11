// Local_LLM_Instrumentation - panic/terminal restore guard.
//
// FTXUI owns the terminal while the fullscreen dashboard is active. If the
// process dies via a signal or uncaught exception, restore basic terminal state
// before handing control back to the default crash path.

#pragma once

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>

namespace ts::panic {

namespace detail {

inline std::atomic<bool> terminal_claimed{false};
inline std::atomic<bool> installed{false};

inline void restore_terminal_unchecked() noexcept {
    // Reset attributes, show cursor, disable common mouse modes, leave alt screen.
    std::fputs("\x1b[0m\x1b[?25h\x1b[?1000l\x1b[?1002l\x1b[?1003l"
               "\x1b[?1006l\x1b[?1049l",
               stderr);
    std::fflush(stderr);
}

inline void signal_handler(int sig) {
    if (terminal_claimed.load(std::memory_order_relaxed)) {
        restore_terminal_unchecked();
    }
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

[[noreturn]] inline void terminate_handler() noexcept {
    if (terminal_claimed.load(std::memory_order_relaxed)) {
        restore_terminal_unchecked();
    }
    std::fputs("\nLocal_LLM_Instrumentation terminated unexpectedly.\n", stderr);
    std::fflush(stderr);
    std::abort();
}

} // namespace detail

inline void restore_terminal() noexcept {
    if (detail::terminal_claimed.load(std::memory_order_relaxed)) {
        detail::restore_terminal_unchecked();
    }
}

inline void install() {
    bool expected = false;
    if (!detail::installed.compare_exchange_strong(expected, true)) return;

    std::set_terminate(detail::terminate_handler);
    std::signal(SIGABRT, detail::signal_handler);
    std::signal(SIGFPE, detail::signal_handler);
    std::signal(SIGILL, detail::signal_handler);
    std::signal(SIGINT, detail::signal_handler);
    std::signal(SIGSEGV, detail::signal_handler);
    std::signal(SIGTERM, detail::signal_handler);
#ifdef SIGBREAK
    std::signal(SIGBREAK, detail::signal_handler);
#endif
}

class TerminalClaim {
public:
    TerminalClaim() {
        detail::terminal_claimed.store(true, std::memory_order_release);
    }

    ~TerminalClaim() {
        restore_terminal();
        detail::terminal_claimed.store(false, std::memory_order_release);
    }

    TerminalClaim(const TerminalClaim&) = delete;
    TerminalClaim& operator=(const TerminalClaim&) = delete;
};

} // namespace ts::panic
