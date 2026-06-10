cmake_minimum_required(VERSION 3.20)
project(local_llm_instrumentation LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
option(LOCAL_LLM_INSTRUMENTATION_BUILD_TESTS "Build the Catch2 unit tests" OFF)
# Sanitizer: address | undefined | thread (empty = none). Driven by the
# asan/ubsan/tsan presets in CMakePresets.json.
set(LOCAL_LLM_INSTRUMENTATION_SANITIZE "" CACHE STRING "Sanitizer to enable (address|undefined|thread)")

# Apply the selected sanitizer's compile + link flags to a target.
function(local_llm_instrumentation_apply_sanitizer tgt)
  if(LOCAL_LLM_INSTRUMENTATION_SANITIZE)
    target_compile_options(${tgt} PRIVATE
      -fsanitize=${LOCAL_LLM_INSTRUMENTATION_SANITIZE} -fno-omit-frame-pointer -g)
    target_link_options(${tgt} PRIVATE -fsanitize=${LOCAL_LLM_INSTRUMENTATION_SANITIZE})
  endif()
endfunction()

include(FetchContent)

# ----------------------------------------------------------------------------
# llama.cpp (provides ggml + the cb_eval non-invasive tensor hook)
# ----------------------------------------------------------------------------
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  llama
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
  GIT_TAG        b9587
  GIT_SHALLOW    TRUE
)

# ----------------------------------------------------------------------------
# FTXUI (terminal UI)
# ----------------------------------------------------------------------------
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS    OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
  GIT_TAG        v6.1.9
  GIT_SHALLOW    TRUE
)

# ----------------------------------------------------------------------------
# toml++ (config) and spdlog (logging; bundles fmtlib)
# ----------------------------------------------------------------------------
FetchContent_Declare(
  tomlplusplus
  GIT_REPOSITORY https://github.com/marzer/tomlplusplus
  GIT_TAG        v3.4.0
  GIT_SHALLOW    TRUE
)
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog
  GIT_TAG        v1.15.1
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(llama ftxui tomlplusplus spdlog)

# ----------------------------------------------------------------------------
# Local_LLM_Instrumentation
# ----------------------------------------------------------------------------
add_executable(local_llm_instrumentation
  src/main.cpp
  src/capture/capturer.cpp
  src/capture/topology.cpp
  src/anomaly.cpp
  src/config.cpp
  src/session/recorder.cpp
  src/session/replay.cpp
  src/tui/app.cpp
)

target_include_directories(local_llm_instrumentation PRIVATE src)

target_link_libraries(local_llm_instrumentation PRIVATE
  llama
  ggml
  ftxui::screen
  ftxui::dom
  ftxui::component
  tomlplusplus::tomlplusplus
  spdlog::spdlog
)

# ggml's CPU backend is compiled with OpenMP (-fopenmp); on MinGW the libgomp
# dependency is not carried transitively through the static archives, and CMake's
# FindOpenMP reports an empty lib name (it assumes the flag is enough). Passing
# -fopenmp as an explicit LINK option makes g++ pull libgomp onto the link line.
target_link_options(local_llm_instrumentation PRIVATE -fopenmp)

# On MinGW/Windows, -static links the runtime (libgcc, libstdc++, libgomp,
# winpthread) into the exe so it doesn't depend on the WinLibs DLLs at runtime.
# Not used on Linux (fully-static glibc links are fragile and unnecessary there).
if(WIN32)
  target_link_options(local_llm_instrumentation PRIVATE -static)
endif()

local_llm_instrumentation_apply_sanitizer(local_llm_instrumentation)

# ----------------------------------------------------------------------------
# Unit tests (Catch2)
# ----------------------------------------------------------------------------
if(LOCAL_LLM_INSTRUMENTATION_BUILD_TESTS)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2
    GIT_TAG        v3.7.1
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

  enable_testing()
  add_executable(unit_tests
    tests/test_ring_buffer.cpp
    tests/test_event_bus.cpp
    tests/test_stats.cpp
    tests/test_config.cpp
    src/config.cpp
  )
  target_include_directories(unit_tests PRIVATE src)
  target_link_libraries(unit_tests PRIVATE
    Catch2::Catch2WithMain
    tomlplusplus::tomlplusplus
  )
  # Self-contained on MinGW so catch_discover_tests can run it at build time
  # (otherwise STATUS_ENTRYPOINT_NOT_FOUND from a mismatched runtime DLL).
  if(WIN32)
    target_link_options(unit_tests PRIVATE -static)
  endif()
  local_llm_instrumentation_apply_sanitizer(unit_tests)

  include(Catch)
  catch_discover_tests(unit_tests)
endif()
