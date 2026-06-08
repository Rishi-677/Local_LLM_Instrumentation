cmake_minimum_required(VERSION 3.20)
project(local_llm_instrumentation LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

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

FetchContent_MakeAvailable(llama ftxui)

# ----------------------------------------------------------------------------
# Local_LLM_Instrumentation
# ----------------------------------------------------------------------------
add_executable(local_llm_instrumentation
  src/main.cpp
  src/capture/capturer.cpp
  src/capture/topology.cpp
  src/anomaly.cpp
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
)

# ggml's CPU backend is compiled with OpenMP (-fopenmp); on MinGW the libgomp
# dependency is not carried transitively through the static archives, and CMake's
# FindOpenMP reports an empty lib name (it assumes the flag is enough). Passing
# -fopenmp as an explicit LINK option makes g++ pull libgomp onto the link line.
#
# -static links the MinGW runtime (libgcc, libstdc++, libgomp, winpthread) into
# the executable so it doesn't depend on the WinLibs DLLs at runtime — this both
# avoids STATUS_ENTRYPOINT_NOT_FOUND from a mismatched DLL on PATH and makes the
# binary self-contained for distribution.
target_link_options(local_llm_instrumentation PRIVATE -fopenmp -static)
