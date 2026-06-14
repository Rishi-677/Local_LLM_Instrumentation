"""End-to-end test: C++ sidecar receiver <-> Python PyTorch sidecar.

  python tests/test_sidecar_e2e.py

Spins up the C++ binary in --sidecar --headless mode, runs the
PyTorch sidecar with a tiny model, and verifies events flow.
"""

import json
import os
import subprocess
import sys
import time

BINARY = os.path.join(os.path.dirname(__file__), "..", "build",
                      "local_llm_instrumentation.exe")
SIDECAR = os.path.join(os.path.dirname(__file__), "..", "py_sidecar",
                       "sidecar.py")
MODEL = "hf-internal-testing/tiny-random-gpt2"
PORT = 19876  # non-default to avoid port conflicts


def main() -> int:
    # ---- start C++ receiver -------------------------------------------------
    print(f"[test] starting C++ receiver on port {PORT} ...")
    cpp = subprocess.Popen(
        [BINARY, "--sidecar", f"--sidecar-port", str(PORT),
         "--sidecar-host", "127.0.0.1", "--headless", "--max-tokens=4"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(1.5)  # give the receiver time to bind/listen

    if cpp.poll() is not None:
        print(f"[FAIL] C++ binary exited prematurely (rc={cpp.returncode})")
        _, err = cpp.communicate()
        print("stderr:", err)
        return 1

    # ---- run Python sidecar -------------------------------------------------
    print(f"[test] starting Python sidecar (model={MODEL}) ...")
    py = subprocess.Popen(
        [sys.executable, SIDECAR,
         f"--model={MODEL}",
         f"--port={PORT}",
         f"--host=127.0.0.1",
         "--max-tokens=4",
         "--device=cpu",
         "--output-attentions"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for both to finish (C++ exits when Python disconnects).
    try:
        py_out, py_err = py.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        py.kill()
        py_out, py_err = py.communicate()
        print("[WARN] Python sidecar timed out")

    # Give C++ a moment to process the disconnect and exit.
    try:
        cpp_out, cpp_err = cpp.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        cpp.kill()
        cpp_out, cpp_err = cpp.communicate()
        print("[WARN] C++ receiver timed out (killed)")

    print("--- Python stdout ---")
    print(py_out)
    if py_err:
        print("--- Python stderr ---")
        print(py_err)

    print("--- C++ stdout ---")
    print(cpp_out)
    if cpp_err:
        print("--- C++ stderr ---")
        print(cpp_err)

    # ---- verify results -----------------------------------------------------
    py_rc = py.returncode
    cpp_rc = cpp.returncode
    print(f"\n[test] Python rc={py_rc}  C++ rc={cpp_rc}")

    if py_rc != 0:
        print("[FAIL] Python sidecar exited with error")
        return 1

    if cpp_rc != 0:
        print("[FAIL] C++ receiver exited with error")
        return 1

    # Check that the C++ output shows events were received.
    combined = cpp_out + cpp_err
    if "events received" not in combined:
        print("[FAIL] C++ output missing 'events received' — no data flowed")
        return 1

    # Check attention was captured.
    if "attention" in combined.lower():
        print("[INFO] Attention events were captured")

    print("[PASS] End-to-end sidecar test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
