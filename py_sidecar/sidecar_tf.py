#!/usr/bin/env python3
"""Local_LLM_Instrumentation — TensorFlow/Keras sidecar.

Hooks into a HuggingFace TF model by wrapping each leaf layer's call
method, computes lightweight tensor stats, and streams NDJSON over a
TCP socket to the C++ dashboard.

Usage:
  python py_sidecar/sidecar_tf.py --model google/gemma-2b --port 9876
  python py_sidecar/sidecar_tf.py --model bert-base-uncased --port 9876 --output-attentions
"""

import argparse
import json
import socket
import sys
import time
import warnings
from typing import Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional TF imports (deferred so help works without TF installed)
# ---------------------------------------------------------------------------
tf: Optional[object] = None

def _import_tf():
    global tf
    import tensorflow._api.v2.compat.v1 as tf_v1
    tf = tf_v1
    tf.disable_eager_execution()


# ---------------------------------------------------------------------------
# Tensor-level helpers
# ---------------------------------------------------------------------------

DTYPE_MAP: dict[str, int] = {
    "float32": 0,
    "float16": 1,
    "bfloat16": 30,
    "int8": 24,
    "int16": 25,
    "int32": 26,
    "int64": 26,
}


def dtype_code(t: "tf.Tensor") -> int:
    return DTYPE_MAP.get(t.dtype.name, 0)


def dtype_size(t: "tf.Tensor") -> int:
    return t.dtype.size


def shape_arr(t: "tf.Tensor") -> list[int]:
    s = list(t.shape)
    while len(s) < 4:
        s.append(0)
    return s[:4]


def compute_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "v_min": 0.0, "v_max": 0.0, "v_mean": 0.0,
            "v_std": 0.0, "l2_norm": 0.0, "sparsity": 0.0,
            "has_nan": False, "has_inf": False, "stats_valid": False,
        }
    ft = values.ravel().astype(np.float64)
    max_scan = 65536
    if ft.size > max_scan:
        stride = ft.size // max_scan
        ft = ft[::stride]
    nan_mask = np.isnan(ft)
    inf_mask = np.isinf(ft)
    has_nan = bool(nan_mask.any())
    has_inf = bool(inf_mask.any())
    valid = ft[~(nan_mask | inf_mask)]
    if valid.size == 0:
        return {
            "v_min": 0.0, "v_max": 0.0, "v_mean": 0.0,
            "v_std": 0.0, "l2_norm": 0.0, "sparsity": 0.0,
            "has_nan": has_nan, "has_inf": has_inf, "stats_valid": False,
        }
    v_min = float(valid.min())
    v_max = float(valid.max())
    v_mean = float(valid.mean())
    v_std = float(valid.std()) if valid.size > 1 else 0.0
    l2_norm = float(np.sqrt((valid ** 2).sum()))
    sparsity = float((np.abs(valid) < 1e-6).mean())
    return {
        "v_min": v_min, "v_max": v_max, "v_mean": v_mean,
        "v_std": v_std, "l2_norm": l2_norm, "sparsity": sparsity,
        "has_nan": has_nan, "has_inf": has_inf, "stats_valid": True,
    }


# ---------------------------------------------------------------------------
# Layer classification (mirrors src/capture/topology.cpp classify())
# ---------------------------------------------------------------------------

def classify_name(name: str) -> tuple[int, int]:
    name_lower = name.lower()
    cls = 5  # Other
    if any(k in name_lower for k in ("embed", "embd", "wpe", "wte")):
        cls = 0
    elif any(k in name_lower for k in ("attn", "kqv", "self_attn",
                                        "q_proj", "k_proj", "v_proj", "o_proj",
                                        "attention")):
        cls = 1
    elif any(k in name_lower for k in ("mlp", "ffn", "gate_proj", "up_proj",
                                        "down_proj", "fc", "dense")):
        cls = 2
    elif "norm" in name_lower or "layernorm" in name_lower:
        cls = 3
    elif any(k in name_lower for k in ("logits", "lm_head", "embed_out",
                                        "output", "classifier")):
        cls = 4
    layer = -1
    for sep in ('.', '-'):
        if sep in name:
            tail = name.rsplit(sep, 1)[1]
            try:
                layer = int(tail)
            except ValueError:
                pass
    return cls, layer


# ---------------------------------------------------------------------------
# Forward call wrapper
# ---------------------------------------------------------------------------

_counter: int = 0


def make_wrapper(layer_name: str, layer: "tf.keras.layers.Layer",
                 original_call: Callable, sock: socket.socket, send_lock):
    """Wrap *layer.call* to emit telemetry after each forward."""
    global _counter

    def wrapped_call(inputs, *args, **kwargs):
        global _counter
        output = original_call(inputs, *args, **kwargs)
        _counter += 1
        ev_id = _counter
        ts_ns = int(time.time_ns())
        cls_id, layer_idx = classify_name(layer_name)

        dev = 0  # CPU
        try:
            dev_str = layer.device if hasattr(layer, "device") and layer.device else "CPU"
            if "gpu" in dev_str.lower() or "cuda" in dev_str.lower():
                dev = 1
            elif "tpu" in dev_str.lower():
                dev = 2
        except Exception:
            pass

        out_tensor = None
        if isinstance(output, tf.Tensor):
            out_tensor = output
        elif isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, tf.Tensor) and hasattr(item, "shape") and item.shape.num_elements() > 0:
                    out_tensor = item
                    break

        stats = {}
        shape = [0, 0, 0, 0]
        dtype_c = 0
        dtype_sz = 0
        if out_tensor is not None:
            try:
                arr = out_tensor.numpy()
                stats = compute_stats(arr)
                shape = shape_arr(out_tensor)
                dtype_c = dtype_code(out_tensor)
                dtype_sz = dtype_size(out_tensor)
            except Exception:
                stats = {
                    "v_min": 0.0, "v_max": 0.0, "v_mean": 0.0,
                    "v_std": 0.0, "l2_norm": 0.0, "sparsity": 0.0,
                    "has_nan": False, "has_inf": False, "stats_valid": False,
                }
        else:
            stats = {
                "v_min": 0.0, "v_max": 0.0, "v_mean": 0.0,
                "v_std": 0.0, "l2_norm": 0.0, "sparsity": 0.0,
                "has_nan": False, "has_inf": False, "stats_valid": False,
            }

        ev = {
            "id": ev_id,
            "timestamp_ns": ts_ns,
            "layer_idx": layer_idx,
            "op_class": cls_id,
            "device": dev,
            "node_name": layer_name,
            "op_name": layer.__class__.__name__,
            "shape": shape,
            "dtype": dtype_c,
            "dtype_size": dtype_sz,
            "latency_us": 0,
            **stats,
            "payload_id": 0,
        }

        line = json.dumps(ev, separators=(",", ":")) + "\n"
        with send_lock:
            try:
                sock.sendall(line.encode("utf-8"))
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        return output

    return wrapped_call


# ---------------------------------------------------------------------------
# Model instrumentation
# ---------------------------------------------------------------------------

def instrument_model(model: "tf.keras.Model", sock: socket.socket, send_lock,
                     skip_patterns: tuple = ("dropout", "activation",
                                              "activity_regularizer")) -> int:
    count = 0
    for layer in model.submodules:
        name = layer.name
        if not name:
            continue
        if any(p in name.lower() for p in skip_patterns):
            continue
        if list(layer.submodules):
            continue
        orig_call = layer.call
        layer.call = make_wrapper(name, layer, orig_call, sock, send_lock)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Attention capture
# ---------------------------------------------------------------------------

def maybe_hook_attention(model: "tf.keras.Model", sock: socket.socket,
                         send_lock) -> int:
    count = 0
    for layer in model.submodules:
        name = layer.name
        name_lower = name.lower()
        if "attention" not in name_lower and "self_attn" not in name_lower:
            continue
        if not list(layer.submodules):
            continue
        orig_call = layer.call
        layer.call = _make_attention_wrapper(name, layer, orig_call, sock, send_lock)
        count += 1
    return count


def _make_attention_wrapper(layer_name: str, layer: "tf.keras.layers.Layer",
                            original_call: Callable, sock: socket.socket, send_lock):
    def wrapped(inputs, *args, **kwargs):
        output = original_call(inputs, *args, **kwargs)
        # Some TF models return (output, attention) or (output, states, attention)
        attn_tensor = None
        if isinstance(output, (tuple, list)):
            for item in output[::-1]:
                if isinstance(item, tf.Tensor) and item.shape.rank >= 3:
                    attn_tensor = item
                    break
        if attn_tensor is None:
            return output

        cls_id, layer_idx = classify_name(layer_name)
        try:
            arr = attn_tensor.numpy()
        except Exception:
            return output

        if arr.ndim < 3:
            return output
        batch, heads, q_len, kv_len = arr.shape[:4]
        for h in range(min(heads, 4)):
            weights = arr[0, h, :, :]
            attn_ev = {
                "id": 0,
                "timestamp_ns": int(time.time_ns()),
                "layer_idx": layer_idx,
                "op_class": cls_id,
                "device": 0,
                "node_name": f"{layer_name}.attn_head_{h}",
                "op_name": "attention_weights",
                "shape": [q_len, kv_len, 0, 0],
                "dtype": 0,
                "dtype_size": 4,
                "latency_us": 0,
                "v_min": float(weights.min()),
                "v_max": float(weights.max()),
                "v_mean": float(weights.mean()),
                "v_std": float(weights.std()) if weights.size > 1 else 0.0,
                "l2_norm": float(np.sqrt((weights ** 2).sum())),
                "sparsity": float((np.abs(weights) < 1e-6).mean()),
                "has_nan": bool(np.any(np.isnan(weights))),
                "has_inf": bool(np.any(np.isinf(weights))),
                "stats_valid": True,
                "payload_id": 0,
                "attention_rows": int(q_len),
                "attention_cols": int(kv_len),
                "attention_head": h,
                "attention_weights": weights.ravel().tolist(),
            }
            with send_lock:
                try:
                    line = json.dumps(attn_ev, separators=(",", ":")) + "\n"
                    sock.sendall(line.encode("utf-8"))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
        return output

    return wrapped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local_LLM_Instrumentation TensorFlow/Keras sidecar")
    p.add_argument("--model", type=str, required=True,
                   help="HuggingFace TF model name or path")
    p.add_argument("--port", type=int, default=9876,
                   help="TCP port to connect to (C++ receiver)")
    p.add_argument("--host", type=str, default="127.0.0.1",
                   help="Host to connect to")
    p.add_argument("--prompt", type=str, default="The quick brown fox",
                   help="Input prompt")
    p.add_argument("--max-tokens", type=int, default=32,
                   help="Max tokens to generate")
    p.add_argument("--output-attentions", action="store_true",
                   help="Capture attention matrices")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    _import_tf()

    print(f"TF Keras sidecar connecting to {args.host}:{args.port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30.0)
    try:
        sock.connect((args.host, args.port))
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        return 1
    sock.settimeout(None)
    print(f"Connected to {args.host}:{args.port}")
    send_lock = __import__("threading").Lock()

    # Load model and tokenizer.
    print(f"Loading model {args.model} ...")
    from transformers import TFAutoModelForCausalLM, AutoTokenizer
    model = TFAutoModelForCausalLM.from_pretrained(
        args.model,
        output_attentions=args.output_attentions,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Instrument.
    n_hooks = instrument_model(model, sock, send_lock)
    print(f"Wrapped {n_hooks} leaf-layer call methods")

    if args.output_attentions:
        attn_count = maybe_hook_attention(model, sock, send_lock)
        print(f"Hooked {attn_count} attention containers")

    # Signal readiness.
    ready = json.dumps({"type": "ready", "model": args.model,
                        "hooks": n_hooks}) + "\n"
    with send_lock:
        sock.sendall(ready.encode("utf-8"))

    # Run inference.
    inputs = tokenizer(args.prompt, return_tensors="tf")
    print(f"Generating up to {args.max_tokens} tokens ...")
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        do_sample=False,
        output_attentions=args.output_attentions,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Signal completion. generate() returns a plain tensor, or a ModelOutput
    # (whose tokens live in .sequences) when attention/return-dict options are set.
    seq = getattr(output, "sequences", output)
    done = json.dumps({"type": "done", "tokens": int(seq.shape[1])}) + "\n"
    with send_lock:
        sock.sendall(done.encode("utf-8"))

    print("Inference complete.")
    sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
