#!/usr/bin/env python3
"""Local_LLM_Instrumentation — PyTorch sidecar.

Hooks into a HuggingFace PyTorch model via register_forward_hook,
computes lightweight tensor stats, and streams them as NDJSON over
a TCP socket to the C++ TUI dashboard.

Usage:
  python py_sidecar/sidecar.py --model google/gemma-2b --port 9876 --prompt "Hello world"
  python py_sidecar/sidecar.py --model microsoft/phi-2 --port 9876
"""

import argparse
import json
import os
import socket
import sys
import time
import warnings
from typing import Optional

# This is the PyTorch sidecar — pin transformers to the torch backend so it
# never auto-imports a (possibly broken) TensorFlow/Flax install at import time.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Tensor-level helpers
# ---------------------------------------------------------------------------

# Maps torch dtype to a simple integer matching ggml_type conventions.
DTYPE_MAP: dict[torch.dtype, int] = {
    torch.float32: 0,   # GGML_TYPE_F32
    torch.float16: 1,   # GGML_TYPE_F16
    torch.bfloat16: 30, # GGML_TYPE_BF16
    torch.int8:     24, # GGML_TYPE_I8
    torch.int16:    25, # GGML_TYPE_I16
    torch.int32:    26, # GGML_TYPE_I32
    torch.int64:    26, # I32 — closest match
}


def dtype_code(t: torch.Tensor) -> int:
    return DTYPE_MAP.get(t.dtype, 0)


def dtype_size(t: torch.Tensor) -> int:
    return t.element_size()


def shape_arr(t: torch.Tensor) -> list[int]:
    """Return shape padded to 4 dims (matching kMaxDims=4)."""
    s = list(t.shape)
    while len(s) < 4:
        s.append(0)
    return s[:4]


def compute_stats(t: torch.Tensor) -> dict:
    """Lightweight numeric summary over a float tensor."""
    if t.numel() == 0 or t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return {
            "v_min": 0.0, "v_max": 0.0, "v_mean": 0.0,
            "v_std": 0.0, "l2_norm": 0.0, "sparsity": 0.0,
            "has_nan": False, "has_inf": False, "stats_valid": False,
        }
    ft = t.float().flatten()
    # Sample large tensors to keep overhead bounded.
    max_scan = 65536
    if ft.numel() > max_scan:
        stride = ft.numel() // max_scan
        ft = ft[::stride]
    nan_mask = torch.isnan(ft)
    inf_mask = torch.isinf(ft)
    has_nan = bool(nan_mask.any())
    has_inf = bool(inf_mask.any())
    valid = ft[~(nan_mask | inf_mask)]
    if valid.numel() == 0:
        return {
            "v_min": 0.0, "v_max": 0.0, "v_mean": 0.0,
            "v_std": 0.0, "l2_norm": 0.0, "sparsity": 0.0,
            "has_nan": has_nan, "has_inf": has_inf, "stats_valid": False,
        }
    v_min = float(valid.min())
    v_max = float(valid.max())
    v_mean = float(valid.mean())
    v_std = float(valid.std() if valid.numel() > 1 else 0.0)
    l2_norm = float(torch.sqrt((valid ** 2).sum()))
    sparsity = float((valid.abs() < 1e-6).float().mean())
    return {
        "v_min": v_min, "v_max": v_max, "v_mean": v_mean,
        "v_std": v_std, "l2_norm": l2_norm, "sparsity": sparsity,
        "has_nan": has_nan, "has_inf": has_inf, "stats_valid": True,
    }


# ---------------------------------------------------------------------------
# Layer classification (mirrors src/capture/topology.cpp classify())
# ---------------------------------------------------------------------------

def classify_name(name: str) -> tuple[int, int]:
    """Return (op_class_id, layer_idx) matching ts::OpClass.

    OpClass mapping: 0=Embed, 1=Attn, 2=MLP, 3=Norm, 4=Output, 5=Other.
    Layer index is parsed from the trailing integer after '-' or '.'.
    """
    name_lower = name.lower()
    cls = 5  # Other
    if any(k in name_lower for k in ("embed", "embd", "inp_tokens", "wpe", "wte")):
        cls = 0
    elif any(k in name_lower for k in ("attn", "kqv", "kq", "wqkv", "self_attn",
                                        "q_proj", "k_proj", "v_proj", "o_proj")):
        cls = 1
    elif any(k in name_lower for k in ("mlp", "ffn", "gate_proj", "up_proj",
                                        "down_proj", "fc")):
        cls = 2
    elif "norm" in name_lower:
        cls = 3
    elif any(k in name_lower for k in ("logits", "lm_head", "embed_out", "output")):
        cls = 4
    # Parse trailing integer for layer index.
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
# Forward hook
# ---------------------------------------------------------------------------

_counter: int = 0


def make_hook(module_name: str, sock: socket.socket, send_lock, output_attentions: bool = False):
    """Create a forward hook that sends telemetry for *module_name*."""
    global _counter

    def hook(_mod, _input, output):
        global _counter
        _counter += 1
        ev_id = _counter
        ts_ns = int(time.time_ns())
        cls_id, layer = classify_name(module_name)

        # Determine device.
        try:
            dev = next(_mod.parameters()).device
            dev_str = "CUDA" if dev.type == "cuda" else ("Metal" if dev.type == "mps" else "CPU")
        except (StopIteration, RuntimeError):
            dev_str = "CPU"
        dev_map = {"CPU": 0, "CUDA": 1, "Metal": 2, "Other": 3}

        # Attempt to get a tensor from the output.
        out_tensor: Optional[torch.Tensor] = None
        if isinstance(output, torch.Tensor):
            out_tensor = output
        elif isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor) and item.numel() > 0:
                    out_tensor = item
                    break

        stats = compute_stats(out_tensor) if out_tensor is not None else {
            "v_min": 0.0, "v_max": 0.0, "v_mean": 0.0,
            "v_std": 0.0, "l2_norm": 0.0, "sparsity": 0.0,
            "has_nan": False, "has_inf": False, "stats_valid": False,
        }

        shape = shape_arr(out_tensor) if out_tensor is not None else [0, 0, 0, 0]
        dtype_c = dtype_code(out_tensor) if out_tensor is not None else 0
        dtype_sz = dtype_size(out_tensor) if out_tensor is not None else 0

        ev = {
            "id": ev_id,
            "timestamp_ns": ts_ns,
            "layer_idx": layer,
            "op_class": cls_id,
            "device": dev_map.get(dev_str, 3),
            "node_name": module_name,
            "op_name": type(_mod).__name__,
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

    return hook


# ---------------------------------------------------------------------------
# Model introspection — register hooks on all leaf modules
# ---------------------------------------------------------------------------

def instrument_model(model: nn.Module, sock: socket.socket, send_lock,
                     skip_patterns: tuple = ("dropout", "activation"),
                     output_attentions: bool = False) -> int:
    """Walk *model* and register a forward hook on every leaf nn.Module.

    Returns the number of hooks registered.
    """
    count = 0
    for name, mod in model.named_modules():
        if not name:
            continue
        if any(p in name.lower() for p in skip_patterns):
            continue
        # Only hook leaf modules (no children).
        if list(mod.children()):
            continue
        mod.register_forward_hook(make_hook(name, sock, send_lock, output_attentions))
        count += 1
    return count


# ---------------------------------------------------------------------------
# Attention capture helper
# ---------------------------------------------------------------------------

def maybe_hook_attention_output(model: nn.Module, sock: socket.socket, send_lock) -> int:
    """If the model supports output_attentions, hook the attention output."""
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "output_attentions"):
        pass

    count = 0
    # For models using the standard HuggingFace attention pattern, hook
    # the last attention layer's output to capture the attention matrix.
    for name, mod in model.named_modules():
        name_lower = name.lower()
        is_attn_container = (
            "self_attn" in name_lower
            or "attention" in name_lower
            or name_lower.endswith(".attn")
            or name_lower.endswith("self_attention")
        )
        if not is_attn_container:
            continue
        if not list(mod.children()):
            continue
        mod.register_forward_hook(
            _make_attention_output_hook(name, sock, send_lock)
        )
        count += 1
    return count


def _make_attention_output_hook(module_name: str, sock: socket.socket, send_lock):
    """Extract attention probabilities if available."""
    def hook(mod, _input, output):
        # Check if output is a tuple with attention at position 1 or -1
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            attn = output[-1]  # often the last element is attention probs
            if isinstance(attn, torch.Tensor) and attn.dim() >= 3:
                # attn shape: [batch, heads, q_len, kv_len]
                cls_id, layer = classify_name(module_name)
                batch, heads, q_len, kv_len = attn.shape
                for h in range(min(heads, 4)):  # cap at 4 heads
                    weights = attn[0, h].float().cpu().numpy()
                    attn_ev = {
                        "id": 0,
                        "timestamp_ns": int(time.time_ns()),
                        "layer_idx": layer,
                        "op_class": cls_id,
                        "device": 0,
                        "node_name": f"{module_name}.attn_head_{h}",
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
                        "attention_weights": weights.flatten().tolist(),
                    }
                    with send_lock:
                        try:
                            line = json.dumps(attn_ev, separators=(",", ":")) + "\n"
                            sock.sendall(line.encode("utf-8"))
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            pass
    return hook


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local_LLM_Instrumentation PyTorch sidecar")
    p.add_argument("--model", type=str, required=True,
                   help="HuggingFace model name or path")
    p.add_argument("--port", type=int, default=9876,
                   help="TCP port to connect to (C++ receiver)")
    p.add_argument("--host", type=str, default="127.0.0.1",
                   help="Host to connect to")
    p.add_argument("--prompt", type=str, default="The quick brown fox",
                   help="Input prompt for inference")
    p.add_argument("--max-tokens", type=int, default=32,
                   help="Max tokens to generate")
    p.add_argument("--device", type=str, default="auto",
                   help="Device (cpu, cuda, mps, auto)")
    p.add_argument("--output-attentions", action="store_true",
                   help="Capture attention matrices")
    p.add_argument("--dtype", type=str, default="auto",
                   help="Model dtype (auto, float16, bfloat16, float32)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve device.
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Resolve dtype.
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = dtype_map.get(args.dtype, "auto")

    print(f"PyTorch sidecar connecting to {args.host}:{args.port} ...")
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

    # Load model.
    print(f"Loading model {args.model} on {device} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device if device == "auto" else None,
        output_attentions=args.output_attentions,
        trust_remote_code=True,
    )
    if device != "auto":
        model = model.to(device)
    model.eval()

    # Force output_attentions so attention-weight hooks can capture matrices.
    if args.output_attentions:
        model.config.output_attentions = True

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Instrument.
    n_hooks = instrument_model(model, sock, send_lock)
    print(f"Registered {n_hooks} forward hooks")

    if args.output_attentions:
        attn_count = maybe_hook_attention_output(model, sock, send_lock)
        print(f"Registered {attn_count} attention hooks")

    # Signal readiness.
    ready = json.dumps({"type": "ready", "model": args.model,
                        "hooks": n_hooks}) + "\n"
    with send_lock:
        sock.sendall(ready.encode("utf-8"))

    # Run inference.
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    print(f"Generating up to {args.max_tokens} tokens ...")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            output_attentions=args.output_attentions,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Signal completion. generate() returns a plain tensor, or a
    # GenerateDecoderOnlyOutput (whose tokens live in .sequences) when
    # return-dict/attention options promote the result to a ModelOutput.
    seq = getattr(generated, "sequences", generated)
    done = json.dumps({"type": "done", "tokens": int(seq.shape[1])}) + "\n"
    with send_lock:
        sock.sendall(done.encode("utf-8"))

    print("Inference complete.")
    sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
