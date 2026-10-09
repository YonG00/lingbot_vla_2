"""Opt-in, low-CPU-memory loading for deploy-only safetensors checkpoints.

This module intentionally does not alter the training/DCP loading path.  It
loads *into existing model storage* to retain tied parameters and nonpersistent
buffers.  The old deployment loader remains available by default.
"""

from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open


@contextmanager
def model_init_on_device(device: str, dtype: torch.dtype):
    """Instantiate the model directly at its inference device/dtype.

    Scope the default-device/dtype change to model construction only.  In
    particular, checkpoint reading and processor/tokenizer setup run outside.
    """
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(device):
            yield
    finally:
        torch.set_default_dtype(original_dtype)


def fast_load_enabled() -> bool:
    value = os.environ.get("LINGBOT_DEPLOY_FAST_LOAD", "0")
    if value not in ("0", "1"):
        raise ValueError("LINGBOT_DEPLOY_FAST_LOAD must be 0 or 1")
    return value == "1"


def _read_headers(directory: str):
    """Read only safetensors headers; do not materialize checkpoint tensors."""
    files = sorted(Path(directory).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors checkpoint shards under {directory}")
    locations = {}
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in locations:
                    raise RuntimeError(f"Duplicate checkpoint tensor {key!r}: {locations[key][0]} and {file}")
                locations[key] = (file, tuple(reader.get_slice(key).get_shape()))
    return files, locations


def check_cuda_load_budget(directory: str, *, reserve_gib: float = 10.0):
    """Conservative preflight before allocating a 6B model on the GPU.

    The estimate assumes BF16 for checkpoint tensors.  It leaves room for
    constructor temporaries, CUDA kernels, and the requested free reserve.
    """
    _, headers = _read_headers(directory)
    estimated_bytes = sum(math.prod(shape) * 2 for _, shape in headers.values())
    free_bytes, _ = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    needed = int(estimated_bytes * 1.6 + (reserve_gib + 4.0) * gib)
    print(
        f"[deploy-fast-load] budget checkpoint_bf16={estimated_bytes / gib:.2f}GiB "
        f"cuda_free={free_bytes / gib:.2f}GiB required_free={needed / gib:.2f}GiB",
        flush=True,
    )
    if free_bytes < needed:
        raise RuntimeError(
            "Insufficient free CUDA memory for opt-in fast load: "
            f"available={free_bytes / gib:.2f}GiB required={needed / gib:.2f}GiB. "
            "Disable LINGBOT_DEPLOY_FAST_LOAD to use the original CPU path."
        )


def require_model_dtype(model: torch.nn.Module, dtype: torch.dtype, device: str):
    """Refuse partially CPU/FP32 builds instead of silently changing numerics."""
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.device.type != device:
            raise RuntimeError(
                f"Fast model init produced {name} on {tensor.device}; expected {device}"
            )
        if tensor.is_floating_point() and tensor.dtype != dtype:
            raise RuntimeError(
                f"Fast model init produced {name} as {tensor.dtype}; expected {dtype}. "
                "Disable LINGBOT_DEPLOY_FAST_LOAD for this model version."
            )


@torch.no_grad()
def load_safetensors_streaming(model: torch.nn.Module, directory: str, *, strict: bool = True):
    """Preflight all names/shapes, then copy one tensor at a time into model.

    The fast path requires CUDA-resident, already correctly typed model tensors.
    The preflight fails *before* any model weight is changed.
    """
    started = time.perf_counter()
    files, locations = _read_headers(directory)
    destination = model.state_dict(keep_vars=True)
    expected_keys = set(destination)
    actual_keys = set(locations)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    bad_shapes = [
        (key, locations[key][1], tuple(destination[key].shape))
        for key in sorted(expected_keys & actual_keys)
        if locations[key][1] != tuple(destination[key].shape)
    ]
    if bad_shapes or (strict and (missing or unexpected)):
        raise RuntimeError(
            "Checkpoint incompatible (no weights copied): "
            f"missing={missing[:8]} ({len(missing)}), "
            f"unexpected={unexpected[:8]} ({len(unexpected)}), "
            f"shape_mismatch={bad_shapes[:5]} ({len(bad_shapes)})"
        )
    for name in expected_keys & actual_keys:
        if destination[name].device.type == "meta":
            raise RuntimeError(f"Model tensor {name} is still on meta; streaming needs allocated storage")
    count = 0
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                target = destination.get(key)
                if target is None:
                    continue
                source = reader.get_tensor(key)
                # torch.Tensor.copy_ converts to the target tensor's dtype,
                # without allocating a second complete checkpoint dictionary.
                target.copy_(source)
                count += 1
                del source
    seconds = time.perf_counter() - started
    print(f"[deploy-fast-load] streaming tensors={count} shards={len(files)} seconds={seconds:.2f}", flush=True)
    return {"tensors": count, "shards": len(files), "seconds": seconds}
