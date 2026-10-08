"""Export a full HF snapshot from a *live* model, without writing DCP.

All distributed ranks MUST call this at the same completed optimizer step.
StateDictOptions(full_state_dict=True, cpu_offload=True) uses collectives and
places the full state on rank 0.  Snapshot is written synchronously, so model
weights cannot change during HF serialization and no background CUDA/FSDP
collective races the next training step.
"""
from __future__ import annotations

import os
import shutil
import torch
import torch.distributed as dist


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def collect_full_model_on_cpu(model):
    """Collect all-rank FSDP1/FSDP2 state with official PyTorch API."""
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_model_state_dict,
    )
    full = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    if _distributed() and dist.get_rank() != 0:
        return None
    snapshot = {}
    for key, tensor in full.items():
        if not isinstance(tensor, torch.Tensor) or getattr(tensor, "is_meta", False):
            raise TypeError(f"HF snapshot {key!r} is not a materialized Tensor")
        if tensor.device.type != "cpu":
            tensor = tensor.detach().to("cpu")
        snapshot[key] = tensor.detach().contiguous().clone()
    return snapshot


def export_model_hf_direct(model, *, global_step: int, checkpoint_root: str,
                           model_assets=None, save_dtype=torch.bfloat16):
    """All ranks capture the same step; rank0 writes one atomic HF snapshot.

    Path is separate from DCP resume candidates. On a replay of the same
    global step from an earlier DCP, keep the old milestone and add _retry_N.
    """
    snapshot = None
    capture_error = None
    try:
        with torch.no_grad():
            snapshot = collect_full_model_on_cpu(model)
    except Exception as exc:
        capture_error = repr(exc)
    if _distributed():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, capture_error)
        if any(err is not None for err in errors):
            raise RuntimeError(f"Direct HF state capture failed: {errors}")
    elif capture_error is not None:
        raise RuntimeError(f"Direct HF state capture failed: {capture_error}")

    rank0 = not _distributed() or dist.get_rank() == 0
    error = None
    destination = None
    if rank0:
        temp_dir = None
        try:
            from lingbotvla.models import save_model_weights
            retry = 0
            while True:
                suffix = f"global_step_{global_step}"
                if retry:
                    suffix += f"_retry_{retry:03d}"
                candidate = os.path.join(checkpoint_root, suffix, "hf_ckpt")
                if not os.path.lexists(candidate):
                    destination = candidate
                    break
                retry += 1
                if retry > 999:
                    raise RuntimeError("HF milestone collision limit exceeded")

            temp_dir = os.path.join(os.path.dirname(destination),
                                    f".hf_ckpt.tmp.{global_step}.{os.getpid()}")
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            # Fail before large writes if the filesystem is clearly full.
            need = sum(t.numel() * torch.empty((), dtype=save_dtype).element_size()
                       for t in snapshot.values())
            free = shutil.disk_usage(os.path.dirname(destination)).free
            if free < need * 1.10:
                raise OSError(f"not enough free space for HF ({free} < {int(need*1.10)} bytes)")
            save_model_weights(temp_dir, snapshot, save_dtype=save_dtype,
                               model_assets=model_assets)
            os.replace(temp_dir, destination)
        except Exception as exc:
            error = f"Direct HF export step={global_step} failed: {exc!r}"
        finally:
            if temp_dir is not None and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    del snapshot
    if _distributed():
        outcome = [error]
        dist.broadcast_object_list(outcome, src=0)
        error = outcome[0]
    if error:
        raise RuntimeError(error)
    return destination if rank0 else None
