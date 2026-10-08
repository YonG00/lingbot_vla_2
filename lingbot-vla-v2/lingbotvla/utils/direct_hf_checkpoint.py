"""Export a full HF snapshot from a *live* model, without writing DCP.

All distributed ranks MUST call this at the same completed optimizer step.
StateDictOptions(full_state_dict=True, cpu_offload=True) uses collectives and
places the full state on rank 0.  Snapshot is written synchronously, so model
weights cannot change during HF serialization and no background CUDA/FSDP
collective races the next training step.
"""
from __future__ import annotations

import json
import os
import shutil
import torch
import torch.distributed as dist


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


#: 用户口径 → torch.dtype（None = native，保持每个张量自身 dtype）
_DTYPE_ALIASES = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp32": torch.float32, "float32": torch.float32,
    "fp16": torch.float16, "float16": torch.float16,
    "native": None,
}


def _short_dtype(dtype) -> str:
    """torch dtype → 与 `hf_export_dtype` 一致的短名（bf16/fp32/fp16/int64/bool…）。"""
    name = str(dtype).replace("torch.", "")
    return {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16",
            "float64": "fp64"}.get(name, name)


def resolve_export_dtype(export_dtype, save_dtype=None):
    """把 `native|bf16|fp32`（或旧 `save_dtype=` torch.dtype）解析为 torch.dtype；None=不改。

    旧参数 `save_dtype` 保留为兼容别名：给了它就以它为准（torch.float32→fp32 等）。
    """
    if save_dtype is not None:
        if isinstance(save_dtype, torch.dtype):
            return save_dtype
        key = str(save_dtype).lower().replace("torch.", "")
        if key not in _DTYPE_ALIASES:
            raise ValueError(f"unsupported legacy save_dtype: {save_dtype!r}")
        return _DTYPE_ALIASES[key]
    if export_dtype is None:
        return None
    if isinstance(export_dtype, torch.dtype):
        return export_dtype
    key = str(export_dtype).lower().replace("torch.", "")
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"unsupported hf export dtype: {export_dtype!r}（可选 native/bf16/fp32）")
    return _DTYPE_ALIASES[key]


def prepare_export_tensors(snapshot, export_dtype="native", *, save_dtype=None):
    """整理待写出的张量，并给出**真实**精度报告（用户 2026-10-08 要求）。

    规则：
      * `native`：**完全不动**每个张量的 dtype（不复制、不转换）；
      * 指定 bf16/fp32：**只转换浮点张量**；整数/布尔/其它非浮点状态**原样保留**
        （绝不为存储精度破坏整型状态，例如 position ids / mask / 步数计数）；
      * 报告源 dtype 直方图、目标 dtype、转换数量，以及**是否发生降精度**
        （如 fp32 主权重 → bf16 存储 ⇒ 明确标记为"有意降低存储精度、并非无损"）。
    """
    target = resolve_export_dtype(export_dtype, save_dtype)
    report = {
        "target": "native" if target is None else _short_dtype(target),
        "source_dtypes": {}, "converted": 0, "kept_non_float": 0,
        "downcast_from": [], "not_lossless": False,
    }
    prepared = {}
    downcast_sources = set()
    for key, tensor in snapshot.items():
        name = _short_dtype(tensor.dtype)
        report["source_dtypes"][name] = report["source_dtypes"].get(name, 0) + 1
        if target is None:
            prepared[key] = tensor
            continue
        if not tensor.is_floating_point():
            report["kept_non_float"] += 1          # 非浮点：绝不转换
            prepared[key] = tensor
            continue
        if tensor.dtype != target:
            if tensor.element_size() > torch.empty((), dtype=target).element_size():
                downcast_sources.add(name)
            tensor = tensor.to(dtype=target)
            report["converted"] += 1
        prepared[key] = tensor
    if downcast_sources:
        report["downcast_from"] = sorted(downcast_sources)
        report["not_lossless"] = True
    return prepared, report


def format_export_report(report) -> str:
    """一行可写进训练日志的精度说明（含"并非无损"的显式提示）。"""
    src = ", ".join(f"{k}×{v}" for k, v in sorted(report["source_dtypes"].items()))
    msg = (f"源 dtype {{{src}}} → 目标 {report['target']}｜转换 {report['converted']} 个浮点张量"
           f"｜保留非浮点 {report['kept_non_float']} 个")
    if report["not_lossless"]:
        msg += (f"｜⚠️ 有意降低存储精度（{'+'.join(report['downcast_from'])}→{report['target']}），"
                "**并非无损**：回读数值 = 源权重按目标 dtype 舍入后的结果")
    return msg


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
    distributed = _distributed()
    snapshot = {}
    for key, tensor in full.items():
        if not isinstance(tensor, torch.Tensor) or getattr(tensor, "is_meta", False):
            raise TypeError(f"HF snapshot {key!r} is not a materialized Tensor")
        # 记录"变换前"的存储指针，用来判断是否还需要复制（用户 10-08 复核 CPU clone）：
        #   * 单进程：`get_model_state_dict` 返回的是**活参数本身**（别名）⇒ 必须复制，
        #     否则训练继续更新权重会让快照变质（tests 的 live_snapshot 用例守这条）；
        #   * 分布式 + cpu_offload：全量 state 是本次 gather **物化出的独立 CPU 张量** ⇒ 无需复制，
        #     省掉一份 ×2 的 CPU 峰值（bf16 6B ≈ 12 GiB 而非 24 GiB）。
        src_ptr = tensor.data_ptr()
        if tensor.device.type != "cpu":
            tensor = tensor.detach().to("cpu")   # 这一步本身已是一次复制
        else:
            tensor = tensor.detach()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()         # 非连续时也会产生新存储
        if (not distributed) and tensor.data_ptr() == src_ptr:
            tensor = tensor.clone()              # 仍与活参数共享存储 ⇒ 必须复制
        snapshot[key] = tensor
    return snapshot


def verify_hf_weight_files(output_dir, expected_keys):
    """Fail closed before publishing a milestone or committing the PASS cursor.

    Check the *actual* safetensors headers, not merely whether some assets or
    an index file happen to exist.  This does not load 12 GiB of weights into
    RAM; ``safe_open.keys()`` reads only tensor metadata.
    """
    from safetensors import safe_open

    shards = sorted(name for name in os.listdir(output_dir)
                    if name.endswith(".safetensors") and
                    os.path.isfile(os.path.join(output_dir, name)))
    if not shards:
        raise OSError("HF export has no safetensors weight shard")

    index_file = os.path.join(output_dir, "model.safetensors.index.json")
    if os.path.isfile(index_file):
        with open(index_file, encoding="utf-8") as fh:
            index = json.load(fh)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or set(weight_map) != set(expected_keys):
            raise OSError("HF weight index does not cover the complete model state")
        if set(weight_map.values()) != set(shards):
            raise OSError("HF weight index disagrees with shard files")
    elif len(shards) != 1:
        raise OSError("HF multi-shard weights missing safetensors index")

    actual = set()
    for name in shards:
        path = os.path.join(output_dir, name)
        if os.path.getsize(path) == 0:
            raise OSError(f"HF weight shard is empty: {name}")
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
        if not keys or actual.intersection(keys):
            raise OSError(f"HF shard is empty or contains duplicate keys: {name}")
        if os.path.isfile(index_file):
            if any(weight_map.get(k) != name for k in keys):
                raise OSError(f"HF index contains incorrect shard mapping: {name}")
        actual.update(keys)
    if actual != set(expected_keys):
        raise OSError(f"HF tensor coverage mismatch: missing={len(set(expected_keys)-actual)}, "
                      f"extra={len(actual-set(expected_keys))}")


def export_model_hf_direct(model, *, global_step: int, checkpoint_root: str,
                           model_assets=None, export_dtype="native", save_dtype=None,
                           logger=None):
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
            # 精度脱钩：只按目标 dtype 转换**浮点**权重，非浮点状态原样保留（见 prepare_export_tensors）。
            prepared, dtype_report = prepare_export_tensors(
                snapshot, export_dtype, save_dtype=save_dtype)
            if logger is not None:
                logger.info_rank0(f"[ckpt] HF 导出精度：{format_export_report(dtype_report)}")
            # Fail before large writes if the filesystem is clearly full.
            need = sum(t.numel() * t.element_size() for t in prepared.values())
            free = shutil.disk_usage(os.path.dirname(destination)).free
            if free < need * 1.10:
                raise OSError(f"not enough free space for HF ({free} < {int(need*1.10)} bytes)")
            # ⚠️ 传 `save_dtype=None`：否则 `save_model_weights` 会把**所有**张量一律 cast，
            #    包括 int/bool 状态（那会破坏整型语义）。转换已在上面的 prepared 里做过。
            save_model_weights(temp_dir, prepared, save_dtype=None,
                               model_assets=model_assets)
            del prepared
            # 禁止把仅有 config/半套权重/损坏索引的 HF 里程碑发布成成功。
            verify_hf_weight_files(temp_dir, snapshot.keys())
            os.replace(temp_dir, destination)
        except Exception as exc:
            error = f"Direct HF export step={global_step} failed: {exc!r}"
        finally:
            if temp_dir is not None and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            # 失败时上面 `os.makedirs(os.path.dirname(destination))` 会留下空的
            # `<root>/global_step_N/` 空壳 ⇒ 顺手清掉（成功时非空 ⇒ rmdir 失败，忽略）。
            if destination is not None:
                try:
                    os.rmdir(os.path.dirname(destination))
                except OSError:
                    pass
    del snapshot
    if _distributed():
        outcome = [error]
        dist.broadcast_object_list(outcome, src=0)
        error = outcome[0]
    if error:
        raise RuntimeError(error)
    return destination if rank0 else None
