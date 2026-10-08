"""HF 直出精度（dtype 脱钩）的 CPU 专项测试。

用户 2026-10-08 要求：
  * 支持 `hf_export_dtype = bf16 | fp32 | native`，与训练/评测精度解耦；
  * **只转换浮点权重**，整数/布尔等非浮点状态绝不转换；
  * bf16 回读结果必须等于"源权重转 bf16"的结果；
  * fp32 导出不得发生意外降精度；
  * 源为 fp32 而导出 bf16 时必须标记为**有意降精度、并非无损**；
  * 原子发布、失败清理、rank0 写盘、旧参数兼容保持。

全部用例只依赖 torch（CPU），不需要 GPU。
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
import torch

from lingbotvla.utils.direct_hf_checkpoint import (
    export_model_hf_direct,
    format_export_report,
    prepare_export_tensors,
    resolve_export_dtype,
)


class TinyModel(torch.nn.Module):
    """浮点权重 + 整数/布尔状态，用来验证"只转浮点"。"""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 3)
        self.register_buffer("step_ids", torch.arange(6, dtype=torch.int64))
        self.register_buffer("mask", torch.tensor([True, False, True]))
        self.register_buffer("half_weight", torch.randn(2, dtype=torch.bfloat16))


def _snapshot(model: TinyModel, dtype=torch.float32):
    snap = {"fc.weight": model.fc.weight.detach().to(dtype).clone(),
            "fc.bias": model.fc.bias.detach().to(dtype).clone(),
            "step_ids": model.step_ids.clone(),
            "mask": model.mask.clone(),
            "half_weight": model.half_weight.clone()}
    return snap


def _real_saver_importable() -> bool:
    """本机 transformers 版本可能过新（缺 AutoModelForVision2Seq）⇒ 端到端用例改在远端跑。"""
    try:
        import lingbotvla.models  # noqa: F401
        return True
    except Exception:
        return False


needs_real_saver = pytest.mark.skipif(
    not _real_saver_importable(),
    reason="本地 transformers 与仓库不匹配（缺 AutoModelForVision2Seq）；端到端保存/回读在远端 conda 环境跑",
)

# --------------------------------------------------------------------------- #
# 1) dtype 解析
# --------------------------------------------------------------------------- #
def test_resolve_export_dtype_modes():
    assert resolve_export_dtype("native") is None
    assert resolve_export_dtype("bf16") is torch.bfloat16
    assert resolve_export_dtype("fp32") is torch.float32
    assert resolve_export_dtype(None) is None
    assert resolve_export_dtype(torch.float32) is torch.float32
    # 旧参数别名
    assert resolve_export_dtype("native", save_dtype=torch.float32) is torch.float32
    assert resolve_export_dtype("bf16", save_dtype="float32") is torch.float32
    with pytest.raises(ValueError, match="unsupported hf export dtype"):
        resolve_export_dtype("int8")


# --------------------------------------------------------------------------- #
# 2) 张量整理与报告
# --------------------------------------------------------------------------- #
def test_native_keeps_every_dtype_untouched():
    snap = {"w": torch.randn(2, 2, dtype=torch.float32),
            "h": torch.randn(2, dtype=torch.bfloat16),
            "i": torch.arange(3, dtype=torch.int64),
            "b": torch.tensor([True, False])}
    out, rep = prepare_export_tensors(snap, "native")
    assert rep["target"] == "native" and rep["converted"] == 0
    assert rep["not_lossless"] is False
    for k in snap:
        assert out[k] is snap[k], "native 模式不得复制或转换"
        assert out[k].dtype == snap[k].dtype


def test_bf16_converts_only_floats_and_flags_lossy_downcast():
    snap = {"w": torch.randn(3, 3, dtype=torch.float32),
            "h": torch.randn(3, dtype=torch.bfloat16),
            "i": torch.arange(4, dtype=torch.int32),
            "b": torch.tensor([True, False, True])}
    out, rep = prepare_export_tensors(snap, "bf16")
    assert out["w"].dtype == torch.bfloat16
    assert out["h"].dtype == torch.bfloat16
    # 非浮点：原样保留（绝不转换）
    assert out["i"].dtype == torch.int32 and torch.equal(out["i"], snap["i"])
    assert out["b"].dtype == torch.bool and torch.equal(out["b"], snap["b"])
    assert rep["kept_non_float"] == 2
    assert rep["converted"] == 1, "只有 fp32 的 w 需要转换（h 已是 bf16）"
    assert rep["downcast_from"] == ["fp32"] and rep["not_lossless"] is True
    # 回读语义：等于源转 bf16
    assert torch.equal(out["w"], snap["w"].to(torch.bfloat16))


def test_fp32_never_downcasts_bf16_source_silently():
    snap = {"w": torch.randn(2, 2, dtype=torch.float32),
            "h": torch.randn(4, dtype=torch.bfloat16),
            "i": torch.arange(2, dtype=torch.int64)}
    out, rep = prepare_export_tensors(snap, "fp32")
    assert out["w"].dtype == torch.float32 and torch.equal(out["w"], snap["w"]), "fp32 导出不得改变 fp32 权重"
    assert out["h"].dtype == torch.float32, "bf16→fp32 是升精度（无损）"
    assert out["i"].dtype == torch.int64
    assert rep["downcast_from"] == [] and rep["not_lossless"] is False


def test_export_report_mentions_not_lossless_when_downcasting():
    _, rep = prepare_export_tensors({"w": torch.randn(2, dtype=torch.float32)}, "bf16")
    msg = format_export_report(rep)
    assert "bf16" in msg and "并非无损" in msg and "fp32" in msg
    _, rep2 = prepare_export_tensors({"w": torch.randn(2, dtype=torch.float32)}, "fp32")
    assert "并非无损" not in format_export_report(rep2)


# --------------------------------------------------------------------------- #
# 3) 端到端：真实保存 + 回读
# --------------------------------------------------------------------------- #
def _read_back(path: str) -> dict:
    from safetensors.torch import load_file

    merged: dict = {}
    for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        merged.update(load_file(shard))
    return merged


@needs_real_saver
def test_bf16_export_roundtrip_matches_source_cast(tmp_path):
    model = TinyModel()
    snap = _snapshot(model, torch.float32)
    out = export_model_hf_direct(model, global_step=7, checkpoint_root=str(tmp_path),
                                 export_dtype="bf16")
    assert out and os.path.isdir(out)
    disk = _read_back(out)
    assert set(disk) == set(snap)
    for k, src in snap.items():
        got = disk[k]
        if src.is_floating_point():
            assert got.dtype == torch.bfloat16
            assert torch.equal(got.to(torch.float32), src.to(torch.bfloat16).to(torch.float32)), k
        else:
            assert got.dtype == src.dtype, f"{k} 非浮点被改写了 dtype"
            assert torch.equal(got, src), k
    assert not glob.glob(os.path.join(os.path.dirname(out), ".hf_ckpt.tmp.*")), "不得留下 tmp 目录"


@needs_real_saver
def test_fp32_export_is_lossless(tmp_path):
    model = TinyModel()
    snap = _snapshot(model, torch.float32)
    out = export_model_hf_direct(model, global_step=8, checkpoint_root=str(tmp_path),
                                 export_dtype="fp32")
    disk = _read_back(out)
    for k, src in snap.items():
        got = disk[k]
        if not src.is_floating_point():
            assert got.dtype == src.dtype, f"{k} 非浮点被改写"
            assert torch.equal(got, src), k
            continue
        if src.dtype == torch.float32:
            assert got.dtype == torch.float32, f"{k} fp32 权重被降精度"
            assert torch.equal(got, src), k
        else:
            # bf16→fp32 是**无损升精度**（存储更大但不丢信息）：回读值必须等于源值
            assert got.dtype == torch.float32, k
            assert torch.equal(got.to(src.dtype), src), f"{k} 升精度过程丢信息"


@needs_real_saver
def test_native_export_default_preserves_mixed_dtypes(tmp_path):
    model = TinyModel()
    out = export_model_hf_direct(model, global_step=9, checkpoint_root=str(tmp_path))
    disk = _read_back(out)
    assert disk["half_weight"].dtype == torch.bfloat16, "native 应保留 bf16 缓冲"
    assert disk["step_ids"].dtype == torch.int64
    assert disk["mask"].dtype == torch.bool
    assert disk["fc.weight"].dtype == torch.float32


@needs_real_saver
def test_legacy_save_dtype_alias_still_works(tmp_path):
    model = TinyModel()
    out = export_model_hf_direct(model, global_step=10, checkpoint_root=str(tmp_path),
                                 save_dtype=torch.bfloat16)
    assert _read_back(out)["fc.weight"].dtype == torch.bfloat16


# --------------------------------------------------------------------------- #
# 4) 原子发布 / 失败清理 / rank0 协议
# --------------------------------------------------------------------------- #
def test_failed_save_leaves_nothing(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    def boom(*a, **kw):
        raise RuntimeError("disk on fire")

    fake = ModuleType("lingbotvla.models")
    fake.save_model_weights = boom          # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lingbotvla.models", fake)
    model = TinyModel()
    with pytest.raises(RuntimeError, match="Direct HF export"):
        export_model_hf_direct(model, global_step=11, checkpoint_root=str(tmp_path))
    assert not glob.glob(str(tmp_path / "global_step_11*")), "失败不得留下已完成目录"
    assert not glob.glob(str(tmp_path / ".hf_ckpt.tmp.*")), "失败必须清理 tmp"


def test_empty_save_is_treated_as_failure(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    fake = ModuleType("lingbotvla.models")
    fake.save_model_weights = lambda path, *a, **kw: os.makedirs(path, exist_ok=True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lingbotvla.models", fake)
    with pytest.raises(RuntimeError, match="Direct HF export"):
        export_model_hf_direct(TinyModel(), global_step=12, checkpoint_root=str(tmp_path))
    assert not glob.glob(str(tmp_path / "global_step_12*"))


def test_non_rank0_writes_nothing(tmp_path, monkeypatch):
    import lingbotvla.utils.direct_hf_checkpoint as dhc

    monkeypatch.setattr(dhc, "_distributed", lambda: True)
    monkeypatch.setattr(dhc.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dhc.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dhc.dist, "all_gather_object", lambda out, obj: out.__setitem__(slice(None), [obj, obj]))
    monkeypatch.setattr(dhc.dist, "broadcast_object_list", lambda out, src=0: None)

    def must_not_run(*a, **kw):  # pragma: no cover - 触发即失败
        raise AssertionError("rank!=0 不得写盘")

    monkeypatch.setattr(dhc, "collect_full_model_on_cpu", lambda model: None)
    import sys
    from types import ModuleType

    fake = ModuleType("lingbotvla.models")
    fake.save_model_weights = must_not_run      # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lingbotvla.models", fake)
    result = export_model_hf_direct(TinyModel(), global_step=13, checkpoint_root=str(tmp_path))
    assert result is None
    assert list(Path(tmp_path).iterdir()) == []


@needs_real_saver
def test_resume_retry_suffix_keeps_previous_milestone(tmp_path):
    model = TinyModel()
    first = export_model_hf_direct(model, global_step=14, checkpoint_root=str(tmp_path), export_dtype="fp32")
    second = export_model_hf_direct(model, global_step=14, checkpoint_root=str(tmp_path), export_dtype="fp32")
    assert first != second and "_retry_001" in second
    assert os.path.isdir(first) and os.path.isdir(second)
