"""CPU 替身诊断入口的自测（假模型不变式 + 反例清单 + fail-closed 入口）。

本机（macOS，无 torchdata）只能跑不依赖真实数据集的部分；完整端到端
（真实 `_infer_one` / `_infer_batch`）在训练机上跑，未满足依赖时**显式 skip**。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from tools import eval_batch_cpu_diag as diag  # noqa: E402


def _real_path_available() -> bool:
    try:
        import lingbotvla.utils.open_loop_validation  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# 假模型必须是**逐样本**的（这是"正常组必须逐位相等"的前提）
# ---------------------------------------------------------------------------
def _call(model, items, noise):
    images = torch.stack([it["images"] for it in items])
    return model.sample_actions(
        images,
        torch.stack([it["img_masks"] for it in items]),
        torch.stack([it["lang_tokens"] for it in items]),
        torch.stack([it["lang_masks"] for it in items]),
        torch.stack([it["state"] for it in items]),
        noise=noise, image_grid_thw=torch.stack([it["image_grid_thw"] for it in items]))


def test_fake_model_is_per_sample_bitwise_invariant():
    """逐样本不变式：batch 的第 i 条输出必须与单独喂第 i 条**逐位相同**。

    ⚠️ 2026-10-09 的坑：第一版假模型用 `images.mean()`（跨样本求均值）⇒ 正常组
    batch/serial 也会差 0.003，把"逐位相等"退化成"差不多" ⇒ 检测器失去分辨力。
    """
    model = diag.FakeModel()
    items = [diag.make_item(0, 0.5), diag.make_item(50, 0.7)]
    noise = torch.arange(2 * diag.N_ACTION_STEPS * diag.MAX_ACTION_DIM,
                         dtype=torch.float32).reshape(2, diag.N_ACTION_STEPS,
                                                      diag.MAX_ACTION_DIM)
    batched = _call(model, items, noise)
    assert batched.shape == (2, diag.N_ACTION_STEPS, diag.MAX_ACTION_DIM)
    for position, item in enumerate(items):
        single = _call(model, [item], noise[position:position + 1])
        assert torch.equal(batched[position], single[0]), f"position {position} 不是逐位相同"


def test_fake_model_responds_to_its_own_input():
    """逐样本，但**仍然**依赖该样本自己的输入（否则"错误输入"反例会失效）。"""
    model = diag.FakeModel()
    a = _call(model, [diag.make_item(0, 0.5)], torch.zeros(1, diag.N_ACTION_STEPS,
                                                          diag.MAX_ACTION_DIM))
    b_item = diag.make_item(0, 0.9)
    b = _call(model, [b_item], torch.zeros(1, diag.N_ACTION_STEPS, diag.MAX_ACTION_DIM))
    assert not torch.equal(a[0], b[0])


def test_fake_model_batch_bias_and_nan_are_injectable():
    items = [diag.make_item(0, 0.5), diag.make_item(50, 0.7)]
    noise = torch.zeros(2, diag.N_ACTION_STEPS, diag.MAX_ACTION_DIM)

    def _call_batch(model):
        return model.sample_actions(
            torch.stack([it["images"] for it in items]),
            torch.stack([it["img_masks"] for it in items]),
            torch.stack([it["lang_tokens"] for it in items]),
            torch.stack([it["lang_masks"] for it in items]),
            torch.stack([it["state"] for it in items]),
            noise=noise, image_grid_thw=None)

    clean = _call_batch(diag.FakeModel())
    biased = _call_batch(diag.FakeModel(batch_bias=1e-3))
    # 偏差只随**批内位置**线性增长（批处理特有偏差的最小复现）
    assert torch.allclose(biased[0] - clean[0], torch.full_like(clean[0], 1e-3), atol=1e-9)
    assert torch.allclose(biased[1] - clean[1], torch.full_like(clean[1], 2e-3), atol=1e-9)
    with_nan = _call_batch(diag.FakeModel(nan_in_batch=True))
    assert bool(torch.isnan(with_nan[0, 0, 0]))
    assert not bool(torch.isnan(with_nan[1]).any())


# ---------------------------------------------------------------------------
# 假数据必须照真实契约（lang_tokens/lang_masks/state 逐条 1-D；images 逐条 3-D）
# ---------------------------------------------------------------------------
def test_fake_item_follows_real_contract():
    item = diag.make_item(0, 0.5)
    assert item["images"].ndim == 3
    assert item["img_masks"].ndim == 1
    assert item["lang_tokens"].ndim == 1
    assert item["lang_masks"].ndim == 1
    assert item["state"].ndim == 1
    assert tuple(item["image_grid_thw"].shape) == (1, 3)
    assert item["actions"].shape == (diag.N_ACTION_STEPS, diag.MAX_ACTION_DIM)


@pytest.mark.skipif(not _real_path_available(),
                    reason="需要完整训练依赖（torchdata / transformers）")
def test_shim_provides_dump_dir_and_probe_state():
    """`_infer_one` 会读 `self.dump_dir`；诊断属性必须齐全（否则 AttributeError）。"""
    validator = diag.build_validator(diag.FakeModel(), diag._Logger(verbose=False))
    assert validator.dump_dir is None
    assert validator._probe_recorder is None
    assert validator._probe_ep_map is None
    assert validator._precision_logged is True


# ---------------------------------------------------------------------------
# 反例清单 / fail-closed 入口
# ---------------------------------------------------------------------------
def test_negative_controls_cover_required_faults():
    faults = {name for name, _ in diag.NEGATIVE_CONTROLS}
    assert {"input", "noise", "swap", "batch_bug", "nonfinite",
            "identity", "hook_missing", "dropfile"} <= faults


def test_main_refuses_non_empty_out_dir(tmp_path):
    (tmp_path / "leftover.txt").write_text("x", encoding="utf-8")
    assert diag.main(["--out-dir", str(tmp_path)]) == 2


def test_main_blocks_when_real_path_unavailable(tmp_path, monkeypatch):
    """真实推理路径导入不了（缺依赖）⇒ BLOCKED + 非零退出，**不得**静默继续。"""
    if _real_path_available():
        pytest.skip("本环境有完整依赖，该分支只在缺依赖时触发")
    assert diag.main(["--out-dir", str(tmp_path / "diag")]) == 2


# ---------------------------------------------------------------------------
# 端到端（需要真实 `_infer_one` / `_infer_batch` ⇒ 只在训练机上跑）
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _real_path_available(),
                    reason="需要完整训练依赖（torchdata / transformers）")
def test_cpu_diag_end_to_end(tmp_path):
    summary = diag.run_all(str(tmp_path / "diag"), selftest=True)
    assert summary["status"] == "PASS", summary["problems"]
    assert summary["groups"]["ok"]["status"] == "PASS"
    assert summary["groups"]["ok"]["production_parity"] is True
    for name, entry in summary["groups"].items():
        if not name.startswith("neg_"):
            continue
        assert entry["state"] == "DETECTED", (name, entry["problems"])
    # 三份证据文件必须存在且非空（缺任一 ⇒ 判定器会 BLOCKED）
    for stem in ("noise", "serial_repeat", "batch_actions"):
        for ext in ("npz", "json"):
            path = tmp_path / "diag" / "ok" / f"{stem}.{ext}"
            assert path.exists() and path.stat().st_size > 0, path
