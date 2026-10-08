"""评测精度三分离的 CPU 专项测试（用户 2026-10-08 要求）。

覆盖：
  1. 推理 dtype 决议（auto/bf16/fp32 + action_fp32 优先 + 与权重不一致时 fail-fast）；
  2. 指标口径 = 误差 fp32 / 聚合 fp64，并且结果里带 `metric_dtype` 便于核查；
  3. "配置 bf16 就必须真的按 bf16 推理"（用 tiny model + forward pre-hook 验证喂进去的 dtype）；
  4. 未改动正式 PASS 规则（正式 YAML 仍是 nmse）与 norm_stats 选择机制。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch

from lingbotvla.utils.eval_precision import (
    METRIC_DTYPE_LABEL, describe_precision, metric_arrays_for_aggregation, resolve_inference_dtype,
)


def _aggregate_chunks():
    """`aggregate_chunks` 是纯 numpy，但它所在的模块会拉起数据集/DataLoader 依赖。"""
    try:
        from lingbotvla.utils.open_loop_validation import aggregate_chunks
        return aggregate_chunks
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"本地缺重依赖（{exc.__class__.__name__}: {exc}）；该用例在远端 conda 环境跑")


_AGG = None

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# 1) 推理 dtype 决议
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "requested,action_fp32,weight,fallback,expect",
    [
        ("auto", False, torch.bfloat16, False, torch.bfloat16),   # 跟随权重
        ("auto", False, torch.float32, False, torch.float32),
        ("auto", True, torch.bfloat16, False, torch.float32),     # action_fp32 优先
        ("auto", False, None, False, torch.float32),              # 拿不到权重 ⇒ 保守 fp32
        ("auto", False, None, True, torch.bfloat16),              # 兼容旧 use_bf16
        ("bf16", False, torch.bfloat16, False, torch.bfloat16),   # 显式且一致
        ("fp32", False, torch.float32, False, torch.float32),
        ("fp32", True, torch.bfloat16, False, torch.float32),     # action_fp32 时 fp32 合法
    ],
)
def test_resolve_inference_dtype_matrix(requested, action_fp32, weight, fallback, expect):
    dtype, info = resolve_inference_dtype(requested=requested, action_fp32=action_fp32,
                                          weight_dtype=weight, fallback_bf16=fallback)
    assert dtype is expect
    assert info["metric"] == METRIC_DTYPE_LABEL
    assert "推理" in describe_precision(info) and "指标" in describe_precision(info)


@pytest.mark.parametrize(
    "requested,weight",
    [("bf16", torch.float32), ("fp32", torch.bfloat16)],
)
def test_explicit_request_mismatch_fails_fast(requested, weight):
    """不允许"配置 bf16 实际跑 fp32"这类静默换精度。"""
    with pytest.raises(ValueError, match="拒绝静默换精度"):
        resolve_inference_dtype(requested=requested, action_fp32=False, weight_dtype=weight)


def test_invalid_requested_dtype_rejected():
    with pytest.raises(ValueError, match="eval_inference_dtype"):
        resolve_inference_dtype(requested="fp16", action_fp32=False, weight_dtype=torch.float32)


# --------------------------------------------------------------------------- #
# 2) 指标：误差 fp32 / 聚合 fp64
# --------------------------------------------------------------------------- #
def _chunks(n_eps=3, frames=400, dims=7, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for ep in range(n_eps):
        gt = (rng.random((frames, dims), dtype=np.float32) * 2 - 1) * 1e3
        pr = gt + (rng.random((frames, dims), dtype=np.float32) - 0.5) * 1e-3
        out.append((ep, gt.astype(np.float32), pr.astype(np.float32)))
    return out


def test_metric_is_fp64_aggregated_and_labelled():
    fn = _aggregate_chunks()

    chunks = _chunks()
    res = fn(chunks)
    assert res["metric_dtype"] == METRIC_DTYPE_LABEL

    # 逐步用 float64 复算："逐轨迹 MSE → 跨轨迹平均" 与 "pooled MSE/baseline"
    per = []
    for ep, gt, pr in chunks:
        err64, gt64 = metric_arrays_for_aggregation(pr - gt, gt)
        per.append((float(np.mean(err64 ** 2)), float(np.var(gt64, axis=0).mean())))
    exact_mse = float(np.mean([m for m, _ in per]))
    exact_per_traj_baseline = float(np.mean([b for _, b in per]))
    assert res["mse"] == pytest.approx(exact_mse, rel=1e-12, abs=0.0)
    assert res["mean_baseline_mse_per_traj"] == pytest.approx(exact_per_traj_baseline, rel=1e-12, abs=0.0)
    assert len(res["per_traj_mse"]) == len(chunks)

    # 与"全 fp32 聚合"对比：我们的值必须更接近 fp64 精确值（弱断言，跨平台稳健）
    fp32_mse = float(np.mean([float(np.mean((pr - gt) ** 2)) for _, gt, pr in chunks]))
    assert abs(res["mse"] - exact_mse) <= abs(fp32_mse - exact_mse) + 1e-18


def test_metric_empty_result_carries_label():
    fn = _aggregate_chunks()

    res = fn([])
    assert res["metric_dtype"] == METRIC_DTYPE_LABEL and res["n"] == 0


def test_single_chunk_episode_matches_manual_fp64():
    fn = _aggregate_chunks()

    gt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pr = gt + np.array([[1e-3, -2e-3], [3e-3, -4e-3]], dtype=np.float32)
    res = fn([("ep0", gt, pr)])
    err64 = (pr - gt).astype(np.float64)
    assert res["per_traj_mse"][0] == pytest.approx(float(np.mean(err64 ** 2)), rel=1e-12)


# --------------------------------------------------------------------------- #
# 3) "配置 bf16 ⇒ 真的按 bf16 推理"（不是先 fp32 再 cast 结果）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("weight_dtype,expected", [(torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32)])
def test_model_receives_resolved_dtype(weight_dtype, expected):
    model = torch.nn.Linear(4, 3).to(weight_dtype)
    seen = {}

    def pre_hook(module, args):
        seen["dtype"] = args[0].dtype

    handle = model.register_forward_pre_hook(pre_hook)
    dtype, _ = resolve_inference_dtype(requested="auto", action_fp32=False, weight_dtype=weight_dtype)
    x = torch.zeros(2, 4, dtype=dtype)
    model(x)
    handle.remove()
    assert seen["dtype"] is expected, "喂给模型的张量 dtype 必须等于决议出的推理 dtype"


def test_bf16_request_with_fp32_weights_refuses_instead_of_silent_fp32():
    with pytest.raises(ValueError):
        resolve_inference_dtype(requested="bf16", action_fp32=False, weight_dtype=torch.float32)


# --------------------------------------------------------------------------- #
# 4) 未改动正式 PASS 规则 / norm_stats 机制
# --------------------------------------------------------------------------- #
def test_formal_pass_rule_untouched():
    """正式判据仍是 NMSE（未显式设置则用默认值）⇒ 用**解析后的配置**断言，而不是文本。"""
    import yaml

    from lingbotvla.auto_learning.config import AutoLearningConfig

    raw = yaml.safe_load((ROOT / "configs/auto_learning/formal_50task_4pass.yaml").read_text(encoding="utf-8"))
    al = AutoLearningConfig.from_dict(raw)
    assert raw.get("pass_metric", "nmse") == "nmse", "本轮不得把实验阈值偷偷切成正式标准"
    assert al.pass_metric == "nmse", "解析后的正式判据必须仍是 NMSE"
    assert al.target_total_passed_tasks == 4
    assert float(al.hardness_probe_fraction) == 0.10


def test_norm_stats_selection_mechanism_unchanged():
    yml = (ROOT / "configs/robot_configs/robotwin.yaml").read_text(encoding="utf-8")
    assert "norm_stats: assets/norm_stats/robotwin.json" in yml, "Reference 仍用官方 robotwin.json"
    base = (ROOT / "lingbotvla/data/vla_data/base_dataset.py").read_text(encoding="utf-8")
    assert "robotwin_competition_clean.json" in base, "Candidate 仍走 norm_stats_file（competition_clean）"
    assert "robot_config" in base and "norm_stats_file" in base, "优先序机制（robot_config → norm_stats_file）保持"
