"""Stage B0 —— **real-model integration tests**（默认 skip）。

与 `test_auto_learning_contracts.py` 的分工
------------------------------------------
这里测的是**必须真实模型 / checkpoint 才能验证**的东西：

    R1  open-loop adapter 与现有 evaluator **对拍**（要 `sample_actions`）
    R2  hardness 固定 noise/time **可复现**（要 `forward` 的 flow-matching 路径）
    R3  safe evaluation context 的**恢复审计**（要真实模型上的 use_cache / attention / 网格缓存）
    R4  baseline 与 evaluator 的 **GT 口径一致**（逐 chunk 逐值比对）

开启方式（环境变量齐全才会跑）::

    export AL_TEST_MODEL_PATH=/data/outputs/<run>/checkpoints/global_step_N/hf_ckpt
    export AL_TEST_CONFIG=/data/outputs/<run>/lingbotvla_cli.yaml
    export AL_TEST_MANIFEST=/data/train/task_splits/manifest.json
    export AL_TEST_TASK=click_bell            # 可选，默认 click_bell
    export QWEN3VL_PATH=/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct
    python -m pytest tests/test_auto_learning_real_model.py -q -s
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ⚠️ 受限沙箱里 pytest 的 `tmp_path` 可能 mkdir 失败 ⇒ 用**仓库内**的临时目录
_TMP_ROOT = REPO / ".pytest_tmp"


@pytest.fixture
def tmp_path():
    import shutil
    import uuid

    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    d = _TMP_ROOT / f"alrm_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


torch = pytest.importorskip("torch", reason="real-model 集成测试需要 torch")

MODEL_PATH = os.environ.get("AL_TEST_MODEL_PATH")
CONFIG_PATH = os.environ.get("AL_TEST_CONFIG")
MANIFEST_PATH = os.environ.get("AL_TEST_MANIFEST")
TASK = os.environ.get("AL_TEST_TASK", "click_bell")

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH and CONFIG_PATH and MANIFEST_PATH),
    reason="需要 AL_TEST_MODEL_PATH / AL_TEST_CONFIG / AL_TEST_MANIFEST",
)


class _Logger:
    def info_rank0(self, msg, *a, **k):
        print(f"[test][open_loop] {msg}", flush=True)

    def info(self, msg, *a, **k):
        print(f"[test][open_loop] {msg}", flush=True)

    def warning(self, msg, *a, **k):
        print(f"[test][open_loop][WARN] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# session 级素材
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def built():
    """建 validator（真实权重）+ catalog + 临时输出目录（**FP32 模式**）。"""
    return _build(use_bf16=False)


@pytest.fixture(scope="module")
def built_bf16():
    """同上，但 **BF16 模式** —— 与单卡训练配置一致（review v0.1 #4）。

    只有需要验证 dtype 语义的用例才用它（多建一份模型很贵）。
    """
    return _build(use_bf16=True)


def _build(use_bf16: bool):
    import yaml

    from lingbotvla.auto_learning.catalog import catalog_from_task_split
    from lingbotvla.utils.open_loop_validation import OpenLoopValidator

    ckpt = Path(MODEL_PATH).resolve()
    if not (ckpt / "config.json").exists() and (ckpt / "hf_ckpt" / "config.json").exists():
        ckpt = ckpt / "hf_ckpt"
    if not (ckpt / "config.json").exists():
        pytest.skip(f"{ckpt} 里没有 config.json（不是 hf_ckpt 目录）")

    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

    server = LingbotVLAv2Server(
        path_to_pi_model=str(ckpt), robot_norm_path=None, use_length=50,
        chunk_ret=True, use_bf16=use_bf16, use_fp32=not use_bf16, use_compile=False,
    )
    vla, processor = server.vla, server.processor
    hf_cfg = vla.config

    raw = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    ns = types.SimpleNamespace(**dict(raw.get("data", {})))
    if not hasattr(ns, "chunk_size"):
        ns.chunk_size = int(hf_cfg.chunk_size)
    if not hasattr(ns, "num_episode"):
        ns.num_episode = None
    align = (raw.get("train", {}) or {}).get("align_params") or {}

    class _NS:
        pass

    args = _NS()
    args.data = ns
    args.model = hf_cfg
    args.train = _NS()
    suffix = "bf16" if use_bf16 else "fp32"
    args.train.output_dir = str(Path(MANIFEST_PATH).resolve().parent / f"_al_test_ids_{suffix}")
    args.train.global_rank = 0
    args.train.use_bf16 = use_bf16
    os.makedirs(args.train.output_dir, exist_ok=True)

    cat = catalog_from_task_split(str(MANIFEST_PATH), strict=True)
    entry = cat.entry(TASK)

    validator = OpenLoopValidator(
        model=vla, args=args, processor=processor, use_depth_align=bool(align),
        writer=None, logger=_Logger(),
        train_monitor_ids=entry.train_traj_ids[:5], val_ids=entry.val_traj_ids,
    )
    return types.SimpleNamespace(
        vla=vla, processor=processor, hf_cfg=hf_cfg, args=args,
        cat=cat, entry=entry, validator=validator,
        out_dir=Path(args.train.output_dir), use_bf16=use_bf16,
    )


# --------------------------------------------------------------------------- #
# R1  open-loop adapter 与现有 evaluator 对拍
# --------------------------------------------------------------------------- #
def test_R1_adapter_matches_evaluate_ids_exactly(built):
    """**同一代码路径**对拍：adapter 的 mse 必须与直接 `evaluate_ids()` 逐值相等。

    ⚠️ 这是 R1 的正确形式。GPU 实测（2026-10-06）发现：拿 `validate()` 的 `val` 指标来比
    **不会**逐值相等，原因是 `_run()` 里 `train_monitor` 先跑、**消耗了噪声流**，
    `val` 拿到的是后续噪声；而单独调 `evaluate_ids()` 会 `self._noise_gen = None`
    **重置到噪声序列起点**。两者是**不同的噪声样本**，MSE 自然不同（实测 0.01086 vs 0.01006）。
    """
    from lingbotvla.auto_learning.evaluator import EvaluatorAdapter

    v = built.validator
    adapter = EvaluatorAdapter(v, built.cat, None, logger=_Logger())

    ids = list(built.entry.val_traj_ids)
    ref = v.evaluate_ids(ids, "r1_ref_val")
    got = adapter.evaluate_task(TASK, "val")

    assert got.episode_ids == ids
    # ⚠️ GPU 实测（2026-10-06, RTX 4090）：**同一路径跑两次也不逐位一致** ——
    #    实测相对差最大 ~5e-3（bf16 量级；CUDA 归约顺序 + 低精度 op 不保证可复现）。
    #    B1 测试计划 §G2 明确写「允许合理数值容差」⇒ 这里用 2% 的宽松容差，
    #    真正要守的是「轨迹数 / chunk 数 / action keys 完全一致」。
    assert got.mse == pytest.approx(ref["mse"], rel=2e-2, abs=1e-6), (
        f"adapter mse={got.mse} vs evaluate_ids mse={ref['mse']}")
    assert got.per_traj_mse == pytest.approx(ref["per_traj_mse"], rel=2e-2, abs=1e-6)
    assert got.n_traj == ref["n"] and got.n_chunks == ref["n_chunks"]
    assert got.action_keys == ref["action_keys"]


def test_R1b_validate_val_is_close_but_not_bitwise(built):
    """`validate()` 的 `val` 与独立评测**近似但不逐位相等** —— 记录这个已知差异。

    噪声流位置不同（见 R1 的说明）。这里只断言「同一量级」，避免把真实差异当成 bug 藏起来。
    """
    from lingbotvla.auto_learning.evaluator import EvaluatorAdapter

    v = built.validator
    adapter = EvaluatorAdapter(v, built.cat, None, logger=_Logger())

    ref = v.validate(0)
    assert ref is not None, "validate() 返回 None（rank0 判断失败）"
    got = adapter.evaluate_task(TASK, "val")

    assert got.n_traj == ref["val"]["n"], "轨迹数必须一致（只有噪声位置不同）"
    rel = abs(got.mse - ref["val"]["mse"]) / max(ref["val"]["mse"], 1e-12)
    assert rel < 0.25, (
        f"两次评测差了 {rel:.1%}（adapter={got.mse} validate={ref['val']['mse']}）——"
        "超过噪声位置能解释的范围，疑似真差异")


# --------------------------------------------------------------------------- #
# R2  hardness 固定 noise / time 可复现
# --------------------------------------------------------------------------- #
def test_R2_hardness_is_deterministic(built):
    from lingbotvla.auto_learning.hardness import HardnessScorer

    v = built.validator
    ds_path = v._episode_ids_file(built.entry.train_traj_ids[:2], "al_test_hardness")
    ds = v._dataset(ds_path)
    ft = v._ft_for(ds_path)
    ep_map = __import__("lingbotvla.utils.open_loop_validation",
                        fromlist=["_episode_index_map"])._episode_index_map(ds)
    from lingbotvla.utils.open_loop_validation import per_episode_starts
    stride = max(1, int(getattr(built.hf_cfg, "chunk_size", 50) or 50))
    starts = per_episode_starts(ep_map, stride)[:2]
    items = [ds[i] for i in starts]

    scorer = HardnessScorer(built.vla, device="cuda", logger=_Logger())
    a = scorer.score(items)
    b = scorer.score(items)
    assert a.shape == (len(items),)
    # ⚠️ 不能用 array_equal：GPU 实测两次差最大 ~5e-3（bf16 量级）。
    #    B1 测试计划 §G2 允许「合理数值容差」。
    np.testing.assert_allclose(a, b, rtol=2e-2, atol=1e-6)
    # 🔑 **真正影响采样的**是「难度排序」，不是 raw loss 的第 4 位小数：
    #    采样用的是 percentile rank ⇒ 只要排序稳定，采样分布就稳定。
    rank_a = np.argsort(np.argsort(a))
    rank_b = np.argsort(np.argsort(b))
    assert np.array_equal(rank_a, rank_b), (
        f"两次打分的**排序**不一致（{a} vs {b}）—— 这会真的改变采样分布")

    # 换 seed ⇒ 值必须变（证明 noise 真的起作用了）
    c = HardnessScorer(built.vla, seed=999, device="cuda", logger=_Logger()).score(items)
    assert not np.array_equal(a, c)

    # 换 flow time ⇒ 值必须变
    d = HardnessScorer(built.vla, flow_time=0.9, device="cuda", logger=_Logger()).score(items)
    assert not np.array_equal(a, d)


# --------------------------------------------------------------------------- #
# R3  safe evaluation context 的恢复审计
# --------------------------------------------------------------------------- #
def test_R3_evaluate_ids_restores_everything(built):
    """评测前后：use_cache / attention / 网格缓存 / 模块 training 标志 / 三套 RNG 全一致。"""
    import random

    v = built.validator
    m = built.vla
    from lingbotvla.utils.open_loop_validation import _use_cache_owners

    before = {
        "use_cache": [getattr(c, "use_cache", None) for c in _use_cache_owners(m)],
        "attn_impl": [getattr(getattr(mm, "config", None), "attention_implementation", None)
                      for mm in m.modules()],
        "training": [mm.training for mm in m.modules()],
        "torch_rng": torch.get_rng_state().clone(),
        "numpy_rng": np.random.get_state()[1].copy(),
        "py_rng": random.getstate(),
    }

    v.evaluate_ids(built.entry.val_traj_ids[:2], "al_test_restore")

    assert [getattr(c, "use_cache", None) for c in _use_cache_owners(m)] == before["use_cache"]
    assert [getattr(getattr(mm, "config", None), "attention_implementation", None)
            for mm in m.modules()] == before["attn_impl"]
    assert [mm.training for mm in m.modules()] == before["training"]
    assert torch.equal(torch.get_rng_state(), before["torch_rng"])
    assert np.array_equal(np.random.get_state()[1], before["numpy_rng"])
    assert random.getstate() == before["py_rng"]


def test_R3b_evaluate_ids_refuses_to_bypass_context(built):
    """`evaluate_ids` 必须走 safe context —— 用「评测期间」探针证明 use_cache 被临时打开。"""
    v = built.validator
    seen = {}
    orig = v._evaluate_ids

    def _probe(ids, tag):
        seen["use_cache"] = [getattr(c, "use_cache", None)
                             for c in __import__(
                                 "lingbotvla.utils.open_loop_validation",
                                 fromlist=["_use_cache_owners"])._use_cache_owners(built.vla)]
        return orig(ids, tag)

    v._evaluate_ids = _probe
    try:
        v.evaluate_ids(built.entry.val_traj_ids[:2], "al_test_ctx")
    finally:
        v._evaluate_ids = orig
    assert seen and all(x is True for x in seen["use_cache"]), (
        "评测期间 use_cache 没有被临时打开 ⇒ 可能绕过了 safe_eval_context")


# --------------------------------------------------------------------------- #
# R4  baseline 与 evaluator 的 GT 口径一致（逐 chunk 逐值）
# --------------------------------------------------------------------------- #
def test_R4_baseline_gt_matches_evaluator_dump(built, tmp_path):
    """`collect_gt_chunks` 的 GT 必须与 evaluator dump 出来的 `_gt.npy` **逐值相同**。"""
    from lingbotvla.utils.open_loop_validation import per_episode_starts

    v = built.validator
    ids = built.entry.val_traj_ids[:2]
    tag = "al_test_gtcmp"
    dump_dir = str(tmp_path / "dump")

    # ① evaluator 路径：dump GT
    v.dump_dir = dump_dir
    try:
        v.evaluate_ids(ids, tag)
    finally:
        v.dump_dir = None

    # ② baseline 路径：只收 GT
    chunks, keys = v.collect_gt_chunks(ids, tag + "_base")
    assert chunks, "collect_gt_chunks 返回空"
    assert keys, "action_keys 为空"

    # ③ 逐 chunk 对拍：按 evaluator 的 local_idx 找到对应 dump
    ds_path = v._episode_ids_file(ids, tag)
    ds = v._dataset(ds_path)
    from lingbotvla.utils.open_loop_validation import _episode_index_map
    ep_map = _episode_index_map(ds)
    stride = max(1, int(getattr(built.hf_cfg, "chunk_size", 50) or 50))
    starts = per_episode_starts(ep_map, stride)

    by_ep = {}
    for k, gt in chunks:
        by_ep.setdefault(k, []).append(gt)

    n_checked = 0
    for local_idx in starts:
        p = os.path.join(dump_dir, f"{tag}_ep{local_idx}_gt.npy")
        if not os.path.exists(p):
            continue
        dumped = np.load(p)
        ep_key = int(ep_map[local_idx])
        cand = by_ep.get(ep_key, [])
        assert cand, f"baseline 没有 episode {ep_key} 的 chunk"
        # 同一个 episode 可能有多个 chunk ⇒ 取形状匹配且逐值相同的那个
        match = [g for g in cand if g.shape == dumped.shape and np.allclose(g, dumped, atol=0)]
        assert match, (
            f"ep{ep_key} local_idx={local_idx}: baseline GT 与 evaluator dump 不一致"
            f"（形状 baseline={[g.shape for g in cand]} vs dump={dumped.shape}）")
        n_checked += 1
    assert n_checked > 0, "没有找到任何可对拍的 dump 文件"


def test_R4b_baseline_dimension_matches_evaluator_action_space(built):
    """baseline 的 action_keys / 维度必须与 evaluator 实际使用的完全一致。"""
    from lingbotvla.utils.open_loop_validation import pick_action_keys

    v = built.validator
    ds_path = v._episode_ids_file(built.entry.val_traj_ids[:2], "al_test_keys")
    ds = v._dataset(ds_path)
    ft = v._ft_for(ds_path)
    item = ds[0]
    gt_phys = ft.unapply(dict(item))
    keys = pick_action_keys(ft, gt_phys, gt_phys)

    chunks, keys2 = v.collect_gt_chunks(built.entry.val_traj_ids[:2], "al_test_keys2")
    assert keys2 == keys
    assert chunks[0][1].shape[1] == sum(
        np.asarray(gt_phys[k]).reshape(np.asarray(gt_phys[k]).shape[0], -1).shape[1]
        for k in keys)


# --------------------------------------------------------------------------- #
# R5  HardnessScorer 的 dtype / device（review v0.1 #4）
# --------------------------------------------------------------------------- #
def _items_for(built, tag, n=2):
    from lingbotvla.utils.open_loop_validation import _episode_index_map, per_episode_starts

    v = built.validator
    ds_path = v._episode_ids_file(built.entry.train_traj_ids[:2], tag)
    ds = v._dataset(ds_path)
    ep_map = _episode_index_map(ds)
    stride = max(1, int(getattr(built.hf_cfg, "chunk_size", 50) or 50))
    starts = per_episode_starts(ep_map, stride)[:n]
    return [ds[i] for i in starts]


def test_R5_hardness_dtype_and_device_match_forward(built):
    """noise/time 的 dtype 必须等于**真实 forward 会用的** dtype；device 从模型推断。"""
    import torch

    from lingbotvla.auto_learning.hardness import HardnessScorer

    sc = HardnessScorer(built.vla, logger=_Logger())
    dt = sc.forward_dtype()
    if getattr(built.hf_cfg, "action_fp32", False):
        assert dt == torch.float32
    else:
        assert dt == next(built.vla.parameters()).dtype
    assert str(next(built.vla.parameters()).device) == sc.device
    assert not str(sc.device).startswith("cuda") or torch.cuda.is_available()

    out = sc.score(_items_for(built, "al_test_dtype"))
    assert out.shape == (2,)
    assert np.isfinite(out).all()


def test_R5b_hardness_bf16_training_mode(built_bf16):
    """与单卡训练一致的 **BF16** 模式。

    旧实现用 `actions.dtype`（dataset 里是 FP32）造 noise/time ⇒ 与 forward 实际使用的
    dtype 不一致。这条用例专门覆盖它。
    """
    import torch

    from lingbotvla.auto_learning.hardness import HardnessScorer

    sc = HardnessScorer(built_bf16.vla, logger=_Logger())
    dt = sc.forward_dtype()
    if not getattr(built_bf16.hf_cfg, "action_fp32", False):
        assert dt == torch.bfloat16, f"BF16 模式下应为 bfloat16，实际 {dt}"
    out = sc.score(_items_for(built_bf16, "al_test_bf16"))
    assert out.shape == (2,)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# R6  2 → 4 轨迹的真实回归（review v0.1 #1）
# --------------------------------------------------------------------------- #
def test_R6_two_then_four_trajectories_are_really_different(built):
    """先评 2 条 scout、再评 4 条 confirm ⇒ 第二次必须真的得到 **4 条轨迹**。

    这是 review v0.1 #1 的端到端回归：tag 若不带 ids 指纹，`_dataset()` 会按
    同一个 episode_ids 文件路径返回**第一次缓存的 2 条 dataset**。
    """
    from lingbotvla.auto_learning.evaluator import EvaluatorAdapter

    ad = EvaluatorAdapter(built.validator, built.cat, None, logger=_Logger())
    val = built.entry.ids_for("val")
    r2 = ad.evaluate_task(TASK, "val", episode_ids=val[:2])
    r4 = ad.evaluate_task(TASK, "val", episode_ids=val[:4])

    assert r2.n_traj == 2, f"scout 应为 2 条轨迹，实际 {r2.n_traj}"
    assert r4.n_traj == 4, (
        f"confirm 应为 4 条轨迹，实际 {r4.n_traj} ⇒ 命中了旧 dataset 缓存（tag 没带 ids 指纹）")
    assert len(r4.per_traj_ids) == 4 and len(set(r4.per_traj_ids)) == 4
    assert set(r2.per_traj_ids) <= set(r4.per_traj_ids)
