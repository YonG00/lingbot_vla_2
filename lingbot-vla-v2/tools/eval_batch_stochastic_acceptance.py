#!/usr/bin/env python3
"""Eval Batch **随机性验收**（方案 B）：证明"Batch2 的变化没有明显超过模型自身的随机波动，
且不改变 GMean200 的 PASS 判定"。**默认 PLAN ONLY**。

为什么需要它
------------
`robby_moe.py` 的两处 `tl.atomic_add` 让 fused MoE 每次前向的求和/打包顺序随机 ⇒ 模型自带
~1e-2 的 run-to-run 抖动（实测：同代码同噪声同入参跑两次，max|Δ|=3.1e-2，比 B1↔B2 的 2.1e-2 还大）。
⇒ 原 `atol=1e-5/rtol=1e-3` 的 strict parity 对任何两次运行都不可达。
本工具**不替换也不放宽** strict parity，而是新增一条独立的随机性验收。

两条腿都要过（缺一 ⇒ BLOCKED）
-----------------------------
* **数值层**：固定权重/Chunk，**配对噪声**（B1 与 B2 用同一份、每次调用独立 clone），各跑 ≥5 次；
  比组内波动 vs 跨组波动（max/mean/P95/P99）+ 蒙特卡洛多重比较修正的系统偏差检验 +
  "B2 自身波动不得明显大于 B1"。
* **指标层**：**同样的验证轨迹**在两种模式下各跑 ≥5 次 ⇒ 逐轨迹 MSE → GMean-MSE →
  到 PASS 阈值的距离 → PASS/FAIL 是否翻转/是否贴线。

用法
----
    # ① PLAN ONLY（默认；不碰 GPU、不加载模型、不建目录）
    python tools/eval_batch_stochastic_acceptance.py --ckpt <hf_ckpt> --out-dir <新目录>

    # ② 执行（一次加载；默认 5 次重复，数值层 2 条 chunk + 指标层整条 val 集）
    python tools/eval_batch_stochastic_acceptance.py --ckpt <hf_ckpt> --out-dir <新目录> --execute

不启用 Eval auto、不改 MoE/权重/阈值/正式配置；跑完只给"是否值得提议启用 auto"的结论。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lingbotvla.auto_learning import stochastic_parity as sp  # noqa: E402
from tools import eval_batch_gpu_probe as gpu_probe  # noqa: E402


DEFAULT_CKPT = "/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt"
DEFAULT_SPLIT = "/data/train/task_splits_50"
DEFAULT_TASK = "click_bell"
DEFAULT_DATASET = "/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30"
DEFAULT_THRESHOLDS = "/data/eval_results/open_loop/ref50k/pass_thresholds_gmean200_warn.json"


class _Logger:
    def info_rank0(self, msg, *a, **k):
        print(f"[stoch] {msg}", flush=True)

    def info(self, msg, *a, **k):
        print(f"[stoch] {msg}", flush=True)

    def warning(self, msg, *a, **k):
        print(f"[stoch][WARN] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 阈值表（只读；不改）
# ---------------------------------------------------------------------------
def load_threshold(path: str, task: str) -> Optional[float]:
    """读 GMean-MSE 的 PASS 线（`metric=mse` / `stat=geomean`）。缺失 ⇒ None（调用方 BLOCKED）。"""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if str(doc.get("metric", "mse")) != "mse" or str(doc.get("stat", "geomean")) != "geomean":
        return None
    tasks = doc.get("tasks") or {}
    value = tasks.get(task)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0 else None


# ---------------------------------------------------------------------------
# PLAN ONLY
# ---------------------------------------------------------------------------
def plan(a: argparse.Namespace) -> int:
    ckpt = Path(a.ckpt)
    cli = ckpt.parent.parent.parent / "lingbotvla_cli.yaml"
    threshold = load_threshold(a.thresholds, a.task)
    checks = [
        ("ckpt/config.json", (ckpt / "config.json").exists(), str(ckpt)),
        ("训练 cli yaml", cli.exists(), str(cli)),
        ("val ids", Path(a.val_ids).exists(), a.val_ids),
        ("数据集根", Path(a.dataset).exists(), a.dataset),
        ("GMean200 阈值表（只读）", threshold is not None, f"{a.thresholds} → {a.task}={threshold}"),
        ("输出目录未被占用",
         not (Path(a.out_dir).exists() and any(Path(a.out_dir).iterdir())), a.out_dir),
        ("reps ≥ 下限", int(a.reps) >= sp.MIN_REPS, f"reps={a.reps} (min {sp.MIN_REPS})"),
    ]
    print("=" * 78)
    print("PLAN ONLY（未触碰 GPU / 未加载模型 / 未创建任何目录）")
    print("=" * 78)
    for name, ok, detail in checks:
        print(f"  {'✅' if ok else '❌'} {name}: {detail}")
    print("  判定规则（全部通过才 PASS，否则 BLOCKED）：")
    print(f"    R1 reps >= {sp.MIN_REPS}          R2 全有限值")
    print(f"    R3 cross.p99 <= {sp.CROSS_P99_RATIO_CAP}×within.p99"
          f"   R4 cross.max <= {sp.CROSS_MAX_RATIO_CAP}×within.max")
    print(f"    R5 within(B2).p99 <= {sp.VARIANCE_RATIO_CAP}×within(B1).p99"
          f"   R6 系统偏差 MC p >= {sp.BIAS_P_CAP}")
    print("    M1 GMean 可用   M2 组内判定不翻转   M3 两模式判定一致   M4 离阈值有真余量（非贴线）")
    print("    ⚠️ 指标层没跑 ⇒ 直接 BLOCKED（不允许只用 max_abs_diff 下结论）")
    print("    流程：**先热身一次（不计入统计）** → 再跑 5×2 次数值层 → 再跑 5×2 次指标层")
    print("=" * 78)
    ok = all(flag for _, flag, _ in checks)
    print("[PLAN ONLY] " + ("✅ 路径齐备，可加 --execute 开跑" if ok
                            else "❌ 有阻塞项（见上）；--execute 前先修"))
    return 0 if ok else 2


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
def _infer(validator, items: Sequence[Dict[str, Any]], ft, *, batch: int,
           noise: torch.Tensor) -> Tuple[List[Dict[str, Any]], float]:
    """跑一组：batch=1 逐条 / batch>=2 一次 forward。返回 (预测, 墙钟秒)。"""
    t0 = time.perf_counter()
    with torch.inference_mode():
        if batch == 1:
            # ⚠️ `_infer_core` 返回的是 **list[dict]**（单条时也是长度 1 的 list）⇒ 必须展开，
            #    否则拿到的 preds 是 [[dict]]，`p["actions"]` 会炸
            #    （2026-10-09 GPU 首跑实测：TypeError: list indices must be integers or slices, not str）
            preds = []
            for i, it in enumerate(items):
                preds.extend(validator._infer_core((it,), ft, fresh_visual_grid=False,
                                                   noise=noise[i:i + 1]))
        else:
            preds = list(validator._infer_batch(list(items), ft, noise=noise))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return preds, time.perf_counter() - t0


def _pred_array(pred: Dict[str, Any]) -> np.ndarray:
    """把一条预测里的**所有浮点数组**按 key 排序打平成一个 1-D 数组。

    ⚠️ **不要假设存在 `"actions"` 键**：真实 `ft.unapply()` 返回的是物理量键
    （`action.arm.position` 之类），2026-10-09 GPU 实测 `p["actions"]` ⇒ `KeyError: 'actions'`。
    按 key 排序保证两条路径取出**同一顺序**的元素，逐位/统计比较才有意义。
    """
    parts: List[np.ndarray] = []
    for key in sorted(pred):
        value = pred[key]
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        arr = np.asarray(value)
        if arr.dtype.kind not in "fc":
            continue
        parts.append(arr.astype(np.float64).reshape(-1))
    if not parts:
        raise ValueError("预测里没有任何浮点数组（期望物理量动作）")
    return np.concatenate(parts)

def _numeric_level(validator, items, ft, *, reps: int, noise_base: torch.Tensor,
                   log) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """固定 Chunk 上各跑 reps 次 B1 / B2（配对噪声；每次调用独立 clone）。"""
    # 🔴 必须**先热身**：模型加载后第一次前向包含 cudnn autotune / 惰性初始化，
    #    实测与后续所有腿的 max|Δ| 高达 0.177（而正常同代码波动只有 0.031）⇒ 会计数污染统计。
    _, warm_secs = _infer(validator, items, ft, batch=1, noise=noise_base.clone())
    log(f"  热身一次（不计入统计）：{warm_secs:.3f}s")
    b1_runs, b2_runs = [], []
    t = {"batch1": 0.0, "batch2": 0.0, "warmup": float(warm_secs)}
    for r in range(reps):
        order = ("batch1", "batch2") if r % 2 == 0 else ("batch2", "batch1")
        for mode in order:
            if mode == "batch1":
                preds, secs = _infer(validator, items, ft, batch=1,
                                     noise=noise_base.clone())
                t["batch1"] += secs
                b1_runs.append(np.stack([_pred_array(p) for p in preds]))
            else:
                preds, secs = _infer(validator, items, ft, batch=len(items),
                                     noise=noise_base.clone())
                t["batch2"] += secs
                b2_runs.append(np.stack([_pred_array(p) for p in preds]))
        log(f"  rep {r + 1}/{reps} 完成（顺序 {order}；B1 累计 {t['batch1']:.2f}s / "
            f"B2 累计 {t['batch2']:.2f}s）")
    numeric = sp.decide_numeric(b1_runs=b1_runs, b2_runs=b2_runs, reps=reps)
    throughput = {"batch1_seconds_per_sample": t["batch1"] / (reps * len(items)),
                  "batch2_seconds_per_sample": t["batch2"] / (reps * len(items)),
                  "batch1_total_seconds": t["batch1"], "batch2_total_seconds": t["batch2"],
                  "warmup_seconds": t["warmup"]}
    raw = {"batch1": np.stack(b1_runs), "batch2": np.stack(b2_runs)}
    return numeric, throughput, raw


def _metric_level(validator, ds, ft, ep_map, *, reps: int, noise_all: torch.Tensor,
                  threshold: float, log, max_episodes: Optional[int] = None) -> Dict[str, Any]:
    """同样的验证轨迹，两种模式各跑 reps 次 ⇒ 逐轨迹 MSE → GMean → 与阈值比。"""
    from lingbotvla.auto_learning.decision.gmean import geometric_mse
    from lingbotvla.utils.open_loop_validation import (
        aggregate_chunks, per_episode_starts, pick_action_keys)

    stride = max(1, int(getattr(validator._model_config, "chunk_size", 50) or 50))
    starts = (per_episode_starts(ep_map, stride) if ep_map is not None
              else list(range(0, len(ds), stride)))
    if max_episodes is not None:
        _before = len(starts)
        starts = sp.restrict_starts_to_episodes(starts, ep_map, max_episodes)
        log(f"  指标层收窄：前 {max_episodes} 个回合（chunk 起点 {_before} → {len(starts)}，"
            f"整回合保留以维持 per-trajectory 口径）")
    log(f"  指标层：{len(starts)} 个 chunk 起点（stride={stride}；"
        f"{'生产 per_episode_starts' if ep_map is not None else '退化 flat'}）")

    def eval_once(mode: str) -> Tuple[Optional[float], List[float], List[Any]]:
        chunks: List[tuple] = []
        pos = 0
        bs = 1 if mode == "batch1" else 2
        while pos < len(starts):
            group = starts[pos:pos + bs]
            pos += len(group)
            its = [ds[i] for i in group]
            noise = noise_all[group]          # 同一 rep 内两种模式用**同一份**噪声（配对）
            if bs == 1:
                preds, _ = _infer(validator, its, ft, batch=1, noise=noise)
            else:
                preds, _ = _infer(validator, its, ft, batch=len(its), noise=noise)
            for idx, it, pred in zip(group, its, preds):
                gt = ft.unapply(dict(it))
                keys = pick_action_keys(ft, gt, pred)
                if not keys:
                    return None, [], []
                g = np.asarray(gt[keys[0]], np.float64)
                p = np.asarray(pred[keys[0]], np.float64)
                n = min(len(g), len(p))
                ep_key = (int(ep_map[idx]) if ep_map is not None and idx < len(ep_map)
                          else f"chunk@{idx}")
                chunks.append((ep_key, g[:n], p[:n]))
        res = aggregate_chunks(chunks)
        per_traj = [float(v) for v in res["per_traj_mse"]]
        gmean = geometric_mse(per_traj, expected_count=len(per_traj),
                              ids=list(res["per_traj_ids"]))
        return gmean, per_traj, list(res["per_traj_ids"])

    warm_g, _, _ = eval_once("batch1")      # 热身（不计入统计）
    log(f"  指标层热身一次：GMean={warm_g}")
    b1_g, b2_g, b1_traj, b2_traj = [], [], [], []
    for r in range(reps):
        for mode, acc_g, acc_t in (("batch1", b1_g, b1_traj), ("batch2", b2_g, b2_traj)):
            g, per_traj, ids = eval_once(mode)
            acc_g.append(g)
            acc_t.append(per_traj)
            log(f"  rep {r + 1}/{reps} {mode}: GMean={g}（阈值为 {threshold}）")
    verdict = sp.decide_metric(b1_gmeans=b1_g, b2_gmeans=b2_g, threshold=threshold,
                               b1_per_traj=b1_traj, b2_per_traj=b2_traj)
    return verdict


def execute(a: argparse.Namespace) -> int:
    out = Path(a.out_dir)
    if out.exists() and any(out.iterdir()):
        print(f"[BLOCKED] 输出目录非空，拒绝覆盖: {out}")
        return 2
    if not torch.cuda.is_available():
        print("[BLOCKED] 没有可用 GPU（本入口是 GPU 验收，不做 CPU 替代）")
        return 2
    threshold = load_threshold(a.thresholds, a.task)
    if threshold is None:
        print(f"[BLOCKED] 读不到 {a.task} 的 GMean-MSE 阈值（{a.thresholds}）⇒ 不猜")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    log = _Logger().info_rank0

    a.out_dir = str(out)                      # build_runtime 会往这里写 _open_loop_ids
    validator, vla, ds, ft, ep_map = gpu_probe.build_runtime(a)
    vla.eval()
    torch.cuda.empty_cache()

    cfg = validator._model_config
    shape = (1, int(getattr(cfg, "n_action_steps", 50)), int(getattr(cfg, "max_action_dim", 55)))
    indices = [int(i) for i in a.dataset_indices]
    items = [ds[i] for i in indices]
    gen = validator._noise_generator(validator.device)
    noise_base = torch.cat([torch.randn(shape, generator=gen, device=validator.device,
                                        dtype=torch.float32) for _ in indices], dim=0)
    log(f"数值层样本 {indices}；配对噪声 shape={tuple(noise_base.shape)}；reps={a.reps}")

    numeric, throughput, raw = _numeric_level(validator, items, ft, reps=int(a.reps),
                                              noise_base=noise_base, log=log)
    log(f"数值层：{numeric['status']}  reasons={numeric['reasons']}")
    np.savez_compressed(out / "numeric_runs.npz", **raw)

    metric = None
    if a.metric_level:
        # 指标层用整条 val 集的 chunk 起点；噪声按最大起点数预抽一次（每条 chunk 固定）
        stride = max(1, int(getattr(cfg, "chunk_size", 50) or 50))
        n_starts = len(ds)
        noise_all = torch.cat([torch.randn(shape, generator=gen, device=validator.device,
                                           dtype=torch.float32) for _ in range(n_starts)], dim=0)
        metric = _metric_level(validator, ds, ft, ep_map, reps=int(a.reps),
                               noise_all=noise_all, threshold=threshold, log=log,
                               max_episodes=a.metric_episodes)
        log(f"指标层：{metric['status']}  reasons={metric['reasons']}")

    verdict = sp.overall_verdict(numeric=numeric, metric=metric, throughput=throughput)
    doc = {"entry": "eval_batch_stochastic_acceptance", "task": a.task, "ckpt": a.ckpt,
           "reps": int(a.reps), "threshold": threshold, "numeric": numeric,
           "metric": metric, "throughput": throughput, "verdict": verdict,
           "dataset_indices": indices}
    # 门证据：把"验收条件"与"结论"绑定成一个可被运行时校验的文件
    _shapes, _grids = [], []
    try:
        for _it in items:
            # 与运行时保持一致的**数值相关**形状（不含 lang_*：长度随任务指令变化）
            _shapes.append({k: list(_it[k].shape) for k in
                            ("images", "img_masks", "state")})
            _grids.append(_it.get("image_grid_thw"))
        _notes = {"lang_shapes": {k: list(_it[k].shape) for k in ("lang_tokens", "lang_masks")
                                  if k in _it} for _it in items[:1]}
        _dtype = str(getattr(next(vla.parameters()), "dtype", ""))
    except Exception:  # noqa: BLE001
        _dtype = None
    gate_payload = sp.gate_payload(checkpoint=a.ckpt, task=a.task,
                                   batch_size=len(indices), dtype=_dtype,
                                   shapes=[_shapes[0]] if _shapes else [],
                                   grids=[_grids[0]] if _grids else [],
                                   notes=_notes)
    gate_doc = sp.write_gate(str(out / "gate.json"), payload=gate_payload, verdict=verdict)
    doc["gate"] = {"signature": gate_doc["signature"], "payload": gate_payload,
                   "verdict_status": gate_doc["verdict_status"],
                   "file": str(out / "gate.json")}
    (out / "summary.json").write_text(json.dumps(doc, ensure_ascii=False, sort_keys=True,
                                                 indent=1, default=str), encoding="utf-8")
    print("=" * 78)
    print(f"数值层 = {numeric['status']}  {numeric['reasons']}")
    print(f"  组内 B1: {numeric['within_batch1']}")
    print(f"  组内 B2: {numeric['within_batch2']}")
    print(f"  跨组   : {numeric['cross_batch1_batch2']}")
    print(f"  偏差   : {numeric['bias_pvalue']}")
    if metric is not None:
        print(f"指标层 = {metric['status']}  {metric['reasons']}")
        print(f"  B1 GMean={metric['batch1']['gmeans']} pass={metric['batch1']['pass']}")
        print(f"  B2 GMean={metric['batch2']['gmeans']} pass={metric['batch2']['pass']}")
        print(f"  阈值={threshold}  距阈值余量 B1={metric['batch1']['margin']}")
    print(f"吞吐   : B1 {throughput['batch1_seconds_per_sample']:.4f} s/样本, "
          f"B2 {throughput['batch2_seconds_per_sample']:.4f} s/样本, "
          f"speedup={verdict['speedup_per_sample']}")
    print("-" * 78)
    print(f"总结论 = {verdict['status']}   reasons={verdict['reasons']}")
    print(f"是否值得提议启用 auto = {verdict['worth_proposing_auto']}")
    print(f"门证据（供 auto 使用）= {out / 'gate.json'}  signature={gate_doc['signature'][:16]}…")
    print(f"  → 训练时：{sp.GATE_ENV}={out / 'gate.json'}  +  AL_EVAL_BATCH_MODE=auto"
          f"  +  AL_EVAL_BATCH_APPROVED=1")
    print("=" * 78)
    return 0 if verdict["status"] == "PASS" else 2


# ---------------------------------------------------------------------------
# CPU 端整链自测（--selftest）：用 stub 走完数值层 + 指标层
# ---------------------------------------------------------------------------
class _StubValidator:
    """只实现被用到的接口；返回结构与**生产契约一致**（list[dict]、物理量键）。"""

    def __init__(self, *, jitter: float = 1e-4, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._jitter = jitter
        self.calls: List[str] = []
        self.device = torch.device("cpu")
        self._model_config = types.SimpleNamespace(chunk_size=2, n_action_steps=2,
                                                   max_action_dim=3, action_fp32=False)

    def _one(self, noise_row):
        j = self._rng.normal(0.0, self._jitter, size=(2, 3))
        return {"action.arm.position": (np.asarray(noise_row, np.float64)[:2, :3] + j).astype(np.float32),
                "action.gripper": np.asarray(noise_row, np.float64)[:2, :1].astype(np.float32)}

    def _infer_core(self, items, ft, *, fresh_visual_grid, noise):
        self.calls.append("core")
        return [self._one(noise[i]) for i in range(len(items))]

    def _infer_batch(self, items, ft, *, noise):
        self.calls.append("batch")
        return [self._one(noise[i]) for i in range(len(items))]


class _StubDataset:
    def __init__(self, n: int):
        self._items = [{"images": torch.zeros(1, 3, 4, 4), "img_masks": torch.ones(1),
                        "lang_tokens": torch.tensor([1]), "lang_masks": torch.ones(1),
                        "state": torch.zeros(1), "image_grid_thw": torch.tensor([[1, 4, 4]])}
                       for _ in range(n)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return dict(self._items[i])


class _StubFT:
    def unapply(self, item):
        return {"action.arm.position": np.zeros((2, 3), np.float32),
                "action.gripper": np.zeros((2, 1), np.float32)}


def selftest() -> int:
    """**CPU 端整链自测**：用 stub 走完 `_numeric_level` + `_metric_level` + 总判定。

    为什么必须有（2026-10-09 两次 GPU 试跑的血账）：
      * `_infer_core` 返回 **list[dict]**（单条也是长度 1 的 list）⇒ 当 dict 用就炸；
      * 真实 `ft.unapply()` **不含 `"actions"` 键**（是物理量键）⇒ 硬编码键名就炸。
    两者都能在 CPU 上 1 秒暴露 ⇒ 以后**每次改这条路都必须先跑本自测**，再上 GPU。
    """
    import types as _types
    mod = _types.ModuleType("lingbotvla.utils.open_loop_validation")

    def _per_episode_starts(ep_map, stride):
        return [0, 2]

    def _pick_action_keys(ft, gt, pred):
        return sorted(set(gt) & set(pred))

    def _aggregate_chunks(chunks):
        per, ids, frames = [], [], []
        for ep, gt, pr in chunks:
            per.append(float(np.mean((np.asarray(gt, np.float64) - np.asarray(pr, np.float64)) ** 2)))
            ids.append(ep)
            frames.append(len(gt))
        return {"per_traj_mse": per, "per_traj_ids": ids, "per_traj_frames": frames, "n": len(per)}

    mod.per_episode_starts = _per_episode_starts
    mod.pick_action_keys = _pick_action_keys
    mod.aggregate_chunks = _aggregate_chunks
    saved = sys.modules.get("lingbotvla.utils.open_loop_validation")
    sys.modules["lingbotvla.utils.open_loop_validation"] = mod      # 仅自测进程内替换（绕开 torchdata）

    problems: List[str] = []
    try:
        v, ft, ds = _StubValidator(), _StubFT(), _StubDataset(4)
        items = [ds[0], ds[2]]
        noise_base = torch.zeros(2, 2, 3)
        print("[selftest] 数值层 ……")
        numeric, throughput, raw = _numeric_level(v, items, ft, reps=5, noise_base=noise_base,
                                                  log=lambda m: None)
        assert numeric["status"] in ("PASS", "BLOCKED"), numeric
        for k in ("within_batch1", "within_batch2", "cross_batch1_batch2", "bias_pvalue"):
            if k not in numeric:
                problems.append(f"numeric_missing:{k}")
        if raw["batch1"].shape[0] != 5 or raw["batch2"].shape[0] != 5:
            problems.append("numeric_rep_count_wrong")
        if "warmup_seconds" not in throughput:
            problems.append("throughput_missing_warmup")

        print("[selftest] 指标层 ……")
        metric = _metric_level(v, ds, ft, [7, 7, 8, 8], reps=5,
                               noise_all=torch.zeros(4, 2, 3), threshold=0.01,
                               log=lambda m: None, max_episodes=2)
        for k in ("batch1", "batch2", "threshold"):
            if k not in metric:
                problems.append(f"metric_missing:{k}")
        print("[selftest] 总判定 ……")
        verdict = sp.overall_verdict(numeric=numeric, metric=metric, throughput=throughput)
        print(f"[selftest] numeric={numeric['status']} metric={metric['status']} "
              f"verdict={verdict['status']} reasons={verdict['reasons']}")
        if verdict["status"] not in ("PASS", "BLOCKED"):
            problems.append("verdict_bad_status")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"selftest_exception:{type(exc).__name__}:{exc}")
        import traceback
        traceback.print_exc()
    finally:
        if saved is not None:
            sys.modules["lingbotvla.utils.open_loop_validation"] = saved
        else:
            sys.modules.pop("lingbotvla.utils.open_loop_validation", None)

    if problems:
        print("[selftest] ❌ FAIL: " + "; ".join(problems))
        return 2
    print("[selftest] ✅ 整链结构与契约自测通过（数值层 + 指标层 + 总判定）")
    return 0

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Eval Batch 随机性验收（默认 PLAN ONLY）")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--val-ids", default=f"{DEFAULT_SPLIT}/{DEFAULT_TASK}.val_ids.json")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--dataset-indices", type=int, nargs="+", default=[0, 50])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--metric-episodes", type=int, default=None,
                    help="指标层只跑前 N 个回合（整回合保留；None=全部）。验收提速用，"
                         "建议用 2（= scout 的真实轨迹数）")
    ap.add_argument("--use-length", type=int, default=50)
    ap.add_argument("--use-bf16", action="store_true", default=True)
    ap.add_argument("--no-metric-level", dest="metric_level", action="store_false", default=True,
                    help="跳过指标层（注意：跳过 ⇒ 总判定必然 BLOCKED）")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU 端整链自测（stub 走完数值层+指标层；上 GPU 前必跑）")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.selftest:
        return selftest()
    return execute(a) if a.execute else plan(a)


if __name__ == "__main__":
    sys.exit(main())
