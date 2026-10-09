"""**随机性验收**（stochastic parity）：在"模型自身非确定"前提下判定批处理是否安全。

背景（2026-10-09 定案）
----------------------
`lingbotvla/ops/robby_moe.py` 的两处 `tl.atomic_add`（token 打包槽位 + 专家输出累加）让
fused MoE 每次前向的求和/打包顺序随线程调度变化 ⇒ 模型自带 ~1e-2 量级的 run-to-run 抖动。
实测：**同一段代码、同一份噪声、同一份入参**跑两次，输出 max|Δ| = 3.1e-2；而 B1 vs B2 只有 2.1e-2。
⇒ 原来 `atol=1e-5 / rtol=1e-3` 的 strict parity **对任何两次运行都不可达**（含 serial vs serial）。

因此新增**独立的**随机性验收（**不替换、不放宽**原 strict parity）：

* 固定权重 / 固定 Chunk / **配对噪声**（同一 rep 的 B1 与 B2 用同一份噪声，且每次调用独立 clone）；
* 各跑 `reps` 次（**下限 5**），分别估计组内波动与跨组波动；
* 报告 max / mean / P95 / P99，并检验**系统性偏差**（均值漂移是否超出组内噪声的标准误）；
* 指标层：同样的验证轨迹在两种模式下的逐轨迹 MSE → **GMean-MSE** → 与 PASS 阈值距离 → PASS/FAIL 稳定性；
* **fail-closed**：样本量不足 / 统计结论不明确 / 指标判定翻转 / 出现非有限值 ⇒ **BLOCKED**。

判定规则全部显式写在 :func:`decide_numeric` / :func:`decide_metric` 里，且**只有全部通过才 PASS**。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


#: 重复次数下限（用户要求：Batch1 / Batch2 各至少 5 次）。
MIN_REPS = 5

#: 跨组 P99 不得比组内 P99 大超过该倍数（超过 ⇒ 批处理引入了额外波动）。
CROSS_P99_RATIO_CAP = 1.5

#: 跨组 max 不得比组内 max 大超过该倍数。
CROSS_MAX_RATIO_CAP = 1.5

#: 逐元素 t 统计量的"单点"阈值（仅用于构造统计量；**不能**直接拿 max 判，见下）。
BIAS_Z_CAP = 3.0

#: 蒙特卡洛零分布的重数（多重比较修正：几千个元素上取 max 必然会超阈）。
BIAS_MC_REPS = 200

#: 偏差检验的 p 值门槛（低于它才判"存在系统性偏差"）。
BIAS_P_CAP = 0.01

#: Batch2 自身波动不得超过 Batch1 自身波动的该倍数（超过 ⇒ 批处理放大了抖动）。
VARIANCE_RATIO_CAP = 2.0

#: 浮点噪声地板：小于它的差异视为 0（避免 bf16 上除零/无意义比值）。
ABS_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# 基础统计
# ---------------------------------------------------------------------------
def _abs_diff(a, b) -> np.ndarray:
    return np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))


def diff_stats(a, b) -> Dict[str, float]:
    """两个同形数组的逐元素绝对差统计（max / mean / p95 / p99）。"""
    d = _abs_diff(a, b).reshape(-1)
    if d.size == 0:
        raise ValueError("empty arrays")
    return {"max": float(d.max()), "mean": float(d.mean()),
            "p95": float(np.percentile(d, 95)), "p99": float(np.percentile(d, 99)),
            "n_elements": int(d.size)}


def _pool_stats(pairs: Sequence[Tuple[Any, Any]]) -> Dict[str, float]:
    if not pairs:
        raise ValueError("no pairs to pool")
    d = np.concatenate([_abs_diff(a, b).reshape(-1) for a, b in pairs])
    return {"max": float(d.max()), "mean": float(d.mean()),
            "p95": float(np.percentile(d, 95)), "p99": float(np.percentile(d, 99)),
            "n_elements": int(d.size), "n_pairs": len(pairs)}


def within_group_stats(runs: Sequence[Any]) -> Dict[str, float]:
    """组内波动：**所有无序对**的逐元素绝对差（N 小，取全对更稳）。"""
    pairs = [(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
    return _pool_stats(pairs)


def cross_group_stats(runs_a: Sequence[Any], runs_b: Sequence[Any]) -> Dict[str, float]:
    """跨组波动：**按 rep 配对**（同一 rep 的 A/B 用同一份噪声 ⇒ 配对消噪）。"""
    if len(runs_a) != len(runs_b):
        raise ValueError("paired comparison requires equal rep counts")
    return _pool_stats([(runs_a[i], runs_b[i]) for i in range(len(runs_a))])


def systematic_bias(runs_a: Sequence[Any], runs_b: Sequence[Any]) -> Dict[str, float]:
    """系统性偏差：逐元素"配对差值"的均值相对于其标准误有多大。

    纯随机抖动 ⇒ 均值漂移随 1/sqrt(N) 收缩；系统性偏差 ⇒ 漂移不随 N 收缩。
    返回 ``max_abs_shift``（逐元素漂移的绝对值最大值）、``se_p99``（组内噪声的 P99/sqrt(N)）、
    ``z_max``（漂移/标准误的最大值）、``frac_z_gt_cap``（超过 cap 的元素占比）。
    """
    if len(runs_a) != len(runs_b) or len(runs_a) < 2:
        raise ValueError("bias test needs paired runs with reps >= 2")
    deltas = np.stack([np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
                       for a, b in zip(runs_a, runs_b)], axis=0)
    shift = deltas.mean(axis=0)
    sd = deltas.std(axis=0, ddof=1)
    n = len(runs_a)
    se = sd / math.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > ABS_FLOOR, np.abs(shift) / se, 0.0)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0)
    return {"max_abs_shift": float(np.abs(shift).max()),
            "mean_abs_shift": float(np.abs(shift).mean()),
            "z_max": float(z.max()),
            "frac_z_gt_cap": float((z > BIAS_Z_CAP).mean()),
            "n_elements": int(shift.size)}


# ---------------------------------------------------------------------------
# 判定（数值层）
# ---------------------------------------------------------------------------
def bias_pvalues(runs_a: Sequence[Any], runs_b: Sequence[Any], *,
                 n_mc: int = BIAS_MC_REPS, seed: int = 20261009) -> Dict[str, float]:
    """系统偏差的**多重比较修正**检验（蒙特卡洛零分布）。

    为什么不能直接看 ``max|t|``：几千个元素上取最大，纯噪声也会给出 4~6 的 |t|（实测 5.9）。
    做法：把观测到的 ``frac(|t|>cap)`` 与 ``max|t|`` 与"同样形状的纯正态零数据"跑出的分布比，
    得到 p 值；p 很小才判"存在系统性偏差"。
    """
    deltas = np.stack([np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
                       for a, b in zip(runs_a, runs_b)], axis=0)
    n = deltas.shape[0]
    shape = deltas.shape[1:]

    def _stats(sample: np.ndarray) -> Tuple[float, float]:
        shift = sample.mean(axis=0)
        sd = sample.std(axis=0, ddof=1)
        se = sd / math.sqrt(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > ABS_FLOOR, np.abs(shift) / se, 0.0)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0)
        return float((z > BIAS_Z_CAP).mean()), float(z.max())

    obs_frac, obs_max = _stats(deltas)
    rng = np.random.default_rng(seed)
    ge_frac = ge_max = 0
    for _ in range(int(n_mc)):
        null = rng.standard_normal((n, *shape))
        f, m = _stats(null)
        ge_frac += int(f >= obs_frac)
        ge_max += int(m >= obs_max)
    return {"obs_frac_gt_cap": obs_frac, "obs_z_max": obs_max,
            "p_frac": ge_frac / float(n_mc), "p_max": ge_max / float(n_mc),
            "n_mc": int(n_mc)}


def decide_numeric(*, b1_runs: Sequence[Any], b2_runs: Sequence[Any],
                   reps: Optional[int] = None) -> Dict[str, Any]:
    """数值层判定：Batch2 的波动是否**没有明显超过**模型自身的波动，且无系统性偏差。

    规则（**全部通过才 PASS**，否则 BLOCKED 并给出原因码）：
      R1 ``reps >= MIN_REPS``：样本量不足 ⇒ 统计结论不可靠；
      R2 全部有限值；
      R3 ``cross.p99 <= max(cap*within.p99, ABS_FLOOR)``；
      R4 ``cross.max <= max(cap_max*within_max, ABS_FLOOR)``；
      R5 ``within_b2.p99 <= max(variance_ratio_cap*within_b1.p99, ABS_FLOOR)``（批量不得放大抖动）；
      R6 系统偏差的**蒙特卡洛 p 值**（``p_frac`` / ``p_max``）都 >= ``BIAS_P_CAP``。
    """
    reasons: List[str] = []
    n = int(reps if reps is not None else min(len(b1_runs), len(b2_runs)))
    if len(b1_runs) != len(b2_runs):
        reasons.append("rep_count_mismatch")
    if n < MIN_REPS:
        reasons.append(f"reps_below_min:{n}<{MIN_REPS}")
    if n < 2 or len(b1_runs) != len(b2_runs):
        return {"status": "BLOCKED", "reasons": reasons or ["reps_too_small"],
                "note": "样本量不足或两组 rep 数不一致 ⇒ 不做任何统计判定（fail-closed）"}
    try:
        if not all(np.isfinite(np.asarray(r, dtype=np.float64)).all()
                   for r in list(b1_runs) + list(b2_runs)):
            reasons.append("nonfinite_output")
    except (TypeError, ValueError):
        reasons.append("non_numeric_output")

    within_b1 = within_group_stats(b1_runs)
    within_b2 = within_group_stats(b2_runs)
    cross = cross_group_stats(b1_runs, b2_runs)
    bias = systematic_bias(b1_runs, b2_runs)

    within_p99 = max(within_b1["p99"], within_b2["p99"])
    within_max = max(within_b1["max"], within_b2["max"])
    if cross["p99"] > max(CROSS_P99_RATIO_CAP * within_p99, ABS_FLOOR):
        reasons.append(f"cross_p99_exceeds_within:{cross['p99']:.3e}>"
                       f"{CROSS_P99_RATIO_CAP}x{within_p99:.3e}")
    if cross["max"] > max(CROSS_MAX_RATIO_CAP * within_max, ABS_FLOOR):
        reasons.append(f"cross_max_exceeds_within:{cross['max']:.3e}>"
                       f"{CROSS_MAX_RATIO_CAP}x{within_max:.3e}")
    if within_b2["p99"] > max(VARIANCE_RATIO_CAP * within_b1["p99"], ABS_FLOOR):
        reasons.append(f"batch2_variance_exceeds_batch1:{within_b2['p99']:.3e}>"
                       f"{VARIANCE_RATIO_CAP}x{within_b1['p99']:.3e}")
    bias_p = bias_pvalues(b1_runs, b2_runs)
    if bias_p["p_frac"] < BIAS_P_CAP or bias_p["p_max"] < BIAS_P_CAP:
        reasons.append(f"systematic_bias:p_frac={bias_p['p_frac']:.3f},"
                       f"p_max={bias_p['p_max']:.3f}")
    return {"status": "BLOCKED" if reasons else "PASS", "reasons": reasons,
            "reps": n, "within_batch1": within_b1, "within_batch2": within_b2,
            "cross_batch1_batch2": cross, "bias": bias, "bias_pvalue": bias_p,
            "rules": {"min_reps": MIN_REPS, "cross_p99_ratio_cap": CROSS_P99_RATIO_CAP,
                      "cross_max_ratio_cap": CROSS_MAX_RATIO_CAP,
                      "variance_ratio_cap": VARIANCE_RATIO_CAP,
                      "bias_z_cap": BIAS_Z_CAP, "bias_p_cap": BIAS_P_CAP}}


# ---------------------------------------------------------------------------
# 判定（指标层：GMean / PASS）
# ---------------------------------------------------------------------------
def decision_margin(gmean: Optional[float], threshold: Optional[float]) -> Optional[float]:
    """相对 PASS 阈值的距离（>1 为余量充足，<1 为危险/不过）。

    口径与项目一致：MSE 越低越好 ⇒ ``margin = threshold / gmean``（gmean=0 时无穷大）。
    """
    if gmean is None or threshold is None or not math.isfinite(threshold) or threshold <= 0:
        return None
    if not math.isfinite(gmean) or gmean < 0:
        return None
    if gmean == 0:
        return float("inf")
    return float(threshold / gmean)


def decide_metric(*, b1_gmeans: Sequence[Optional[float]], b2_gmeans: Sequence[Optional[float]],
                  threshold: Optional[float], b1_per_traj: Sequence[Sequence[float]] = (),
                  b2_per_traj: Sequence[Sequence[float]] = ()) -> Dict[str, Any]:
    """指标层判定：两种模式下 **PASS/FAIL 判定必须稳定**，且离阈值有余量。

    规则（**全部通过才 PASS**）：
      M1 每次都拿到有限 GMean（否则 BLOCKED：指标不可用）；
      M2 每组的 PASS/FAIL 在 reps 内**完全一致**（flip ⇒ BLOCKED：判定不可靠）；
      M3 两组的 PASS/FAIL 一致（B1 过而 B2 不过，或反之 ⇒ BLOCKED）；
      M4 各组 GMean 的 **极差 < 到阈值的距离**（贴线通过 ⇒ BLOCKED：knife-edge）。
    """
    reasons: List[str] = []
    usable = [g for g in list(b1_gmeans) + list(b2_gmeans) if g is not None and math.isfinite(g)]
    if len(usable) != len(b1_gmeans) + len(b2_gmeans) or not usable:
        reasons.append("gmean_missing_or_nonfinite")
        return {"status": "BLOCKED", "reasons": reasons}
    if threshold is None or not math.isfinite(threshold) or threshold <= 0:
        reasons.append("threshold_missing")
        return {"status": "BLOCKED", "reasons": reasons}

    b1_pass = [g <= threshold for g in b1_gmeans]
    b2_pass = [g <= threshold for g in b2_gmeans]
    if len(set(b1_pass)) > 1:
        reasons.append(f"batch1_decision_flips:{b1_pass}")
    if len(set(b2_pass)) > 1:
        reasons.append(f"batch2_decision_flips:{b2_pass}")
    if set(b1_pass) != set(b2_pass):
        reasons.append(f"mode_decision_mismatch:b1={b1_pass[0]},b2={b2_pass[0]}")

    def _span(vals):
        return float(max(vals) - min(vals))

    span_b1, span_b2 = _span(b1_gmeans), _span(b2_gmeans)
    worst = max(b1_gmeans) if b1_pass[0] else min(b1_gmeans)
    distance = abs(threshold - worst)
    if max(span_b1, span_b2) >= distance:
        reasons.append(f"knife_edge:span={max(span_b1, span_b2):.3e}>=distance={distance:.3e}")
    return {"status": "BLOCKED" if reasons else "PASS", "reasons": reasons,
            "threshold": float(threshold),
            "batch1": {"gmeans": list(b1_gmeans), "pass": b1_pass, "span": span_b1,
                       "margin": [decision_margin(g, threshold) for g in b1_gmeans]},
            "batch2": {"gmeans": list(b2_gmeans), "pass": b2_pass, "span": span_b2,
                       "margin": [decision_margin(g, threshold) for g in b2_gmeans]},
            "per_traj_max_abs_diff": (
                float(np.max(np.abs(np.asarray(b1_per_traj, dtype=np.float64)
                                    - np.asarray(b2_per_traj, dtype=np.float64))))
                if b1_per_traj and b2_per_traj
                and np.shape(b1_per_traj) == np.shape(b2_per_traj) else None)}


# ---------------------------------------------------------------------------
# 随机性门：把"验收通过"变成**可被运行时校验**的证据（stochastic gate）
# ---------------------------------------------------------------------------
#: 运行时读取验收证据的环境变量（值是 `eval_batch_stochastic_acceptance.py` 产出的 gate.json）。
GATE_ENV = "AL_EVAL_BATCH_STOCHASTIC_APPROVED"


def gate_payload(*, checkpoint: Optional[str], task: Optional[str], batch_size: int,
                 dtype: Optional[str], shapes: Sequence[Any], grids: Sequence[Any]) -> Dict[str, Any]:
    """门要绑定的"运行条件"（任一不符 ⇒ 证据不适用 ⇒ 门失效）。

    ``shapes`` / ``grids`` 会被规范化成字符串，保证 JSON 往返后签名稳定。
    """
    def _canon_shape(x: Any) -> str:
        import json as _json
        if isinstance(x, Mapping):
            return _json.dumps({str(k): [int(v) for v in val] for k, val in sorted(dict(x).items())},
                               sort_keys=True)
        return str(x)

    return {"checkpoint": None if checkpoint is None else str(checkpoint),
            "task": None if task is None else str(task),
            "batch_size": int(batch_size),
            "dtype": None if dtype is None else str(dtype),
            # 去掉空白 ⇒ 张量 repr / list repr 都能得到同一个串（避免签名假不一致）
            "shapes": [_canon_shape(x) for x in shapes],
            "grids": [str(x).replace(" ", "") for x in grids]}


def gate_signature(payload: Mapping[str, Any]) -> str:
    import hashlib
    import json as _json
    blob = _json.dumps(dict(payload), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_gate(path: str, *, payload: Mapping[str, Any], verdict: Mapping[str, Any]) -> Dict[str, Any]:
    """由验收工具写出 `gate.json`：把**运行条件签名**与**验收结论**绑在一起。"""
    import json as _json
    doc = {"kind": "eval_batch_stochastic_gate", "schema_version": 1,
           "payload": dict(payload), "signature": gate_signature(payload),
           "verdict_status": str(verdict.get("status")),
           "verdict_reasons": list(verdict.get("reasons", [])),
           "worth_proposing_auto": bool(verdict.get("worth_proposing_auto", False))}
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        _json.dump(doc, fh, ensure_ascii=False, sort_keys=True, indent=1)
    import os as _os
    _os.replace(tmp, path)
    return doc


def load_gate(path: Optional[str], *, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """**fail-closed** 读取并校验门证据。任何不确定 ⇒ ``ok=False`` + 原因码。

    通过条件（全部满足）：文件存在且是合法 JSON；``signature`` 与**当前运行条件**一致；
    ``verdict_status == "PASS"``。
    """
    import json as _json
    if not path:
        return {"ok": False, "reasons": ["gate_not_configured"], "doc": None}
    try:
        doc = _json.loads(open(path, encoding="utf-8").read())
    except FileNotFoundError:
        return {"ok": False, "reasons": [f"gate_file_missing:{path}"], "doc": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reasons": [f"gate_file_unreadable:{type(exc).__name__}"], "doc": None}
    if not isinstance(doc, dict) or doc.get("kind") != "eval_batch_stochastic_gate":
        return {"ok": False, "reasons": ["gate_bad_kind"], "doc": doc}
    expected = gate_signature(payload)
    if doc.get("signature") != expected:
        return {"ok": False, "reasons": ["gate_signature_mismatch"], "doc": doc}
    if str(doc.get("verdict_status")) != "PASS":
        return {"ok": False, "reasons": [f"gate_verdict_not_pass:{doc.get('verdict_status')}"],
                "doc": doc}
    return {"ok": True, "reasons": [], "doc": doc}


def restrict_starts_to_episodes(starts: Sequence[int], ep_map: Any,
                                max_episodes: Optional[int]) -> List[int]:
    """把 chunk 起点收窄到**前 N 个回合**（指标层提速用；None = 全量）。

    ⚠️ 必须整回合保留：per-trajectory MSE 是按回合聚合的，切碎会改变口径。
    """
    if max_episodes is None or ep_map is None:
        return list(starts)
    if int(max_episodes) < 1:
        raise ValueError("max_episodes must be >= 1 or None")
    seen: List[Any] = []
    for idx in starts:
        ep = ep_map[idx]
        if ep not in seen:
            if len(seen) >= int(max_episodes):
                continue
            seen.append(ep)
    keep = set(seen)
    return [int(i) for i in starts if ep_map[i] in keep]


# ---------------------------------------------------------------------------
# 总判定
# ---------------------------------------------------------------------------
def overall_verdict(*, numeric: Mapping[str, Any], metric: Optional[Mapping[str, Any]],
                    throughput: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """总判定：**数值层 + 指标层都 PASS 才算 PASS**；指标层没跑 ⇒ BLOCKED。

    ``throughput`` 只决定"是否值得提议启用 auto"，不参与安全性判定（速度 ≠ 安全）。
    """
    reasons: List[str] = []
    if numeric.get("status") != "PASS":
        reasons.extend([f"numeric:{r}" for r in numeric.get("reasons", [])])
    if metric is None:
        reasons.append("metric_level_not_run")          # 用户要求 6：不能只比 max_abs_diff
    elif metric.get("status") != "PASS":
        reasons.extend([f"metric:{r}" for r in metric.get("reasons", [])])
    status = "PASS" if not reasons else "BLOCKED"
    speedup = None
    worth_enabling = False
    if throughput:
        s1, s2 = throughput.get("batch1_seconds_per_sample"), throughput.get("batch2_seconds_per_sample")
        if s1 and s2 and s2 > 0:
            speedup = float(s1) / float(s2)
            worth_enabling = speedup > 1.0
    return {"status": status, "reasons": reasons, "speedup_per_sample": speedup,
            "worth_proposing_auto": bool(status == "PASS" and worth_enabling),
            "note": ("PASS 仅表示'批处理的波动未超过模型自身波动且不改变 PASS 判定'；"
                     "启用 Eval auto 仍需用户单独批准（当前保持 serial）")}
