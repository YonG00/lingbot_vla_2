#!/usr/bin/env python3
"""标定「每任务通关线」：**倍数线** vs **σ 线**，用闭环锚点判定哪个可分、以及怎么选。

背景（2026-10-08）
-----------------
* 开环指标与闭环成功**非单调**（`stamp_seal` confirm nmse 0.0893→0% vs `click_alarmclock` 0.1020→100%）
  ⇒ 全局一把尺（全局 nmse / 全局 MSE）不可用。
* 按任务的线有两种口径：
    - **倍数线**：`line = μ_ref × M`，判定量 `r = mse_cand / μ_ref`
    - **σ 线**  ：`line = μ_ref + K·σ_ref`，判定量 `z = (mse_cand − μ_ref)/σ_ref ≡ (r−1)/CV`
      （CV = σ_ref/μ_ref；这就是"差成品几个自身标准差"）
* 本工具用**已有闭环锚点**量化：两种口径各自的"PASS 最大 / FAIL 最小"间隙、
  以及该间隙在**测量噪声**下能否站住（bootstrap 出「可分的概率」）。

关键实测事实（决定了 σ 线在这里为什么反而更差）
* 单条轨迹的**测量重复性**：同 episode 跑两遍，MSE 相对差中位 **48.8%**（去噪噪声不重置）
  ⇒ 单次测量相对 sd ≈ 48.8/√2 ≈ **34.5%**；n 条轨迹求均值的标准误 ≈ 34.5%/√n。
* 参考模型逐轨迹 MSE 的 **CV 跨任务差 4.9×**（0.58~2.88）⇒ σ 线会把"成品本身就不稳"的任务
  容差撑宽（z 变小），于是 FAIL 任务可能落到 PASS 区间内 ⇒ **交叉**。

用法
----
    python -m lingbotvla.auto_learning.tools.calibrate_pass_line \
        --ref-per-traj ref_per_traj.jsonl \
        --candidate-json candidate_mse.json \
        --anchors anchors.json \
        --noise-rel-sd 0.345 --candidate-n 4 \
        --out calib_report.md

  ref_per_traj.jsonl   每行 `{"task":..,"traj":..,"mse":..}`（**逐轨迹**，每任务 n 条）
  candidate_mse.json   `{"任务": mse}`（候选模型在同一批 val 轨迹上的开环 MSE 均值）
  anchors.json         `{"任务": 闭环成功率 0~1}`；0/1 之外的算 mid（不参与标定，只打印）
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_NOISE_REL_SD = 0.488 / math.sqrt(2)  # 同轨迹两遍相差 48.8% ⇒ 单次测量相对 sd


# --------------------------------------------------------------------------- #
def load_per_traj(path: str) -> Dict[str, List[float]]:
    by: Dict[str, List[float]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        by[str(row["task"])].append(float(row["mse"]))
    return dict(by)


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean_sd(v: List[float]) -> Tuple[float, float]:
    if len(v) < 2:
        return (v[0] if v else float("nan")), 0.0
    return statistics.mean(v), statistics.stdev(v)


def _gap(values: Dict[str, float], pass_t: List[str], fail_t: List[str]) -> Optional[float]:
    """间隙系数 = FAIL 最小 / PASS 最大；>1 表示可分。"""
    if not pass_t or not fail_t:
        return None
    p = max(values[t] for t in pass_t)
    f = min(values[t] for t in fail_t)
    return f / p if p > 0 else None


# --------------------------------------------------------------------------- #
def bootstrap(
    by_task: Dict[str, List[float]],
    cand: Dict[str, float],
    pass_t: List[str],
    fail_t: List[str],
    *,
    noise_rel_sd: float,
    candidate_n: int,
    iters: int,
    seed: int = 0,
) -> Dict[str, float]:
    """重采样参考轨迹（μ/σ 的估计噪声）+ 候选侧按 n 条轨迹的噪声模型扰动。

    返回 {'P_multiple': 可分概率, 'P_z': 可分概率}
    """
    rng = random.Random(seed)
    ok_m = ok_z = 0
    for _ in range(iters):
        mu: Dict[str, float] = {}
        sd: Dict[str, float] = {}
        for t in list(pass_t) + list(fail_t):
            v = by_task[t]
            s = [rng.choice(v) for _ in v]
            mu[t], sd[t] = _mean_sd(s)
        pert: Dict[str, float] = {}
        for t in list(pass_t) + list(fail_t):
            n = max(1, int(candidate_n))
            pert[t] = max(cand[t] * (1.0 + rng.gauss(0.0, noise_rel_sd / math.sqrt(n))), 1e-12)
        r = {t: pert[t] / mu[t] for t in pert}
        z = {t: ((pert[t] - mu[t]) / sd[t]) if sd[t] > 0 else 0.0 for t in pert}
        gm, gz = _gap(r, pass_t, fail_t), _gap(z, pass_t, fail_t)
        ok_m += int(gm is not None and gm > 1.0)
        ok_z += int(gz is not None and gz > 1.0)
    return {"P_multiple": ok_m / iters, "P_z": ok_z / iters}


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="标定每任务通关线：倍数线 vs σ 线")
    ap.add_argument("--ref-per-traj", required=True, help="参考模型逐轨迹 MSE（jsonl）")
    ap.add_argument("--candidate-json", required=True, help="候选模型 {任务: mse}（json）")
    ap.add_argument("--anchors", required=True, help="闭环锚点 {任务: 成功率}（json）")
    ap.add_argument("--noise-rel-sd", type=float, default=DEFAULT_NOISE_REL_SD,
                    help=f"单次测量的相对 sd（默认 {DEFAULT_NOISE_REL_SD:.3f}）")
    ap.add_argument("--candidate-n", type=int, default=4, help="候选侧用于求均值的轨迹数（默认 4）")
    ap.add_argument("--cv-warn", type=float, default=1.5, help="CV 超过它 ⇒ 该任务开环数不可信（默认 1.5）")
    ap.add_argument("--iters", type=int, default=3000, help="bootstrap 次数")
    ap.add_argument("--out", default="", help="输出 markdown 报告路径")
    a = ap.parse_args(argv)

    by_task = load_per_traj(a.ref_per_traj)
    cand_all = load_json(a.candidate_json)
    anchors = {k: float(v) for k, v in load_json(a.anchors).items()}

    P = [t for t, v in anchors.items() if v >= 0.999]
    F = [t for t, v in anchors.items() if v <= 0.001]
    MID = [t for t, v in anchors.items() if 0.001 < v < 0.999]
    missing = [t for t in list(anchors) if t not in by_task or t not in cand_all]
    if missing:
        sys.exit(f"这些锚点缺参考逐轨迹数据或候选 MSE：{missing}")

    out: List[str] = []
    w = out.append
    w("# 每任务通关线标定：倍数线 vs σ 线\n")
    w(f"- 参考逐轨迹：`{a.ref_per_traj}`（每任务 n={statistics.median([len(v) for v in by_task.values()]):.0f} 条）")
    w(f"- 候选 MSE：`{a.candidate_json}`（默认按 n={a.candidate_n} 条轨迹求均值建模其噪声）")
    w(f"- 锚点：`{a.anchors}` ⇒ PASS={P}｜FAIL={F}｜mid={MID}")
    w(f"- 测量噪声：单次相对 sd = {a.noise_rel_sd:.1%}（同轨迹两遍相差中位 48.8% ⇒ /√2）\n")

    stats: Dict[str, Dict[str, float]] = {}
    for t in list(anchors):
        mu, sd = _mean_sd(by_task[t])
        cv = sd / mu if mu else float("nan")
        stats[t] = {"mu": mu, "sd": sd, "cv": cv, "n": len(by_task[t])}

    w("## ① 逐锚点取值\n")
    w("| 任务 | n | μ_ref | σ_ref | CV | 候选 MSE | **倍数 r** | **σ 距离 z** | 标签 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for t in sorted(anchors, key=lambda x: cand_all[x] / stats[x]["mu"]):
        s = stats[t]
        r = cand_all[t] / s["mu"]
        z = (cand_all[t] - s["mu"]) / s["sd"] if s["sd"] > 0 else float("inf")
        lab = "PASS" if t in P else ("FAIL" if t in F else f"mid({anchors[t]:.0%})")
        w(f"| {t} | {s['n']:.0f} | {s['mu']:.6f} | {s['sd']:.6f} | {s['cv']:.2f} | "
          f"{cand_all[t]:.5f} | {r:.0f}× | {z:.0f}σ | {lab} |")

    rvals = {t: cand_all[t] / stats[t]["mu"] for t in anchors}
    zvals = {t: (cand_all[t] - stats[t]["mu"]) / stats[t]["sd"] for t in anchors if stats[t]["sd"] > 0}
    gm, gz = _gap(rvals, P, F), _gap(zvals, P, F)
    w("\n## ② 两种口径的可分性（用锚点判定）\n")
    w("| 口径 | PASS 最大 | FAIL 最小 | 间隙系数 | 结论 |")
    w("|---|---|---|---|---|")
    if gm is not None:
        w(f"| **倍数** `r` | {max(rvals[t] for t in P):.0f}× | {min(rvals[t] for t in F):.0f}× | "
          f"**{gm:.2f}×** | {'✅ 可分' if gm > 1 else '❌ 交叉'} |")
    if gz is not None:
        w(f"| **σ 距离** `z` | {max(zvals[t] for t in P):.0f}σ | {min(zvals[t] for t in F):.0f}σ | "
          f"**{gz:.2f}×** | {'✅ 可分' if gz > 1 else '❌ 交叉'} |")

    bs = bootstrap(by_task, cand_all, P, F, noise_rel_sd=a.noise_rel_sd,
                   candidate_n=a.candidate_n, iters=a.iters)
    w("\n## ③ 这些间隙吃得住测量噪声吗（bootstrap）\n")
    w(f"- 候选侧只有 n={a.candidate_n} 条轨迹 ⇒ 其均值的相对标准误 ≈ "
      f"{a.noise_rel_sd / math.sqrt(max(a.candidate_n,1)):.1%}")
    w(f"- **倍数线**: 同一份重采样里可分（FAIL最小 > PASS最大）的概率 = **{bs['P_multiple']:.1%}**")
    w(f"- **σ 线** : 可分概率 = **{bs['P_z']:.1%}**")
    w(f"- ⇒ {'两者都被噪声盖住，先降噪再谈选哪个' if max(bs.values()) < 0.9 else '至少一种口径在噪声下站得住'}")

    w("\n## ④ 建议\n")
    if gm is not None and gz is not None and gm > gz:
        w(f"- 当前数据：**倍数线可分（{gm:.2f}×）、σ 线交叉（{gz:.2f}×）** ⇒ "
          f"用 `line = μ_ref × M`，`M ≈ {(max(rvals[t] for t in P) * min(rvals[t] for t in F)) ** 0.5:.0f}`"
          f"（区间 [{max(rvals[t] for t in P):.0f}, {min(rvals[t] for t in F):.0f}]）。")
    else:
        w("- 两种口径都没分离 ⇒ 换参考模型（用已验证能闭环成功的模型）或改两段式（开环粗筛 + 闭环确认）。")
    w(f"- σ 线语义上仍是正确的**显著性/容差**判据（“差成品几个自身标准差”），但它把"
      f"「成品本身不稳」的任务容差撑宽 ⇒ 在这里产生交叉；**要当通关线，需先降噪**"
      f"（`open_loop_eval.py --fixed_seed_per_traj` 或 `--noise_repeats 3`）。")
    hi_cv = sorted([t for t in stats if stats[t]["cv"] > a.cv_warn], key=lambda t: -stats[t]["cv"])
    if hi_cv:
        _lst = ", ".join("%s(%.2f)" % (t, stats[t]["cv"]) for t in hi_cv)
        w("- 🔴 这些任务 CV > %.2f（参考模型自身就不稳）⇒ **不要用开环线判**，直接走闭环确认：%s"
          % (a.cv_warn, _lst))
    w(f"- 测量精度现状：候选侧 n=4 时其均值标准误 ≈ {a.noise_rel_sd/math.sqrt(4):.1%}；"
      f"建议候选侧 ≥10 条轨迹、或每条轨迹 k≥3 个噪声种子取平均（成本 ×k）。")

    txt = "\n".join(out) + "\n"
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
        print(f"✅ 报告已写出：{a.out}")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
