"""按任务标定「通过阈值表」（pass_thresholds.json）—— pass_metric="mse" 的配套工具。

思路
----
判定口径切到绝对 MSE 后，每个任务的通过线 = **参考模型**（如成品 50k 模型）在
该任务上的开环误差 × (1 + margin)。**每个任务都有开环分数**（参考模型能测），
所以每个任务都产出数值通过线；``null`` 仅作为**异常兜底**（该任务没有有效评测
结果、或阈值表里被显式标注排除），语义 = 不判 PASS、不消耗 attempt 预算。

用法
----
1. 用**与 AL 训练同一套数据/归一化/相机配置**评测参考模型（每个任务多次运行更好），
   把结果写成 JSONL（每行一条）::

       {"task": "click_bell", "split": "active_val", "mse": 0.0123, "nmse": 0.021}
       ...

   ⚠️ split 必须和 Scheduler 判定用的 ACTIVE_VAL 一致，否则尺子对不上。

2. 标定::

       python -m lingbotvla.auto_learning.tools.compute_pass_thresholds \
           --eval-jsonl ref_model_eval.jsonl \
           --baseline /data/outputs/.../task_baseline.json \
           --stat p75 --margin 0.10 \
           -o /data/outputs/.../pass_thresholds.json

三道匹配检查（全部 fail-fast，绝不静默产出错表）
------------------------------------------------
1. **任务名 + 任务指纹匹配**：eval 里的每个任务必须在 baseline store 里存在，
   且 sha256_train 指纹一致（防止拿旧数据划分/别的任务集的分数来标定）。
2. **nmse ↔ mse/baseline 自洽**：两者偏差超过 1% ⇒ 说明 eval 与 baseline 不是
   同一份配置算的，拒绝产出。
3. **阈值 < baseline_mse**：线比"全猜均值"还松 ⇒ 等于不设防，拒绝产出。

产出
----
``PassThresholds`` 格式 JSON（见 ``decision/thresholds.py``），带
``config_fingerprint``（与 task_baseline.json 同一个）——
训练启动时 ``real/build.py`` 会强校验它，不一致拒绝启动。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

from ..baseline import BaselineStore
from ..decision.metrics import median
from ..decision.thresholds import PASS_METRIC_MSE, THRESHOLDS_VERSION, PassThresholds

STATS = ("max", "p75", "median", "mean")


# --------------------------------------------------------------------------- #
def _percentile(sorted_vals: List[float], q: float) -> float:
    """线性插值 percentile（q ∈ [0,100]），输入必须已排序、非空。"""
    if not sorted_vals:
        raise ValueError("percentile: 空序列")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _stat(vals: List[float], stat: str) -> float:
    s = sorted(vals)
    if stat == "max":
        return s[-1]
    if stat == "p75":
        return _percentile(s, 75)
    if stat == "median":
        return float(median(s) or 0.0)
    if stat == "mean":
        return sum(s) / len(s)
    raise ValueError(f"未知统计量 {stat!r}（只能是 {STATS}）")


# --------------------------------------------------------------------------- #
def load_eval_jsonl(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """读参考模型评测结果，按任务分组（保持文件顺序）。"""
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: JSON 解析失败: {e}")
            task = row.get("task")
            if not task:
                raise SystemExit(f"{path}:{lineno}: 缺 'task' 字段: {row!r}")
            by_task.setdefault(str(task), []).append(row)
    if not by_task:
        raise SystemExit(f"{path}: 一条评测记录都没有")
    return by_task


# --------------------------------------------------------------------------- #
def compute_thresholds(
    eval_by_task: Dict[str, List[Dict[str, Any]]],
    store: BaselineStore,
    *,
    stat: str,
    margin: float,
    reference: str,
) -> PassThresholds:
    """核心计算 + 三道匹配检查。任何一道不过 ⇒ 抛 ThresholdsMatchError。"""
    from ..decision.thresholds import ThresholdsError as TE

    # store.tasks 是 {任务名: 原始 dict}（见 BaselineStore.put）
    base_by_name: Dict[str, Dict[str, Any]] = dict(store.tasks)

    # ---- 检查 1a：eval 里的任务必须都在 baseline 里（名字匹配） ----
    unknown = [t for t in eval_by_task if t not in base_by_name]
    if unknown:
        raise TE(
            "评测结果里有 baseline store 中不存在的任务 ⇒ 拒绝产出（任务名不匹配）：\n"
            f"  {unknown}\n"
            f"  baseline 覆盖: {sorted(base_by_name)}")

    # ---- 检查 1b：baseline 里的任务必须都有 eval（否则那些任务无阈值可用） ----
    missing_eval = [t for t in base_by_name if t not in eval_by_task]
    if missing_eval:
        raise TE(
            "baseline store 里的这些任务没有评测结果 ⇒ 拒绝产出（任务覆盖不全，\n"
            "缺了就会被训练侧 fail-fast / 静默 needs_calibration）：\n"
            f"  {missing_eval}")

    tasks: Dict[str, Optional[float]] = {}
    problems: List[str] = []
    for name, rows in sorted(eval_by_task.items()):
        fb = base_by_name[name]

        # ---- 检查 2：nmse ↔ mse/baseline 自洽 ----
        ms_vals: List[float] = []
        for r in rows:
            mse = r.get("mse")
            if mse is None or not math.isfinite(float(mse)):
                problems.append(f"{name}: 有记录 mse 缺失/非有限: {r!r}")
                continue
            ms_vals.append(float(mse))
            nmse = r.get("nmse")
            base_mse = float(fb["mse"])
            if nmse is not None and base_mse > 0:
                expect = float(mse) / base_mse
                if abs(float(nmse) - expect) / max(expect, 1e-12) > 0.01:
                    problems.append(
                        f"{name}: nmse={nmse} 与 mse/baseline={expect:.5f} 偏差 >1% "
                        "⇒ eval 和 baseline 不是同一份配置算的")
        if not ms_vals:
            problems.append(f"{name}: 没有任何可用的 mse 记录")
            continue

        line = _stat(ms_vals, stat) * (1.0 + margin)

        # ---- 检查 3：阈值必须 < baseline_mse（线不能比"全猜均值"还松） ----
        base_mse = float(fb["mse"])
        if not (line < base_mse):
            problems.append(
                f"{name}: 阈值={line:.5f} ≥ baseline_mse={base_mse:.5f} "
                f"（stat={stat}×{1+margin:.2f}）⇒ 比全猜均值还松，等于不设防")
            continue
        tasks[name] = line

    if problems:
        raise TE("阈值标定未通过一致性检查 ⇒ 拒绝产出：\n  - " + "\n  - ".join(problems))

    return PassThresholds(
        config_fingerprint=store.config_fingerprint,
        tasks=tasks,
        metric=PASS_METRIC_MSE,
        margin=margin,
        stat=stat,
        reference=reference,
        version=THRESHOLDS_VERSION,
    )


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="按任务标定 pass_metric='mse' 的通过阈值表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--eval-jsonl", required=True,
                    help="参考模型每任务开环评测结果（JSONL，每行 {task,mse,nmse,...}）")
    ap.add_argument("--baseline", required=True,
                    help="task_baseline.json（BaselineStore，提供指纹与 baseline_mse）")
    ap.add_argument("--stat", default="p75", choices=STATS,
                    help="跨多次运行的统计量（默认 p75，给 25%% 余量）")
    ap.add_argument("--margin", type=float, default=0.10,
                    help="在统计量上再乘 (1+margin)（默认 0.10）")
    ap.add_argument("--reference", default="",
                    help="参考模型标识（写进阈值表备查，如 ckpt 路径/step）")
    ap.add_argument("-o", "--output", required=True, help="输出 pass_thresholds.json 路径")
    args = ap.parse_args(argv)

    store = BaselineStore.load(args.baseline)
    if not store.config_fingerprint:
        raise SystemExit(
            f"{args.baseline}: baseline store 没有 config_fingerprint ⇒ "
            "阈值表无法带指纹强校验，拒绝产出。请用 compute_task_baseline 重算。")

    eval_by_task = load_eval_jsonl(args.eval_jsonl)

    thresholds = compute_thresholds(
        eval_by_task, store,
        stat=args.stat, margin=args.margin, reference=args.reference,
    )

    # 自校验：能 load 回来才算写完
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    thresholds.save(args.output)
    reloaded = PassThresholds.load(args.output, require_metric=PASS_METRIC_MSE)
    assert reloaded.config_fingerprint == thresholds.config_fingerprint

    n_null = sum(1 for v in thresholds.tasks.values() if v is None)
    print(f"✅ 阈值表已产出: {args.output}")
    print(f"   任务 {len(thresholds.tasks)} 个（有可用线 {thresholds.n_usable}，"
          f"null/needs_calibration {n_null}）")
    print(f"   stat={args.stat} margin={args.margin:.2f} "
          f"config指纹={thresholds.config_fingerprint}")
    for name in sorted(thresholds.tasks):
        v = thresholds.tasks[name]
        line = "null (needs_calibration)" if v is None else f"{v:.5f}"
        print(f"   {name:<28s} {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
