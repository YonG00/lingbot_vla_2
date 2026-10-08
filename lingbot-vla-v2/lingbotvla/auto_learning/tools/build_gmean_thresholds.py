"""安全生成基于参考模型逐轨迹 MSE 几何均值的任务 PASS 阈值表。

这是离线标定工具，不修改模型或训练循环。产物兼容现有 PassThresholds。

输入每行: {"task": "click_bell", "traj": 42, "mse": 0.00012,
             "split": "active_val", "nmse": <optional>}

仅用 reference (例如 50k) 的逐轨迹 MSE 建立任务基准；候选模型的 MSE
应该在相同轨迹、精度、噪声种子与归一化条件下单独测量。

重要: 本工具产生的是“开环候选 PASS 线”，不是闭环成功的证明。
CV 高波动时可显式使用 --high-cv-policy warn：保留阈值并在诊断中告警。
默认 null 沿用旧 fail-closed 行为；null 会让 Scheduler 跳过该任务。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..baseline import BaselineStore
from ..decision.thresholds import PASS_METRIC_MSE, PassThresholds, ThresholdsError


def _number(value: Any, where: str, *, strictly_positive: bool = True) -> float:
    if isinstance(value, bool):
        raise ThresholdsError(f"{where}: 布尔值不是合法数值")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ThresholdsError(f"{where}: 不是数值：{value!r}") from None
    if not math.isfinite(x) or (strictly_positive and x <= 0.0):
        raise ThresholdsError(f"{where}: 必须是有限正数：{value!r}")
    return x


def _sha256_file(path: str) -> str:
    """兼容 Python 3.8+，避免依赖 Python 3.11 才有的 hashlib.file_digest。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference(
    path: str,
    store: BaselineStore,
    *,
    split: str = "active_val",
) -> Dict[str, List[float]]:
    """读任务逐轨迹 MSE；相同任务下重复 traj ID 直接报错，避免样本数虚增。"""
    values: Dict[str, List[float]] = defaultdict(list)
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line_no, text in enumerate(f, 1):
            if not text.strip():
                continue
            where = f"{path}:{line_no}"
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ThresholdsError(f"{where}: JSON 解析失败：{exc}") from exc
            if not isinstance(row, dict):
                raise ThresholdsError(f"{where}: 必须是 JSON object")
            task = row.get("task")
            if not isinstance(task, str) or not task:
                raise ThresholdsError(f"{where}: 缺少非空 task")
            if task not in store.tasks:
                raise ThresholdsError(f"{where}: 任务 {task!r} 不存在于 baseline store")
            traj = next((row[k] for k in ("traj", "traj_id", "episode_id") if k in row), None)
            if type(traj) not in (str, int) or not str(traj).strip():
                raise ThresholdsError(f"{where}: traj/traj_id/episode_id 必须是非空字符串或整数")
            key = (task, str(traj))
            if key in seen:
                raise ThresholdsError(f"{where}: 重复轨迹 {key!r}；请先聚合多个噪声种子")
            seen.add(key)
            if row.get("split") is not None and row["split"] != split:
                raise ThresholdsError(
                    f"{where}: split={row['split']!r}，要求 {split!r}"
                )
            if row.get("config_fingerprint") is not None and row["config_fingerprint"] != store.config_fingerprint:
                raise ThresholdsError(f"{where}: config_fingerprint 与 baseline 不一致")
            if row.get("task_fingerprint") is not None and row["task_fingerprint"] != store.tasks[task].get("fingerprint"):
                raise ThresholdsError(f"{where}: task_fingerprint 与 baseline 不一致")
            mse = _number(row.get("mse"), f"{where} mse")
            if row.get("nmse") is not None:
                nmse = _number(row["nmse"], f"{where} nmse", strictly_positive=False)
                baseline = _number(store.tasks[task]["mse"], f"{where} baseline")
                expected = mse / baseline
                if abs(nmse - expected) / max(expected, 1e-12) > 0.01:
                    raise ThresholdsError(f"{where}: nmse 与 mse/baseline 不一致（偏差 >1%）")
            values[task].append(mse)
    if not values:
        raise ThresholdsError("参考逐轨迹文件无有效数据")
    missing = sorted(set(store.tasks) - set(values))
    if missing:
        raise ThresholdsError(f"参考数据缺任务：{missing}")
    return dict(values)


def load_reference_log(
    log_path: str,
    split_dir: str,
    store: BaselineStore,
    *,
    val_trajs: int = 10,
) -> Dict[str, List[float]]:
    """直接从 open_loop_eval.log + 每任务 val_ids 还原逐轨迹 MSE。

    严格拒绝重复、缺失、多余轨迹，防止上游日志“最后一次覆盖”的静默问题。
    注意：这个日志不能证明精度/噪声种子一致，使用者仍需核对运行参数。
    """
    if val_trajs < 2:
        raise ThresholdsError("val_trajs 必须 >= 2")
    split = Path(split_dir)
    expected: Dict[int, str] = {}
    tasks_seen = set()
    for path in sorted(split.glob("*.val_ids.json")):
        if path.name.startswith("combined"):
            continue
        task = path.name[: -len(".val_ids.json")]
        if task not in store.tasks:
            raise ThresholdsError(f"split 中的任务 {task!r} 不在 baseline 中")
        try:
            ids = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ThresholdsError(f"无法读取 {path}: {exc}") from exc
        if not isinstance(ids, list) or len(ids) < val_trajs:
            raise ThresholdsError(f"{path}: val 轨迹不足 {val_trajs} 条")
        for tid in ids[:val_trajs]:
            if type(tid) is not int or tid < 0:
                raise ThresholdsError(f"{path}: 非法轨迹编号 {tid!r}")
            if tid in expected:
                raise ThresholdsError(f"轨迹 {tid} 在任务 {expected[tid]!r} 与 {task!r} 中重复")
            expected[tid] = task
        tasks_seen.add(task)
    missing_tasks = sorted(set(store.tasks) - tasks_seen)
    if missing_tasks:
        raise ThresholdsError(f"split 缺任务：{missing_tasks}")

    pattern = re.compile(r"\bMSE for trajectory (\d+):\s*([^\s,]+)")
    found: Dict[int, float] = {}
    with open(log_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            match = pattern.search(line)
            if not match:
                continue
            tid = int(match.group(1))
            if tid not in expected:
                raise ThresholdsError(f"{log_path}:{line_no}: 意外轨迹 {tid}")
            if tid in found:
                raise ThresholdsError(
                    f"{log_path}:{line_no}: 轨迹 {tid} 重复记录；"
                    "请先审查重复测量，不能悄悄用最后一条覆盖"
                )
            found[tid] = _number(match.group(2), f"{log_path}:{line_no} mse")
    missing_ids = sorted(set(expected) - set(found))
    if missing_ids:
        raise ThresholdsError(f"日志缺 {len(missing_ids)} 条轨迹：{missing_ids[:20]}")
    by_task: Dict[str, List[float]] = defaultdict(list)
    for tid, task in expected.items():
        by_task[task].append(found[tid])
    return dict(by_task)


def build_thresholds(
    per_task: Dict[str, List[float]],
    store: BaselineStore,
    *,
    multiplier: float,
    min_trajectories: int = 10,
    max_cv: float = 1.5,
    baseline_cap: float = 0.99,
    high_cv_policy: str = "null",
    reference: str,
) -> Tuple[PassThresholds, Dict[str, Any]]:
    """根据 GMean 生成 PassThresholds 和逐任务诊断表；CV 超限不自动 PASS。"""
    mult = _number(multiplier, "multiplier")
    cv_limit = _number(max_cv, "max_cv")
    cap = _number(baseline_cap, "baseline_cap")
    if not (0 < cap < 1):
        raise ThresholdsError("baseline_cap 必须在 (0,1) 内，确保阈值严格低于 baseline")
    if type(min_trajectories) is not int or min_trajectories < 2:
        raise ThresholdsError("min_trajectories 必须 >= 2")
    if high_cv_policy not in ("null", "error", "warn"):
        raise ThresholdsError("high_cv_policy 只能是 null / error / warn")
    if not store.config_fingerprint:
        raise ThresholdsError("baseline store 缺 config_fingerprint")
    if set(per_task) != set(store.tasks):
        raise ThresholdsError(
            f"任务覆盖不一致: missing={sorted(set(store.tasks) - set(per_task))}, "
            f"unknown={sorted(set(per_task) - set(store.tasks))}"
        )
    tasks: Dict[str, Optional[float]] = {}
    diagnostics: Dict[str, Any] = {}
    for task, vals in sorted(per_task.items()):
        if len(vals) < min_trajectories:
            raise ThresholdsError(f"{task}: 只有 {len(vals)} 条轨迹，要求 >= {min_trajectories}")
        checked = [_number(x, f"{task} MSE") for x in vals]
        baseline = _number(store.tasks[task]["mse"], f"{task} baseline")
        mean = statistics.mean(checked)
        cv = statistics.stdev(checked) / mean
        geo = math.exp(statistics.mean(math.log(x) for x in checked))
        raw_line = geo * mult
        if not math.isfinite(raw_line) or raw_line <= 0:
            raise ThresholdsError(f"{task}: GMean × 倍数溢出")
        capped = min(raw_line, cap * baseline)
        high_cv = cv > cv_limit
        is_capped = raw_line > cap * baseline
        if high_cv and high_cv_policy == "error":
            raise ThresholdsError(f"{task}: CV={cv:.3f} > {cv_limit}，参考波动过高")
        if high_cv and high_cv_policy == "null":
            tasks[task] = None
            decision = "needs_closed_loop_manual"
        else:
            tasks[task] = capped
            decision = ("high_cv_warning" if high_cv else
                        "baseline_capped" if is_capped else "usable")
        diagnostics[task] = {
            "n": len(vals), "gmean": geo, "arithmetic_mean": mean, "cv": cv,
            "baseline_mse": baseline, "unclipped_line": raw_line,
            "effective_line": tasks[task], "status": decision,
            "high_cv_warning": high_cv, "baseline_capped": is_capped,
        }
    obj = PassThresholds(
        config_fingerprint=store.config_fingerprint,
        tasks=tasks,
        metric=PASS_METRIC_MSE,
        stat="geomean",
        margin=0.0,
        reference=reference,
    )
    return obj, diagnostics


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="从 50k 参考模型逐轨迹 MSE 生成 GMean × M 开环候选 PASS 线")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ref-per-traj", help="逐轨迹 JSONL：task, traj, mse")
    source.add_argument("--ref-log", help="已有 open_loop_eval.log，需配合 --split-dir")
    parser.add_argument("--split-dir", help="--ref-log 模式必填：包含各任务 *.val_ids.json 的目录")
    parser.add_argument("--val-trajs", type=int, default=10, help="--ref-log 中每任务使用的 val 轨迹数")
    parser.add_argument("--baseline", required=True, help="task_baseline.json")
    parser.add_argument("--reference", required=True, help="参考权重标识，如 global_step_50000")
    parser.add_argument("--multiplier", required=True, type=float, help="实验候选倍数（例如 220；不是定论）")
    parser.add_argument("--min-trajectories", type=int, default=10)
    parser.add_argument("--max-cv", type=float, default=1.5)
    parser.add_argument("--baseline-cap", type=float, default=0.99)
    parser.add_argument("--high-cv-policy", choices=("null", "error", "warn"), default="null")
    parser.add_argument("--expected-split", default="active_val")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args(argv)

    store = BaselineStore.load(args.baseline)
    if args.ref_log:
        if not args.split_dir:
            parser.error("--ref-log 需要同时提供 --split-dir")
        per_task = load_reference_log(args.ref_log, args.split_dir, store, val_trajs=args.val_trajs)
    else:
        per_task = load_reference(args.ref_per_traj, store, split=args.expected_split)
    table, details = build_thresholds(
        per_task, store, multiplier=args.multiplier,
        min_trajectories=args.min_trajectories, max_cv=args.max_cv,
        baseline_cap=args.baseline_cap, high_cv_policy=args.high_cv_policy,
        reference=args.reference,
    )
    # 不修改 PassThresholds 的运行时 schema；保留附加的离线审核信息。
    obj = table.to_dict()
    source_path = args.ref_log or args.ref_per_traj
    source_hash = _sha256_file(source_path)
    obj["calibration"] = {
        "method": "reference_per_trajectory_geomean_multiplier",
        "multiplier": args.multiplier,
        "min_trajectories": args.min_trajectories,
        "max_cv": args.max_cv,
        "baseline_cap": args.baseline_cap,
        "high_cv_policy": args.high_cv_policy,
        "expected_split": args.expected_split,
        "source_type": "open_loop_eval_log" if args.ref_log else "per_trajectory_jsonl",
        "source_file": os.path.basename(source_path),
        "source_sha256": source_hash,
        "tasks": details,
    }
    out = Path(args.output)
    if out.is_dir():
        raise ThresholdsError(f"输出路径是目录，不是文件：{out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=out.parent, prefix=out.name + ".", suffix=".tmp", delete=False) as f:
            temp = f.name
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        # 写入前校验格式和指纹（PassThresholds 会忽略额外的 calibration 字段）
        reloaded = PassThresholds.load(temp, expect_fingerprint=store.config_fingerprint, require_metric=PASS_METRIC_MSE)
        if reloaded.tasks != table.tasks:
            raise ThresholdsError("写入后阈值表校验失败")
        os.replace(temp, out)
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)
    excluded = [name for name, info in details.items() if info["status"] == "needs_closed_loop_manual"]
    capped = [name for name, info in details.items() if info["baseline_capped"]]
    print(f"阈值表: {out} | 可用 {table.n_usable}/{len(table.tasks)}")
    warned = [name for name, info in details.items() if info["high_cv_warning"]]
    print(f"阈值为 null（Scheduler 会跳过）：{excluded}")
    print(f"参考 CV 超标任务（warn 策略下仍训练并允许开环 PASS）：{warned}")
    print(f"触发 baseline 上限：{capped}")
    print("注意：此阈值仅用于开环候选 PASS，不证明闭环成功。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
