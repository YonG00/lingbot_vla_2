"""改判对拍：新的「按任务 MSE 阈值」 vs 旧口径 ``pass_nmse``（诊断工具，不参与训练）。

回答两个问题
------------
1. **改判清单**：换成按任务 MSE 阈值后，哪些任务的 PASS 判定会翻转？
   （旧口径 = 全局 ``nmse <= pass_nmse``；新口径 = 逐任务 ``mse <= thresholds[task]``）
2. **尺子可比性**：阈值表是用「参考模型 × N 条 val 轨迹」标定的，而候选模型
   （如 step500 探针）可能只有 2/4 条 ⇒ 实测同一任务 2 条 vs 4 条能差 30–50%。
   ⇒ 本工具会把参考模型在 2/4/10 条轨迹下的线**一并算出来**，指出哪些翻转
   只是「样本数不同」造成的假信号。

输入
----
* ``--events``      AL 事件 JSONL（``task/<t>/scout_nmse`` / ``scout_mse`` / ``confirm_*``）
* ``--baseline``    ``task_baseline.json``（BaselineStore）
* ``--thresholds``  ``pass_thresholds.json``（PassThresholds，metric=mse）
* ``--ref-eval-log`` 可选：参考模型 open_loop_eval 日志（逐条 ``MSE for trajectory <id>: <mse>``）
* ``--split-dir``   可选：``task_splits_50``（把 traj id 映射回任务，配合 ``--ref-eval-log``）
* ``--anchors``     可选：``{"task": 闭环成功率}`` 的 JSON，仅用于打印对照

用法
----
    python -m lingbotvla.auto_learning.tools.recheck_thresholds \
        --events /data/outputs/al_probe_step500/auto_learning_events.jsonl \
        --baseline /data/train/task_splits_50/task_baseline.json \
        --thresholds /data/train/task_splits_50/pass_thresholds.json \
        --ref-eval-log /data/eval_results/open_loop/ref50k/open_loop_eval.log \
        --split-dir /data/train/task_splits_50 \
        --out /data/tmp/thresholds_recheck.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..baseline import BaselineStore
from ..decision.thresholds import PassThresholds


# --------------------------------------------------------------------------- #
def load_events_metrics(path: str) -> Dict[str, Dict[str, float]]:
    """事件 JSONL → ``{task: {metric: value}}``（同名指标取最后一次）。"""
    out: Dict[str, Dict[str, float]] = {}
    pat = re.compile(r"^(?:task|debug)/([^/]+)/([a-z_]+)$")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = row.get("name") or ""
            val = row.get("value")
            if not isinstance(val, (int, float)):
                continue
            m = pat.match(name)
            if m:
                out.setdefault(m.group(1), {})[m.group(2)] = float(val)
    return out


def load_ref_per_traj(path: str) -> Dict[int, float]:
    """参考模型日志 → ``{traj_id: mse}``。"""
    out: Dict[int, float] = {}
    rx = re.compile(r"MSE for trajectory (\d+):\s*([-+0-9.eE]+)")
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = rx.search(line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def load_split(split_dir: str) -> Dict[str, List[int]]:
    """``task_splits_50`` → ``{task: [val_traj_id...]}``（跳过 combined）。"""
    out: Dict[str, List[int]] = {}
    for p in sorted(Path(split_dir).glob("*.val_ids.json")):
        if p.name.startswith("combined"):
            continue
        out[p.name[: -len(".val_ids.json")]] = json.loads(p.read_text())
    return out


def _p75(vals: List[float]) -> float:
    s = sorted(vals)
    if not s:
        return float("nan")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * 0.75
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] * (1 - (pos - lo)) + s[hi] * (pos - lo)


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="按任务 MSE 阈值 vs 旧 pass_nmse 改判对拍")
    ap.add_argument("--events", required=True, help="AL 事件 JSONL（含 scout/confirm 指标）")
    ap.add_argument("--baseline", required=True, help="task_baseline.json")
    ap.add_argument("--thresholds", required=True, help="pass_thresholds.json")
    ap.add_argument("--old-pass-nmse", type=float, default=0.35, help="旧全局阈值（默认 0.35）")
    ap.add_argument("--ref-eval-log", default="", help="参考模型 open_loop_eval 日志（可选）")
    ap.add_argument("--split-dir", default="", help="task_splits_50（可选，配 --ref-eval-log）")
    ap.add_argument("--ref-margin", type=float, default=0.10, help="参考线裕量（默认 0.10）")
    ap.add_argument("--anchors", default="", help="闭环成功率锚点 JSON（可选，仅打印）")
    ap.add_argument("--probe", default="step500", help="候选模型标识（仅打印用）")
    ap.add_argument("--out", required=True, help="输出 markdown 路径")
    a = ap.parse_args(argv)

    store = BaselineStore.load(a.baseline)
    th = PassThresholds.load(a.thresholds, require_metric="mse")
    ev = load_events_metrics(a.events)
    anchors: Dict[str, float] = {}
    if a.anchors:
        anchors = {k: float(v) for k, v in json.loads(Path(a.anchors).read_text()).items()}

    if th.config_fingerprint != store.config_fingerprint:
        print("⚠️ 阈值表指纹 != baseline 指纹（训练侧会 fail-fast）", file=sys.stderr)

    # 参考模型在 2/4/10 条轨迹下的 p75×(1+margin) 线（用于「同尺度」对照）
    ref_lines: Dict[str, Dict[int, float]] = {}
    if a.ref_eval_log and a.split_dir:
        per_traj = load_ref_per_traj(a.ref_eval_log)
        for task, ids in load_split(a.split_dir).items():
            row: Dict[int, float] = {}
            for k in (2, 4, 10):
                vals = [per_traj[t] for t in ids[:k] if t in per_traj]
                if len(vals) == k:
                    row[k] = _p75(vals) * (1.0 + a.ref_margin)
            if row:
                ref_lines[task] = row

    rows: List[Dict[str, Any]] = []
    for task in sorted(store.tasks):
        base_mse = float(store.tasks[task]["mse"])
        m = ev.get(task, {})
        scout_nmse = m.get("scout_nmse")
        scout_mse = m.get("scout_mse", (scout_nmse * base_mse) if scout_nmse else None)
        confirm_nmse = m.get("confirm_nmse")
        confirm_mse = m.get("confirm_mse", (confirm_nmse * base_mse) if confirm_nmse else None)
        line = th.tasks.get(task)
        # 旧口径：最终判定看 confirm（有则用 confirm，否则 scout）
        final_nmse = confirm_nmse if confirm_nmse is not None else scout_nmse
        final_mse = confirm_mse if confirm_mse is not None else scout_mse
        old_pass = (final_nmse is not None) and (final_nmse <= a.old_pass_nmse)
        new_pass = None if line is None else (
            (final_mse is not None) and (final_mse <= line))
        rows.append({
            "task": task, "base_mse": base_mse,
            "scout_nmse": scout_nmse, "scout_mse": scout_mse,
            "confirm_nmse": confirm_nmse, "confirm_mse": confirm_mse,
            "line": line, "old_pass": old_pass, "new_pass": new_pass,
            "old_nmse_gate": a.old_pass_nmse,
            "ref": ref_lines.get(task, {}),
            "anchor": anchors.get(task),
        })

    flips = [r for r in rows if r["new_pass"] is not None and r["old_pass"] != r["new_pass"]]
    old_n = sum(1 for r in rows if r["old_pass"])
    new_n = sum(1 for r in rows if r["new_pass"])

    # 判定「翻转是否可能只是样本数不同」：候选模型只有 2/4 条，参考线却用 10 条
    def _ref_verdict(r: Dict[str, Any], k: int) -> Optional[bool]:
        v = r["ref"].get(k)
        if v is None:
            return None
        mse = r["scout_mse"] if k <= 2 else (r["confirm_mse"] if r["confirm_mse"] is not None else r["scout_mse"])
        return None if mse is None else (mse <= v)

    out_lines: List[str] = []
    w = out_lines.append
    w(f"# 按任务 MSE 阈值 —— 改判对拍（候选 = {a.probe}）\n")
    w(f"- 阈值表: `{a.thresholds}`（stat={th.stat}, margin={th.margin:.2f}, "
      f"指纹={th.config_fingerprint}, 可用线 {th.n_usable}/{len(th.tasks)}）")
    w(f"- 旧口径: 全局 `nmse <= {a.old_pass_nmse}`")
    w(f"- 事件源: `{a.events}`\n")
    w(f"**旧口径 PASS {old_n}/{len(rows)} → 新口径 PASS {new_n}/{len(rows)}，翻转 {len(flips)} 个**\n")

    w("| 任务 | baseline_mse | scout(2) nmse/mse | confirm(4) nmse/mse | 新线(10) | 旧判 | 新判 | 翻转 | 参考线@2/4 | 闭环实测 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    def _sort_key(x: Dict[str, Any]) -> float:
        return x["scout_nmse"] if x["scout_nmse"] is not None else 9e9

    for r in sorted(rows, key=_sort_key):
        ref = r["ref"]
        ref_s = "/".join(
            ("—" if ref.get(k) is None else f"{ref[k]:.4f}") for k in (2, 4))
        w("| {t} | {b:.5f} | {sn} / {sm} | {cn} / {cm} | {line} | {o} | {n} | {f} | {ref} | {anc} |".format(
            t=r["task"], b=r["base_mse"],
            sn="—" if r["scout_nmse"] is None else f"{r['scout_nmse']:.4f}",
            sm="—" if r["scout_mse"] is None else f"{r['scout_mse']:.5f}",
            cn="—" if r["confirm_nmse"] is None else f"{r['confirm_nmse']:.4f}",
            cm="—" if r["confirm_mse"] is None else f"{r['confirm_mse']:.5f}",
            line="—" if r["line"] is None else f"{r['line']:.5f}",
            o="PASS" if r["old_pass"] else "no",
            n="—" if r["new_pass"] is None else ("PASS" if r["new_pass"] else "no"),
            f="🔁" if (r["new_pass"] is not None and r["old_pass"] != r["new_pass"]) else "",
            ref=ref_s,
            anc="—" if r["anchor"] is None else f"{r['anchor']:.0%}",
        ))

    w("\n## 翻转明细\n")
    if not flips:
        w("（无）")
    for r in flips:
        direction = "旧 PASS → 新 no" if r["old_pass"] else "旧 no → 新 PASS"
        w(f"- **{r['task']}**: {direction}｜scout_nmse={r['scout_nmse']} "
          f"confirm_nmse={r['confirm_nmse']}｜终判 mse="
          f"{r['confirm_mse'] if r['confirm_mse'] is not None else r['scout_mse']} vs 新线 {r['line']}")
        for k in (2, 4):
            rv = _ref_verdict(r, k)
            if rv is not None:
                w(f"    - 用参考模型同 N={k} 条轨迹的线（{r['ref'][k]:.5f}）判：{'PASS' if rv else 'no'}"
                  f"{'  ← 与翻转后一致' if rv == r['new_pass'] else '  ← 翻转可能是样本数造成的'}")
        if r["anchor"] is not None:
            w(f"    - 闭环实测锚点: {r['anchor']:.0%}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print("\n".join(out_lines[:6]))
    print(f"...\n✅ 写出 {a.out}（{len(rows)} 任务，翻转 {len(flips)}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
