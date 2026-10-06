"""结果呈现：控制台表格 + summary.json + heatmap（文档 §39–§40）。

matplotlib 是**可选**依赖：没有就只落 CSV，不报错。
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..decision.metrics import wilson_interval


# --------------------------------------------------------------------------- #
# 控制台
# --------------------------------------------------------------------------- #
def format_table(
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
    headers: Optional[Sequence[str]] = None,
    align_right: Optional[Iterable[str]] = None,
) -> str:
    headers = list(headers) if headers else [str(c) for c in columns]
    right = set(align_right or [])

    def cell(row: Dict[str, Any], col: str) -> str:
        v = row.get(col)
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"
        if isinstance(v, bool):
            return "Y" if v else "n"
        return str(v)

    widths = []
    for col, head in zip(columns, headers):
        w = max(len(head), max((len(cell(r, col)) for r in rows), default=0))
        widths.append(w)

    def line(cells: Sequence[str]) -> str:
        parts = []
        for c, w, col in zip(cells, widths, columns):
            parts.append(c.rjust(w) if col in right else c.ljust(w))
        return "  ".join(parts).rstrip()

    out = [line(headers), line(["-" * w for w in widths])]
    for r in rows:
        out.append(line([cell(r, c) for c in columns]))
    return "\n".join(out)


def format_event(event: Dict[str, Any]) -> Optional[str]:
    """把一个原子动作渲染成一行人类可读的 trace。"""
    action = event.get("action")
    if action == "bootstrap":
        tag = {
            "pass": "PASS",
            "candidate": "cand",
            "confirm_failed": "cand*",
            "metric_invalid": "INVALID",
        }.get(event.get("result", ""), "?")
        extra = ""
        if "confirm_nmse" in event:
            extra = f" → confirm4={event['confirm_nmse']:.4f}"
        note = f"  [{event['note']}]" if event.get("note") else ""
        return (
            f"[scout ] {event['task']:<16s} nmse2={_fmt(event.get('scout_nmse'))}"
            f"{extra}  ⇒ {tag}{note}"
        )
    if action == "select":
        return (
            f"[select] {event['task']:<16s} attempt={event['attempt']} round={event['round']} "
            f"scout={_fmt(event.get('scout_nmse'))} baseline_val={_fmt(event.get('baseline_val_nmse'))} "
            f"| hardness v{event.get('hardness_version')} "
            f"扫{event.get('hardness_scanned_trajs')}条/{event.get('n_samples')}帧 "
            f"({event.get('hardness_coverage')})"
        )
    if action == "train_unit":
        tail = "" if event["decision"] == "CONTINUE" else f"  ← {_short(event.get('reason'))}"
        return (
            f"[unit  ] step={event['step']:<5d} {event['task']:<16s} "
            f"loss={event['loss']:.4f} T={_fmt(event['train_nmse'])} V={_fmt(event['val_nmse'])} "
            f"LP50={_fmt(event['lp50'])} {'OVERFIT ' if event.get('overfit') else ''}"
            f"⇒ {event['decision']}{tail}"
        )
    if action == "round_rollover":
        return f"[round ] → round {event['round']}，{len(event['promoted'])} 个 DEFER 重新进入候选: {event['promoted']}"
    if action == "review":
        parts = []
        for o in event["outcomes"]:
            mark = o["action"]
            parts.append(f"{o['task']}({mark})")
        return f"[review] 复查 {event['reviewed']} 个 PASS 任务: " + ", ".join(parts)
    if action == "finish":
        return f"[done  ] stop_reason={event.get('stop_reason')}"
    return None


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _short(text: Any, width: int = 62) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= width else s[: width - 1] + "…"


# --------------------------------------------------------------------------- #
# 文件
# --------------------------------------------------------------------------- #
def write_summary(outdir: str, payload: Dict[str, Any]) -> str:
    path = os.path.join(outdir, "summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def write_registry_csv(outdir: str, rows: Sequence[Dict[str, Any]]) -> str:
    path = os.path.join(outdir, "registry.csv")
    _write_csv(path, rows)
    return path


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------- #
# Heatmap（文档 §39：纵轴 task / 横轴 evaluation event / 颜色 NMSE）
# --------------------------------------------------------------------------- #
def build_heatmap_matrix(
    heatmap_rows: Sequence[Dict[str, Any]],
    tasks: Sequence[str],
    max_events: int = 120,
):
    """把稀疏的 eval 事件前向填充成稠密矩阵。返回 (tasks, event_labels, matrix)。"""
    events = sorted({int(r["event"]) for r in heatmap_rows})
    if len(events) > max_events:
        step = len(events) / max_events
        events = [events[int(i * step)] for i in range(max_events)]
        events = sorted(set(events))
    index = {e: i for i, e in enumerate(events)}
    last: Dict[str, Optional[float]] = {t: None for t in tasks}
    matrix: List[List[Optional[float]]] = []
    by_event: Dict[int, Dict[str, float]] = {}
    for r in heatmap_rows:
        e = int(r["event"])
        if e not in index:
            continue
        nm = r.get("nmse")
        if nm is not None:
            by_event.setdefault(e, {})[r["task"]] = float(nm)
    for e in events:
        for t, v in (by_event.get(e) or {}).items():
            last[t] = v
        matrix.append([last[t] for t in tasks])
    labels = [str(e) for e in events]
    return list(tasks), labels, matrix


def write_heatmap_csv(outdir: str, tasks, labels, matrix) -> str:
    """matrix 是 (event × task)；CSV 写成 (task × event)，方便人直接看某个任务的曲线。"""
    path = os.path.join(outdir, "heatmap.csv")
    cols = list(zip(*matrix)) if matrix else []
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task"] + list(labels))
        for t, col in zip(tasks, cols):
            w.writerow([t] + ["" if v is None else f"{v:.6f}" for v in col])
    return path


CJK_FONT_CANDIDATES = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "STHeiti",
    "Arial Unicode MS",
    "Songti SC",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "SimHei",
    "Microsoft YaHei",
]


def _setup_font(plt) -> None:  # pragma: no cover - 取决于环境
    """尽量让图里的中文能显示；找不到就静默回退（不报错、不刷 warning）。"""
    try:
        import matplotlib
        import matplotlib.font_manager as fm

        available = {f.name for f in fm.fontManager.ttflist}
        picked = [n for n in CJK_FONT_CANDIDATES if n in available]
        if picked:
            matplotlib.rcParams["font.sans-serif"] = picked + ["sans-serif"]
            matplotlib.rcParams["axes.unicode_minus"] = False
        else:
            import warnings

            warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    except Exception:
        pass


def plot_all(
    outdir: str,
    metrics_rows: Sequence[Dict[str, Any]],
    heatmap_rows: Sequence[Dict[str, Any]],
    tasks: Sequence[str],
    pass_nmse: float,
    transitions: Sequence[Dict[str, Any]] = (),
) -> List[str]:
    """画三张图：训练/评测曲线、覆盖率曲线、NMSE heatmap。matplotlib 缺失时返回 []。"""
    try:  # pragma: no cover - 取决于环境
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _setup_font(plt)
    except Exception:
        return []

    paths: List[str] = []
    paths.append(_plot_curves(outdir, metrics_rows, pass_nmse, plt))
    paths.append(_plot_coverage(outdir, metrics_rows, len(tasks), plt))
    paths.append(_plot_heatmap(outdir, heatmap_rows, tasks, pass_nmse, transitions, plt))
    return [p for p in paths if p]


def _plot_curves(outdir, metrics_rows, pass_nmse, plt) -> str:  # pragma: no cover
    steps = [r["step"] for r in metrics_rows]
    loss = [r["loss"] for r in metrics_rows]
    val = [r["val_nmse"] for r in metrics_rows]
    train = [r["train_nmse"] for r in metrics_rows]

    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(steps, loss, color="#c0392b", lw=1.4, label="batch flow loss (fake)")
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("loss", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax2 = ax1.twinx()
    ax2.plot(steps, train, color="#2980b9", lw=1.2, alpha=0.8, label="train-monitor NMSE")
    ax2.plot(steps, val, color="#27ae60", lw=1.4, label="val NMSE")
    ax2.axhline(pass_nmse, color="#7f8c8d", ls="--", lw=1, label=f"pass_nmse={pass_nmse}")
    ax2.set_ylabel("NMSE")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)
    ax1.set_title(
        "Stage A demo — current skill: loss vs open-loop NMSE\n"
        "（NMSE 曲线在任务切换处会跳变：不同任务的起点/上限本来就不同）",
        fontsize=10,
    )
    fig.tight_layout()
    path = os.path.join(outdir, "curve_loss_nmse.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_coverage(outdir, metrics_rows, n_tasks, plt) -> str:  # pragma: no cover
    steps = [r["step"] for r in metrics_rows]
    cov = [r.get("coverage") for r in metrics_rows]
    if not steps or all(c is None for c in cov):
        return ""
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(steps, [c if c is not None else 0 for c in cov], color="#8e44ad", lw=1.6)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("coverage (PASS / N)")
    ax.set_title(f"Skill coverage over training ({n_tasks} tasks)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(outdir, "curve_coverage.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


TRANSITION_COLORS = {
    "PASS": "#1e8449",
    "DEFER": "#d68910",
    "DEFER_OVERFIT": "#b9770e",
    "EXHAUSTED": "#922b21",
    "REOPEN": "#2471a3",
}


def _plot_heatmap(outdir, heatmap_rows, tasks, pass_nmse, transitions, plt) -> str:  # pragma: no cover
    import numpy as np

    tasks, labels, matrix = build_heatmap_matrix(heatmap_rows, tasks)
    if not matrix:
        return ""
    # build_heatmap_matrix 返回 (event × task)，画图要的是 (task × event)
    arr = np.array([[np.nan if v is None else v for v in row] for row in matrix], dtype=float).T

    fig, ax = plt.subplots(
        figsize=(max(10.0, 0.16 * len(labels)), max(4.0, 0.30 * len(tasks) + 1.4))
    )
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=8)
    ax.set_ylabel("task")

    stride = max(1, len(labels) // 20)
    ticks = list(range(0, len(labels), stride))
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("evaluation event")
    ax.set_title("NMSE heatmap（绿=好 / 红=差；白=尚未评估）")
    fig.colorbar(im, ax=ax, label="NMSE", fraction=0.025, pad=0.01)

    # 任务状态迁移标记（文档 §39 要求标 SELECTED / PASS / DEFER / EXHAUSTED / REOPEN）
    ev_to_col = {int(e): i for i, e in enumerate(labels)}
    ev_step = {}
    for r in heatmap_rows:
        ev = int(r["event"])
        if ev in ev_to_col and ev not in ev_step:
            ev_step[ev] = int(r.get("step") or 0)
    ordered = sorted(ev_step.items(), key=lambda kv: kv[1])
    if transitions and len(labels) > 1:
        seen = set()
        for t in transitions:
            kind = str(t.get("kind", ""))
            color = TRANSITION_COLORS.get(kind)
            if not color:
                continue
            target = int(t.get("step") or 0)
            # 找最后一个 step <= target 的评测事件
            pick = None
            for ev, st in ordered:
                if st <= target:
                    pick = ev
                else:
                    break
            if pick is None:
                pick = ordered[0][0]
            col = ev_to_col[pick]
            label = kind if kind not in seen else None
            seen.add(kind)
            ax.axvline(col, color=color, lw=0.9, alpha=0.7, label=label)
            ax.plot([col], [-0.75], marker="v", color=color, ms=5, clip_on=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, fontsize=7, frameon=False)

    fig.tight_layout()
    path = os.path.join(outdir, "heatmap_nmse.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def comparison_table(auto_summary: Dict[str, Any], baselines: Sequence[Any], n_tasks: int) -> str:
    rows = [
        {
            "method": "autonomous",
            "steps": auto_summary["global_step"],
            "pass": auto_summary["registry"]["pass"],
            "coverage": auto_summary["registry"]["coverage"],
            "median_nmse": auto_summary["median_val_nmse"],
            "worst_nmse": auto_summary["worst_val_nmse"],
            "forgotten": auto_summary["registry"].get("forgotten", 0),
            "note": f"训练了 {len(auto_summary['trained_tasks'])} 个任务 / {auto_summary['rounds']} 轮",
        }
    ]
    for b in baselines:
        s = b.summary()
        rows.append(
            {
                "method": f"uniform:{s['mode']}",
                "steps": s["global_step"],
                "pass": s["pass"],
                "coverage": s["coverage"],
                "median_nmse": s["median_nmse"],
                "worst_nmse": s["worst_nmse"],
                "forgotten": 0,
                "note": "无优先级 / 无 replay / 无遗忘检测",
            }
        )
    for r in rows:
        lo, hi = wilson_interval(int(round(r["coverage"] * n_tasks)), n_tasks)
        r["coverage_ci"] = f"[{lo:.0%},{hi:.0%}]"
    text = format_table(
        rows,
        ["method", "steps", "pass", "coverage", "coverage_ci", "median_nmse", "worst_nmse", "note"],
        ["方法", "steps", "PASS", "覆盖率", "95%CI", "中位NMSE", "最差NMSE", "说明"],
    )
    return text
