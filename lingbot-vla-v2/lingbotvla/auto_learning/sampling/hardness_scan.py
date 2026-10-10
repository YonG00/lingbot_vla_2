"""Hardness 扫描：样本层「难易」→ 采样概率（文档 §19–§26）。

分工（很重要）：
    任务层能力  →  open-loop NMSE（贵，用真 `sample_actions()`）
    样本层难度  →  per-sample flow loss（便宜，只是 proxy）

本模块只做**纯逻辑**（抽子集 → 百分位 → 权重 → 归一化）；
「难度从哪来」由 `ports.HardnessScorer` 提供 —— Stage A 是假 world 的
`sample_loss`，Stage B 是 `return_per_sample_loss` 的训练/评测路径。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from ..config import AutoLearningConfig
from ..decision.metrics import difficulty_to_weight, normalize_weights, percentile_rank
from ..ports import HardnessScorer, TaskEntry
from ..state.registry import TaskRecord


def uniform_cap_frames(samples: Sequence[int], cap: int) -> List[int]:
    """按 `np.linspace(0, n-1, k)` **均匀抽样**把一条轨迹的帧压到 ≤ `cap`。

    用户 2026-10-10 定：`if n > cap: idx = np.linspace(0, n - 1, cap).astype(int)`。
    ⇒ 保序、含首尾、等间隔；`cap` 为 0/负数或 `n <= cap` 时**原样返回**（不复制）。

    ⚠️ 抽样**改变难度分数**（扫的帧变了）——这是需求本身的取舍，不是 bug。
    """
    k = int(cap)
    if k <= 0 or len(samples) <= k:
        return list(samples)
    import numpy as np  # 局部导入：本模块其余部分是纯逻辑，不必强依赖 numpy

    idx = np.linspace(0, len(samples) - 1, k).astype(int)
    return [int(samples[i]) for i in idx]


@dataclass
class HardnessScan:
    """一次扫描的完整产物（可持久化，供 resume / 复盘）。"""

    task: str
    version: int
    scanned_traj_ids: List[int]
    scanned_sample_ids: List[int]
    losses: Dict[int, float]
    difficulties: Dict[int, float]
    weights: Dict[int, float]
    probs: Dict[int, float]
    n_total: int
    n_scanned: int
    unscored_difficulty: float
    mean_loss_scanned: float = 0.0
    p90_loss_scanned: float = 0.0

    @property
    def coverage(self) -> float:
        return self.n_scanned / max(1, self.n_total)

    def to_state(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "version": self.version,
            "scanned_traj_ids": self.scanned_traj_ids,
            "losses": {str(k): v for k, v in self.losses.items()},
            "difficulties": {str(k): v for k, v in self.difficulties.items()},
            "weights": {str(k): v for k, v in self.weights.items()},
            "probs": {str(k): v for k, v in self.probs.items()},
            "n_total": self.n_total,
            "n_scanned": self.n_scanned,
            "unscored_difficulty": self.unscored_difficulty,
        }

    @classmethod
    def from_state(cls, raw: Dict[str, Any]) -> "HardnessScan":
        losses = {int(k): float(v) for k, v in (raw.get("losses") or {}).items()}
        diffs = {int(k): float(v) for k, v in (raw.get("difficulties") or {}).items()}
        weights = {int(k): float(v) for k, v in (raw.get("weights") or {}).items()}
        probs = {int(k): float(v) for k, v in (raw.get("probs") or {}).items()}
        vals = sorted(losses.values())
        return cls(
            task=raw["task"],
            version=int(raw.get("version", 1)),
            scanned_traj_ids=list(raw.get("scanned_traj_ids") or []),
            scanned_sample_ids=sorted(losses.keys()),
            losses=losses,
            difficulties=diffs,
            weights=weights,
            probs=probs,
            n_total=int(raw.get("n_total", len(probs))),
            n_scanned=int(raw.get("n_scanned", len(losses))),
            unscored_difficulty=float(raw.get("unscored_difficulty", 0.5)),
            mean_loss_scanned=(sum(vals) / len(vals)) if vals else 0.0,
            p90_loss_scanned=(vals[int(0.9 * (len(vals) - 1))] if vals else 0.0),
        )


class HardnessScanner:
    """方案 B（文档 §20）：抽约 1/3 的**完整轨迹**，扫这些轨迹的所有有效 frame。

    比「每条轨迹只取代表帧」可靠；比全量便宜。
    """

    def __init__(self, cfg: AutoLearningConfig, scorer: HardnessScorer) -> None:
        self.cfg = cfg
        self.scorer = scorer

    # ---------------------------------------------------------------- #
    def probe_traj_ids(self, record: TaskRecord) -> List[int]:
        """按真实比例抽 trajectory，并在多次 attempt 之间**轮换**。

        `k = ceil(n_traj × fraction)`（**不是** `round(1/fraction)` ——
        后者在 0.4 / 0.6 这类取值上会偏成 0.5）。

        轮换方式：把 `n_traj` 切成 `n_windows = ceil(n_traj / k)` 个长度恰为 `k` 的
        滑动窗口，第 `hardness_version` 次扫描用第 `version % n_windows` 个。

        ⇒ 每次抽到的条数**恰好等于配置比例**；`n_windows` 次扫描覆盖全部轨迹。
        ⚠️ 只有 `fraction ≤ 0.5` 时连续两次才**严格零重合**（`2k ≤ n`）；
        `fraction > 0.5` 时相邻窗口必然有交叠 —— 这是取整后的必然结果，不是 bug。
        """
        trajs = list(record.train_traj_ids)
        n = len(trajs)
        if n == 0:
            return []
        k = max(1, min(n, int(math.ceil(n * self.cfg.hardness_probe_fraction))))
        n_windows = max(1, int(math.ceil(n / k)))
        start = (record.hardness_version % n_windows) * k
        return sorted(trajs[(start + i) % n] for i in range(k))

    # ---------------------------------------------------------------- #
    def scan(self, record: TaskRecord, entry: TaskEntry) -> HardnessScan:
        cfg = self.cfg
        chosen = self.probe_traj_ids(record)

        scanned_ids: List[int] = []
        # 🔴 帧数上限（2026-10-10 用户定）：轨迹帧数 > `hardness_max_frames_per_traj` 时，
        #    用 `np.linspace(0, n-1, cap)` 均匀抽样压到 ≤ cap（保序、含首尾）。
        #    动机：任务间扫描量差异极大（259 vs 1315 个样本 ⇒ 最慢单任务 ETA ~115 分钟）。
        _cap = int(getattr(cfg, 'hardness_max_frames_per_traj', 0) or 0)
        _frames_before = 0
        _capped_trajs = 0
        for t in chosen:
            _frames = list(entry.samples_by_traj.get(t, []))
            _frames_before += len(_frames)
            if _cap > 0 and len(_frames) > _cap:
                _capped_trajs += 1
            scanned_ids.extend(uniform_cap_frames(_frames, _cap))
        if not scanned_ids:
            raise RuntimeError(
                f"任务 {record.task_name} 的扫描子集为空（检查 TaskEntry.samples_by_traj）"
            )
        _frames_after = len(scanned_ids)

        # 🔴 外层可见性（2026-10-10 加）：`scan()` 是 `Scheduler._select()` 里最贵的一步，
        #    而它自身（抽样比例、帧数、总时长）此前**完全没有日志**，只有内层 scorer 会打点。
        #    这里给出"扫什么/扫多少/多久"，让"日志不动"能被立刻区分为「在扫」或「真卡死」。
        _scan_t0 = time.perf_counter()
        _scan_log = getattr(self.scorer, 'logger', None)

        def _say(msg: str) -> None:
            if _scan_log is None:
                return
            try:
                _scan_log.info_rank0(msg)
            except Exception:  # noqa: BLE001 —— 日志失败不该打断扫描
                pass

        _say(f'[hardness] 扫描开始：task={record.task_name} '
             f'轨迹 {len(chosen)}/{len(record.train_traj_ids)} 条'
             f'（fraction={cfg.hardness_probe_fraction}）'
             f' ⇒ 待打分样本 {len(scanned_ids)}/该任务 {len(record.sample_ids)} 帧'
             + (f'（帧数上限 {_cap}：{_capped_trajs} 条轨迹被均匀压缩，'
                f'{_frames_before} → {_frames_after} 帧）' if _cap > 0 and _capped_trajs else ''))

        scored = self.scorer.score(record.task_name, scanned_ids)
        losses = {int(s): float(v) for s, v in scored.items()}
        order = list(losses.keys())
        ranks = percentile_rank([losses[s] for s in order])
        difficulties = {s: r for s, r in zip(order, ranks)}

        _vals = sorted(losses.values())
        _say(f'[hardness] 扫描完成：task={record.task_name} 打分 {len(losses)}/{len(scanned_ids)} 个样本'
             f' 用时 {time.perf_counter() - _scan_t0:.1f}s'
             + (f' | loss 均值 {sum(_vals) / len(_vals):.4f} p90 {_vals[int(0.9 * (len(_vals) - 1))]:.4f}'
                if _vals else ''))

        # 未扫描样本 → 默认中等难度（文档 §21），避免「没扫到 = 永远没机会训练」
        default_w = difficulty_to_weight(
            cfg.difficulty_unscored_default,
            cfg.hardness_weight_min,
            cfg.hardness_weight_max,
            cfg.hardness_alpha,
        )
        weights: Dict[int, float] = {}
        for sid in record.sample_ids:
            if sid in difficulties:
                weights[sid] = difficulty_to_weight(
                    difficulties[sid],
                    cfg.hardness_weight_min,
                    cfg.hardness_weight_max,
                    cfg.hardness_alpha,
                )
            else:
                weights[sid] = default_w

        probs = normalize_weights(weights)
        vals = sorted(losses.values())
        return HardnessScan(
            task=record.task_name,
            version=record.hardness_version + 1,
            scanned_traj_ids=chosen,
            scanned_sample_ids=sorted(scanned_ids),
            losses=losses,
            difficulties=difficulties,
            weights=weights,
            probs=probs,
            n_total=len(record.sample_ids),
            n_scanned=len(scanned_ids),
            unscored_difficulty=cfg.difficulty_unscored_default,
            mean_loss_scanned=(sum(vals) / len(vals)) if vals else 0.0,
            p90_loss_scanned=(vals[int(0.9 * (len(vals) - 1))] if vals else 0.0),
        )
