"""EvaluatorAdapter —— 把 `OpenLoopValidator` 包成 `ports.Evaluator`。

🔴 两条红线
-----------
1. **不重写** `sample_actions()` glue / preprocess / chunk aggregation / metric pipeline
   —— 全部复用 `OpenLoopValidator`。
2. **不裸调 `_evaluate_ids()`** —— 走 `validator.evaluate_ids()`，
   它内部包着 `safe_eval_context`（RNG / training flag / use_cache / attention /
   视觉网格缓存 的 snapshot-restore-审计）。

本模块懒加载重依赖，`import` 本身不需要 torch。
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Sequence

from .eval_context import resolve_logger
from .ports import EvalResult


class EvaluatorAdapter:
    """`evaluate_task(task_id, split, episode_ids) -> EvalResult`。

    典型用法::

        adapter = EvaluatorAdapter(validator, catalog, store)
        r = adapter.evaluate_task("click_bell", "val")
        r.mse, r.nmse, r.per_traj_mse, r.eval_seconds
    """

    def __init__(
        self,
        validator: Any,
        catalog: Any,
        baseline_store: Any = None,
        *,
        logger: Any = None,
        tag_prefix: str = "al",
        verify_split: bool = True,
    ):
        self.validator = validator
        self.catalog = catalog
        self.baseline_store = baseline_store
        self.logger = resolve_logger(logger)
        self.tag_prefix = tag_prefix
        # True（默认）：传入的 episode_ids 必须**属于**声明的 split ⇒ 防 train/val 泄漏
        self.verify_split = verify_split

    # -- 内部 ---------------------------------------------------------------
    def _baseline_for(self, task_id: str):
        if self.baseline_store is None:
            return None
        b = self.baseline_store.get(task_id)
        if b is None and self.baseline_store.warnings:
            self.logger.warning(f"[auto_learning] {self.baseline_store.warnings[-1]}")
        return b

    def _check_ids_in_split(self, entry, split: str, ids: Sequence[int]) -> None:
        if not self.verify_split:
            return
        allowed = set(entry.ids_for(split))
        bad = [int(i) for i in ids if int(i) not in allowed]
        if bad:
            raise ValueError(
                f"[{entry.name}] 传入的 episode_ids 有 {len(bad)} 个**不属于** {split} split: "
                f"{bad[:5]} ⇒ 拒绝评测（防 train/val 泄漏）")

    # -- 公开接口 -----------------------------------------------------------
    def evaluate_task(
        self,
        task_id: str,
        split: str,
        episode_ids: Optional[Sequence[int]] = None,
    ) -> EvalResult:
        """在 `task_id` 的某个 split 上跑一次开环评测。

        ``episode_ids=None`` ⇒ 用 manifest 里的**全部**该 split 回合。
        """
        entry = self.catalog.entry(task_id)
        ids: List[int] = (
            [int(i) for i in episode_ids] if episode_ids is not None else entry.ids_for(split)
        )
        if not ids:
            raise ValueError(f"[{task_id}] {split} split 的回合列表为空，无法评测")
        self._check_ids_in_split(entry, split, ids)

        tag = f"{self.tag_prefix}_{task_id}_{split}"
        t0 = time.time()
        # 🔴 走 evaluate_ids（内含 safe_eval_context），**不要**直接调 _evaluate_ids
        raw = self.validator.evaluate_ids(ids, tag)
        elapsed = time.time() - t0

        baseline = self._baseline_for(task_id)
        b_mse = float(baseline.mse) if baseline is not None else None
        nmse = (float(raw["mse"]) / b_mse) if (b_mse and b_mse > 0) else None

        res = EvalResult(
            task=task_id, split=split, episode_ids=ids,
            mse=float(raw["mse"]),
            nmse=nmse,
            baseline_mse=b_mse,
            mae=float(raw.get("mae", float("nan"))),
            per_traj_mse=list(raw.get("per_traj_mse", [])),
            per_traj_ids=list(raw.get("per_traj_ids", [])),
            per_traj_frames=list(raw.get("per_traj_frames", [])),
            n_traj=int(raw.get("n", 0)),
            n_chunks=int(raw.get("n_chunks", 0)),
            frames=int(raw.get("frames", 0)),
            dims=int(raw.get("dims", 0)),
            eval_seconds=elapsed,
            baseline_fingerprint=(baseline.fingerprint if baseline is not None else None),
        )
        self.logger.info_rank0(
            f"[auto_learning] eval {task_id}/{split}: mse={res.mse:.6f} "
            f"nmse={('%.4f' % res.nmse) if res.nmse is not None else 'n/a'} "
            f"({res.n_traj}轨迹/{res.n_chunks}chunk, {res.eval_seconds:.1f}s)")
        return res


__all__ = ["EvaluatorAdapter"]
