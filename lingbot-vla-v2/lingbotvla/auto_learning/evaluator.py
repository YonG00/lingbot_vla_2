"""EvaluatorAdapter —— 把 `OpenLoopValidator` 包成 `ports.Evaluator`。

🔴 三条红线
-----------
1. **不重写** `sample_actions()` glue / preprocess / chunk aggregation / metric pipeline
   —— 全部复用 `OpenLoopValidator`。
2. **不裸调 `_evaluate_ids()`** —— 走 `validator.evaluate_ids()`，
   它内部包着 `safe_eval_context`（RNG / training flag / use_cache / attention /
   视觉网格缓存 的 snapshot-restore-审计）。
3. **tag 必须带 episode ids 指纹**（review v0.1 #1）：
   `_episode_ids_file(ids, tag)` 是**按 tag 落同一个路径**、而 `_dataset()` 又**按路径缓存**
   ⇒ 若 tag 固定为 `al_<task>_<split>`，则「2 条 scout → 4 条 confirm」的第二次会拿到
   **第一次的 2 条 dataset**（JSON 被覆盖了也没用），直接破坏 2→4 设计。

本模块懒加载重依赖，`import` 本身不需要 torch。
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, List, Optional, Sequence

from .eval_context import resolve_logger
from .ports import EvalResult


def ids_fingerprint(ids: Sequence[int]) -> str:
    """一组 episode ids 的稳定短指纹（顺序无关）。"""
    blob = ",".join(str(int(i)) for i in sorted(set(int(i) for i in ids)))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


class EvaluatorAdapter:
    """`evaluate_task(task_id, split, episode_ids) -> EvalResult`。

    典型用法::

        adapter = EvaluatorAdapter(validator, catalog, store)
        scout   = adapter.evaluate_task("click_bell", "val", ids[:2])
        confirm = adapter.evaluate_task("click_bell", "val", ids[:4])   # 真的是 4 条
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
        require_baseline: bool = False,
    ):
        self.validator = validator
        self.catalog = catalog
        self.baseline_store = baseline_store
        self.logger = resolve_logger(logger)
        self.tag_prefix = tag_prefix
        # True（默认）：传入的 episode_ids 必须**属于**声明的 split ⇒ 防 train/val 泄漏
        self.verify_split = verify_split
        # True：没有有效 fixed baseline 时**直接报错**（Scheduler 启动时应打开，
        # 免得 NMSE 静默变成 None —— review v0.1 #5）
        self.require_baseline = require_baseline

    # -- 内部 ---------------------------------------------------------------
    def _clear_dataset_cache(self) -> None:
        """清掉 validator 的数据集缓存（review v0.2 #5）。

        * validator 没有该接口 ⇒ 直接返回（测试替身 / 旧版本）；
        * 清理本身失败 ⇒ 只 warning，**不打断评测**（缓存多留一份不是正确性问题）。
        """
        clear = getattr(self.validator, "clear_dataset_cache", None)
        if clear is None:
            return
        try:
            freed = clear()
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"[auto_learning] clear_dataset_cache 失败（已忽略）: {e}")
            return
        if freed:
            self.logger.info_rank0(f"[auto_learning] 已释放数据集缓存 {freed} 份")

    def _baseline_for(self, task_id: str):
        if self.baseline_store is None:
            return None
        entry = self.catalog.entry(task_id)
        b = self.baseline_store.get(task_id, entry.sha256_train)
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

        baseline = self._baseline_for(task_id)
        if baseline is None and self.require_baseline:
            raise RuntimeError(
                f"[{task_id}] 没有可用的 fixed baseline ⇒ 拒绝评测（require_baseline=True）。\n"
                f"  先跑：python -m lingbotvla.auto_learning.tools.compute_task_baseline "
                f"--manifest <manifest.json> --config <lingbotvla_cli.yaml>")

        # 🔴 ids 指纹进 tag：否则 2 条 scout 与 4 条 confirm 会共用同一个
        #    episode_ids 文件路径 ⇒ `_dataset()` 按路径缓存 ⇒ 第二次拿到旧的 2 条。
        tag = f"{self.tag_prefix}_{task_id}_{split}_{ids_fingerprint(ids)}"
        t0 = time.time()
        # 🔴 走 evaluate_ids（内含 safe_eval_context），**不要**直接调 _evaluate_ids
        try:
            raw = self.validator.evaluate_ids(ids, tag)
        finally:
            # 🔴 review v0.2 #5：`OpenLoopValidator` 的数据集缓存**按 tag 缓存、只增不减**，
            #    而 tag 现在含 ids 指纹 ⇒ 正式长跑（50 任务 × scout2/confirm4/active…）
            #    会不断产生新 key、内存持续增长。
            #    ⇒ 每次评测后清掉；`evaluate_ids` 是 safe_eval_context 的**外层**，
            #    走到这里 context 已经退出，清缓存不会污染评测状态。
            #    （若将来 rebuild dataset 太慢，再改成小的 LRU，**不要**无限缓存。）
            self._clear_dataset_cache()
        elapsed = time.time() - t0

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
            action_keys=list(raw.get("action_keys", [])),
            baseline_fingerprint=(baseline.fingerprint if baseline is not None else None),
        )
        self.logger.info_rank0(
            f"[auto_learning] eval {task_id}/{split} n_ids={len(ids)}: mse={res.mse:.6f} "
            f"nmse={('%.4f' % res.nmse) if res.nmse is not None else 'n/a'} "
            f"({res.n_traj}轨迹/{res.n_chunks}chunk, {res.eval_seconds:.1f}s)")
        return res


__all__ = ["EvaluatorAdapter", "ids_fingerprint"]
