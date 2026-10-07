"""用**真实组件**组装 `ports.Backend`（Stage B1）。

关键约定（见 `docs/stage_b1_integration_design.md` §3）
----------------------------------------------------
* 训练数据集**只建一次**，白名单 = 所有参与任务的 **train 回合并集**
  ⇒ 索引空间全程不变；`sample_id` == dataset local_idx
* `TaskCatalog.attach_samples(resolver)` 把回合展开成 local_idx
* `Evaluator` 走 `EvaluatorAdapter`（内含 `safe_eval_context`）
* `HardnessScorer` 按 `(task, sample_ids)` 打分 —— **用训练数据集**取 item，
  否则 local_idx 对不上（子集数据集的索引是局部的）
* `Trainer` 在**延迟训练模式**下**永不被调用**（训练由外层真实循环跑），
  但 `ports.Backend.missing()` 仍要求它有 `train_steps` ⇒ 给一个显式报错的桩

⚠️ 本模块只 import `ports` / `types` / `testing`（假件），**torch 相关一律懒加载**。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..ports import Backend, TrajectoryMetrics
from .determinism import deterministic_sampling


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
class RealEvaluator:
    """把 `EvaluatorAdapter` 包成 `ports.Evaluator`（Scheduler 只认 `evaluate`）。"""

    def __init__(self, adapter: Any, catalog: Any, baseline_store: Any = None):
        self.adapter = adapter
        self.catalog = catalog
        self.baseline_store = baseline_store

    def evaluate(self, task: str, split: str,
                 episode_ids: Sequence[int]) -> TrajectoryMetrics:
        """`split` 用 Stage A 的 `EvalSplit` 值（scout/confirm/active_val/train_monitor/review）。

        🔴 **不做 scout→confirm 的自动扩张**：Stage A 的 Scheduler 已经按
        `EvalSplit` 传了**它认为该评的回合集合**，这里只负责「评这个集合」。
        """
        res = self.adapter.evaluate_task(task, _split_name(split), list(episode_ids))
        return res.to_trajectory_metrics()

    def baseline_mse(self, task: str) -> float:
        if self.baseline_store is None:
            return 0.0
        entry = self.catalog.entry(task)
        b = self.baseline_store.get(task, entry.sha256_train)
        return float(b.mse) if b is not None else 0.0


def _split_name(split: Any) -> str:
    """把 `EvalSplit` / 字符串统一成 evaluator 认的 split 名。

    ⚠️ `EvaluatorAdapter.evaluate_task` 的 `split` 必须落在 manifest 的
    `train`/`val` 里（它要校验「不跨 split 泄漏」）；Stage A 的 `EvalSplit` 更细，
    这里按「train_monitor → train，其余 → val」映射。
    """
    name = getattr(split, "value", split)
    return "train" if str(name) == "train_monitor" else "val"


# --------------------------------------------------------------------------- #
# Hardness
# --------------------------------------------------------------------------- #
class RealHardnessScorer:
    """把真实 `HardnessScorer`（per-sample L1_fm）包成 `ports.HardnessScorer`。

    ``dataset`` 必须是**训练数据集**（与 `sample_id` 同一索引空间）。
    """

    def __init__(self, scorer: Any, dataset: Any, *, max_batch: int = 8, logger: Any = None):
        self.scorer = scorer
        self.dataset = dataset
        self.max_batch = int(max_batch)
        self.logger = logger
        self._aug_warned = False

    def score(self, task: str, sample_ids: Sequence[int]) -> Dict[int, float]:
        ids = [int(s) for s in sample_ids]
        if not ids:
            return {}
        out: Dict[int, float] = {}
        for i in range(0, len(ids), self.max_batch):
            chunk = ids[i:i + self.max_batch]
            # 🔴 review v0.2 #8：取 item **必须**在确定性上下文里 ——
            #    否则 `image_augment=true` 会让同一 sample_id 两次扫描得到不同图，
            #    并消耗全局 RNG（污染训练随机流）。
            with deterministic_sampling(self.dataset) as rep:
                if rep["n_ft"] == 0 and not self._aug_warned:
                    self._aug_warned = True
                    if self.logger is not None:
                        try:
                            self.logger.warning(
                                "[auto_learning][hardness] 没能定位到 feature_transform ⇒ "
                                "无法临时关闭图像增强；若数据集开了 image_augment，"
                                "hardness 分数将不可复现。")
                        except Exception:  # noqa: BLE001
                            pass
                items = [self.dataset[j] for j in chunk]
            vals = self.scorer.score(items)
            for sid, v in zip(chunk, vals):
                out[sid] = float(v)
        return out


# --------------------------------------------------------------------------- #
# Trainer（延迟模式下不会被调用）
# --------------------------------------------------------------------------- #
class RealTrainerStub:
    """`defer_train=True` 时 Scheduler **不会**调它；这里显式报错而不是静默。"""

    def train_steps(self, request, num_steps):
        raise RuntimeError(
            "RealTrainer.train_steps() 不该被调用：Auto Learning 走**延迟训练模式**"
            "（`scheduler.defer_train=True`），训练由外层真实循环跑，"
            "结果经 `AutoLearnLoopHook` 回填。")


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #
def build_real_backend(
    *,
    catalog: Any,
    resolver: Any,
    adapter: Any,
    hardness: Any,
    dataset: Any = None,
    baseline_store: Any = None,
) -> Backend:
    """组装真实 `Backend`。

    ``catalog`` 必须已经 `attach_samples(resolver)` 过（否则 `train_sample_ids` 为空）。
    """
    if not getattr(catalog.entry(catalog.task_names()[0]), "train_sample_ids", None):
        raise ValueError(
            "catalog 还没 attach_samples()：TaskEntry.train_sample_ids 为空，"
            "Sampler 无法工作。先调 catalog.attach_samples(resolver)。")
    return Backend(
        catalog=catalog,
        resolver=resolver,
        evaluator=RealEvaluator(adapter, catalog, baseline_store),
        scorer=hardness,
        trainer=RealTrainerStub(),
        baseline=baseline_store,
    )


def build_real_hardness(model: Any, dataset: Any, *, logger: Any = None, **kw) -> RealHardnessScorer:
    """便捷构造：真实 `HardnessScorer` + 训练数据集。"""
    from ..hardness import HardnessScorer

    return RealHardnessScorer(HardnessScorer(model, **kw), dataset, logger=logger)


__all__ = ["RealEvaluator", "RealHardnessScorer", "RealTrainerStub",
           "build_real_backend", "build_real_hardness"]
