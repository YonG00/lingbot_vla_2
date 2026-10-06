"""Fixed Task BaselineMSE —— task-global-mean + trajectory-balanced（只算一次）。

口径（Stage B0 已定）
---------------------
    1) 用固定 **train split**（manifest 的 ``train_ids``，默认 40 条）收集 GT chunk
       —— 与 evaluator **同一条路径**（`collect_gt_chunks`）。
    2) ``μ_task = 所有 train chunk 的所有帧、所有维的均值``        → shape (D,)
    3) 对每条 train trajectory i：``mse_i = mean((gts_i - μ_task)²)``
    4) ``baseline_mse = mean_i(mse_i)``                            → 轨迹等权

第 3–4 步**等价于**直接调用 evaluator 的纯函数 ``aggregate_chunks``：:

    aggregate_chunks([(ep_i, gts_i, broadcast_to(μ_task, gts_i.shape)) for ...])["mse"]

⇒ baseline 与 evaluator 的 ``mse`` **共用同一个聚合函数**，
action space / normalization / valid region **完全一致**。

⚠️ 因此 ``NMSE = eval_mse / baseline_mse`` 与旧的 ``1 - r2``（分母用**评测集自身**
的 pooled 方差）**不可直接比较** —— 数据来源与统计量都不同。

只依赖 `numpy`；GT 收集那一步由调用方（`OpenLoopValidator.collect_gt_chunks`）注入。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

BASELINE_VERSION = 1

# μ 的加权方式
MU_GLOBAL = "global"          # 所有帧 pooled 平均（默认；字面意义的「全局 mean action」）
MU_TRAJECTORY = "trajectory"  # 先每条轨迹求均值、再对轨迹等权平均


@dataclass(frozen=True)
class FixedBaseline:
    """一个 task 的固定分母（**只预计算一次，不随模型更新**）。"""

    task: str
    mse: float
    mu: Tuple[float, ...]
    fingerprint: str
    n_train_episodes: int = 0
    n_chunks: int = 0
    n_frames: int = 0
    dims: int = 0
    action_keys: Tuple[str, ...] = ()
    mu_weighting: str = MU_GLOBAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "mse": self.mse,
            "mu": list(self.mu),
            "fingerprint": self.fingerprint,
            "n_train_episodes": self.n_train_episodes,
            "n_chunks": self.n_chunks,
            "n_frames": self.n_frames,
            "dims": self.dims,
            "action_keys": list(self.action_keys),
            "mu_weighting": self.mu_weighting,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FixedBaseline":
        return cls(
            task=d["task"], mse=float(d["mse"]),
            mu=tuple(float(x) for x in d["mu"]),
            fingerprint=d["fingerprint"],
            n_train_episodes=int(d.get("n_train_episodes", 0)),
            n_chunks=int(d.get("n_chunks", 0)),
            n_frames=int(d.get("n_frames", 0)),
            dims=int(d.get("dims", 0)),
            action_keys=tuple(d.get("action_keys", ())),
            mu_weighting=d.get("mu_weighting", MU_GLOBAL),
        )


# --------------------------------------------------------------------------- #
# 纯计算（不依赖 torch / lerobot，可单测）
# --------------------------------------------------------------------------- #


def compute_mu(chunks: Sequence[Tuple[Any, np.ndarray]],
               *, weighting: str = MU_GLOBAL) -> np.ndarray:
    """``μ_task``：task 级共享常数动作。

    ``weighting="global"``（默认）：所有 chunk 的所有帧 pooled 平均。
    ``weighting="trajectory"``：先按 episode 求均值、再对轨迹等权平均。
    """
    if not chunks:
        raise ValueError("chunks 为空，无法计算 μ_task")
    if weighting == MU_GLOBAL:
        allgt = np.concatenate([np.asarray(gt, dtype=np.float64) for _, gt in chunks], axis=0)
        return allgt.mean(axis=0)
    if weighting == MU_TRAJECTORY:
        per_ep: Dict[Any, List[np.ndarray]] = {}
        order: List[Any] = []
        for k, gt in chunks:
            if k not in per_ep:
                per_ep[k] = []
                order.append(k)
            per_ep[k].append(np.asarray(gt, dtype=np.float64))
        means = [np.concatenate(per_ep[k], axis=0).mean(axis=0) for k in order]
        return np.mean(np.stack(means, axis=0), axis=0)
    raise ValueError(f"未知 weighting: {weighting!r}（只能是 {MU_GLOBAL!r} / {MU_TRAJECTORY!r}）")


def trajectory_balanced_baseline(
    chunks: Sequence[Tuple[Any, np.ndarray]],
    mu: np.ndarray,
    *,
    aggregate: Optional[Callable[[List[tuple]], Dict[str, Any]]] = None,
) -> float:
    """轨迹等权的 baseline MSE。

    默认走 evaluator 的 ``aggregate_chunks``（**同一份实现** ⇒ 口径一致）；
    测试里可以注入桩函数以便在无 torch 环境下验证接线。
    """
    if not chunks:
        raise ValueError("chunks 为空，无法计算 baseline")
    agg = aggregate
    if agg is None:
        from lingbotvla.utils.open_loop_validation import aggregate_chunks as agg  # 懒加载
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    pairs: List[tuple] = []
    for k, gt in chunks:
        g = np.asarray(gt, dtype=np.float64)
        if g.ndim != 2:
            raise ValueError(f"GT chunk 必须是 (N, D) 二维，实际 {g.shape}")
        if g.shape[1] != mu.shape[0]:
            raise ValueError(
                f"μ_task 维度({mu.shape[0]}) 与 GT 维度({g.shape[1]}) 不一致 —— "
                "多半是 action_keys / 数据集配置变了")
        pairs.append((k, g, np.broadcast_to(mu, g.shape).copy()))
    return float(agg(pairs)["mse"])


def build_baseline(
    task: str,
    chunks: Sequence[Tuple[Any, np.ndarray]],
    *,
    fingerprint: str,
    action_keys: Sequence[str] = (),
    n_train_episodes: int = 0,
    mu_weighting: str = MU_GLOBAL,
    aggregate: Optional[Callable[[List[tuple]], Dict[str, Any]]] = None,
) -> FixedBaseline:
    """从 GT chunks 组装一个 ``FixedBaseline``（纯计算，不碰数据集）。"""
    mu = compute_mu(chunks, weighting=mu_weighting)
    mse = trajectory_balanced_baseline(chunks, mu, aggregate=aggregate)
    n_frames = int(sum(np.asarray(gt).shape[0] for _, gt in chunks))
    dims = int(mu.shape[0])
    return FixedBaseline(
        task=task, mse=mse, mu=tuple(float(x) for x in mu), fingerprint=fingerprint,
        n_train_episodes=int(n_train_episodes), n_chunks=len(chunks),
        n_frames=n_frames, dims=dims, action_keys=tuple(action_keys),
        mu_weighting=mu_weighting,
    )


# --------------------------------------------------------------------------- #
# 指纹
# --------------------------------------------------------------------------- #


def file_sha256(path: str, *, chunk: int = 1 << 20) -> Optional[str]:
    """文件内容哈希（用于 `norm_stats_file` —— 归一化统计变了分母就不可比）。"""
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def baseline_fingerprint(
    *,
    dataset_root: Optional[str] = None,
    sha256_train: Optional[str] = None,
    norm_stats_file: Optional[str] = None,
    cameras: Optional[Sequence[str]] = None,
    joints: Optional[Sequence[str]] = None,
    chunk_size: Optional[int] = None,
    img_size: Optional[int] = None,
    per_episode_stride: bool = True,
    mu_weighting: str = MU_GLOBAL,
) -> str:
    """**配置级**指纹：任一字段变化 ⇒ 拒绝复用旧缓存。

    ⚠️ 这里**不含** per-task 的 `sha256_train`（虽然参数里可以传，方便单测）。
    生产路径请用两层：

        config_fp = baseline_fingerprint(... 不含 sha256_train ...)
        task_fp   = task_fingerprint(config_fp, entry.sha256_train)
    """
    payload = {
        "v": BASELINE_VERSION,
        "dataset_root": dataset_root,
        "sha256_train": sha256_train,
        "norm_stats_sha256": file_sha256(norm_stats_file) if norm_stats_file else None,
        "norm_stats_path": os.path.basename(norm_stats_file) if norm_stats_file else None,
        "cameras": list(cameras or []),
        "joints": list(joints or []),
        "chunk_size": chunk_size,
        "img_size": img_size,
        "per_episode_stride": bool(per_episode_stride),
        "mu_weighting": mu_weighting,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def task_fingerprint(config_fingerprint: str, sha256_train: Optional[str]) -> str:
    """**任务级**指纹 = 配置级指纹 + 该 task 的 train split 哈希。

    review v0.1 #2：原来 store 只持有**配置级**指纹，而写入的是**任务级**指纹
    ⇒ `get()` 永远不相等 ⇒ **缓存永不命中**。现在两层分开。
    """
    blob = f"{config_fingerprint}|{sha256_train or ''}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #


@dataclass
class BaselineStore:
    """`task_baseline.json` 的读写（**不落盘时完全无副作用**）。

    结构::

        {"version": 1,
         "config_fingerprint": "...",            # 全任务共享
         "tasks": {"click_bell": {..., "fingerprint": "<task_fp>"}}}

    两层指纹（review v0.1 #2）：

    * ``config_fingerprint`` —— 数据路径 / 归一化 / 相机 / chunk / mu 加权方式
    * 每个 task 存的 ``fingerprint`` = ``task_fingerprint(config_fp, sha256_train)``

    ``get(task, sha256_train)`` 逐 task 校验；不一致 ⇒ 返回 ``None`` 并记 warning。
    """

    path: str
    config_fingerprint: str = ""
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str, *, config_fingerprint: str = "") -> "BaselineStore":
        store = cls(path=path, config_fingerprint=config_fingerprint)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            if int(raw.get("version", 0)) != BASELINE_VERSION:
                store.warnings.append(
                    f"缓存版本不符（{raw.get('version')} != {BASELINE_VERSION}），忽略旧缓存")
            else:
                store.tasks = dict(raw.get("tasks", {}))
                if config_fingerprint and raw.get("config_fingerprint") not in (None, config_fingerprint):
                    store.warnings.append(
                        f"缓存配置指纹不符（{raw.get('config_fingerprint')} != "
                        f"{config_fingerprint}）⇒ 逐任务复核")
        return store

    def expected_fingerprint(self, sha256_train: Optional[str]) -> str:
        return task_fingerprint(self.config_fingerprint, sha256_train)

    def get(self, task: str, sha256_train: Optional[str] = None) -> Optional[FixedBaseline]:
        """取该 task 的 baseline；指纹不符 / 不存在 ⇒ ``None``。"""
        rec = self.tasks.get(task)
        if rec is None:
            return None
        want = self.expected_fingerprint(sha256_train)
        if rec.get("fingerprint") != want:
            self.warnings.append(
                f"[{task}] baseline 指纹不符（缓存 {rec.get('fingerprint')} != "
                f"当前 {want}）⇒ 拒绝复用，需重算")
            return None
        return FixedBaseline.from_dict(rec)

    def put(self, b: FixedBaseline) -> None:
        self.tasks[b.task] = b.to_dict()

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        payload = {
            "version": BASELINE_VERSION,
            "config_fingerprint": self.config_fingerprint,
            "tasks": self.tasks,
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------- #
# 从真实数据集收集（需要数据集；不需要模型权重）
# --------------------------------------------------------------------------- #


def compute_fixed_baseline(
    validator,
    task: str,
    train_ids: Sequence[int],
    *,
    fingerprint: str,
    tag: Optional[str] = None,
    mu_weighting: str = MU_GLOBAL,
    aggregate: Optional[Callable[[List[tuple]], Dict[str, Any]]] = None,
    strict: bool = True,
) -> FixedBaseline:
    """在真实数据集上算一个 task 的固定分母。

    ``validator`` 需提供 ``collect_gt_chunks(ids, tag, strict=...)`` —— 也就是
    ``OpenLoopValidator``（可以 ``model=None`` + ``model_config=<LingbotVLAV2Config>``，
    即**不需要权重、不需要 GPU**）。

    ``strict=True``（默认，review v0.1 #6）：拿不到 ``local_idx→episode_index`` 映射时
    **直接报错** —— 否则会退化成「每个 chunk 当一条 trajectory」，
    一次性固定分母被永久污染，NMSE 尺子从此不可信。

    ``aggregate`` 仅供测试注入桩函数；默认走 evaluator 的 ``aggregate_chunks``。
    """
    tag = tag or f"baseline_{task}"
    chunks, action_keys = validator.collect_gt_chunks(list(train_ids), tag, strict=strict)
    if not chunks:
        raise RuntimeError(f"[{task}] 收集到的 GT chunk 为空，无法算 baseline")
    return build_baseline(
        task, chunks, fingerprint=fingerprint, action_keys=action_keys,
        n_train_episodes=len(train_ids), mu_weighting=mu_weighting, aggregate=aggregate,
    )


__all__ = [
    "FixedBaseline", "BaselineStore",
    "compute_mu", "trajectory_balanced_baseline", "build_baseline",
    "baseline_fingerprint", "task_fingerprint", "file_sha256", "compute_fixed_baseline",
    "MU_GLOBAL", "MU_TRAJECTORY", "BASELINE_VERSION",
]
