"""采样小工具。全部走显式传入的 `random.Random`，便于存档 / 恢复。"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, TypeVar

T = TypeVar("T")


def validate_probs(probs: Dict[T, float], *, context: str = "") -> None:
    """采样概率的合法性检查（测试方案 §I06）。

    负值 / NaN / 全 0 都会让「按概率抽」失去意义。**报错，不静默兜底** ——
    静默兜底只会把问题推到更晚、更难查的地方。
    """
    tag = f"（{context}）" if context else ""
    if not probs:
        raise ValueError(f"采样概率为空{tag}")
    for key, value in probs.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"采样概率含 NaN/Inf{tag}：key={key} value={value!r}")
        if value < 0:
            raise ValueError(f"采样概率为负{tag}：key={key} value={value}")
    if sum(probs.values()) <= 0:
        raise ValueError(f"采样概率全为 0，无法构成分布{tag}")


def shuffled(seq: Sequence[T], rng: random.Random) -> List[T]:
    out = list(seq)
    rng.shuffle(out)
    return out


def weighted_choice(items: Sequence[T], weights: Sequence[float], rng: random.Random) -> T:
    """按权重抽一个（权重不必归一化）。全 0 时退化为均匀。"""
    if not items:
        raise ValueError("weighted_choice: empty items")
    total = sum(weights)
    if total <= 0:
        return items[rng.randrange(len(items))]
    r = rng.random() * total
    acc = 0.0
    for item, w in zip(items, weights):
        acc += w
        if r <= acc:
            return item
    return items[-1]


def weighted_sample_without_replacement(
    items: Sequence[T],
    weights: Sequence[float],
    k: int,
    rng: random.Random,
) -> List[T]:
    """不放回加权抽样。

    复杂度 O(k·n) —— n 是单个任务的训练帧数（几千），k 是 new_slots（7），
    完全够用，且实现简单到不会出错。
    """
    pool = list(items)
    w = [float(x) for x in weights]
    k = min(k, len(pool))
    out: List[T] = []
    for _ in range(k):
        total = sum(w)
        if total <= 0:
            idx = rng.randrange(len(pool))
        else:
            r = rng.random() * total
            acc = 0.0
            idx = len(pool) - 1
            for i, wi in enumerate(w):
                acc += wi
                if r <= acc:
                    idx = i
                    break
        out.append(pool.pop(idx))
        w.pop(idx)
    return out


def weighted_sample_from_probs(
    probs: Dict[T, float],
    k: int,
    rng: random.Random,
) -> List[T]:
    keys = list(probs.keys())
    return weighted_sample_without_replacement(keys, [probs[x] for x in keys], k, rng)


def uniform_sample(seq: Sequence[T], k: int, rng: random.Random) -> List[T]:
    pool = list(seq)
    k = min(k, len(pool))
    out: List[T] = []
    for _ in range(k):
        idx = rng.randrange(len(pool))
        out.append(pool.pop(idx))
    return out


# --------------------------------------------------------------------------- #
# 预编译加权抽样器（alias method）
# --------------------------------------------------------------------------- #
class WeightedSampler:
    """把一个固定的加权分布**预编译**成 alias table，之后每次抽样 O(1)。

    为什么需要它：一个 learning unit 要抽 `num_steps` 次 batch（默认 50 次）。
    每次现算累积分布是 O(n)，3000 个样本 × 50 次 = 15 万次无谓操作；
    alias table 建一次 O(n)，之后 50 次抽样几乎免费。

    ⚠️ 分布变了必须重建（`hardness_version` / `pass_sampling_version` 变了就是变了）。
    """

    __slots__ = ("keys", "_w", "_prob", "_alias")

    def __init__(self, keys: Sequence[T], weights: Sequence[float]) -> None:
        if len(keys) != len(weights):
            raise ValueError("keys 与 weights 长度不一致")
        if not keys:
            raise ValueError("加权分布为空")
        total = 0.0
        for w in weights:
            if not isinstance(w, (int, float)) or not math.isfinite(float(w)):
                raise ValueError(f"采样权重含 NaN/Inf：{w!r}")
            if w < 0:
                raise ValueError(f"采样权重为负：{w}")
            total += float(w)
        if total <= 0:
            raise ValueError("采样权重全为 0，无法构成分布")

        n = len(keys)
        self.keys = list(keys)
        self._w = [float(w) / total for w in weights]

        scaled = [p * n for p in self._w]
        small = [i for i, p in enumerate(scaled) if p < 1.0]
        large = [i for i, p in enumerate(scaled) if p >= 1.0]
        self._prob = [1.0] * n
        self._alias = list(range(n))
        while small and large:
            s = small.pop()
            l = large.pop()
            self._prob[s] = scaled[s]
            self._alias[s] = l
            scaled[l] = scaled[l] + scaled[s] - 1.0
            (small if scaled[l] < 1.0 else large).append(l)
        for i in small:
            self._prob[i] = 1.0
        for i in large:
            self._prob[i] = 1.0

    # ---------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.keys)

    def draw_index(self, rng: random.Random) -> int:
        i = rng.randrange(len(self.keys))
        return i if rng.random() < self._prob[i] else self._alias[i]

    def draw(self, rng: random.Random) -> T:
        return self.keys[self.draw_index(rng)]

    def draw_without_replacement(self, k: int, rng: random.Random) -> List[T]:
        """不放回抽 k 个。

        主体走 O(1) 的 alias 抽样 + 去重重试；只有分布**极端偏斜**
        （例如某样本概率恰好为 0）时才退化到精确的 O(n) 路径 —— 保证不会死循环。
        """
        n = len(self.keys)
        k = min(k, n)
        picked: set = set()
        out: List[int] = []
        attempts = 0
        limit = 20 * k + 100
        while len(out) < k and attempts < limit:
            attempts += 1
            i = self.draw_index(rng)
            if i in picked:
                continue
            picked.add(i)
            out.append(i)
        if len(out) < k:
            rest = [i for i in range(n) if i not in picked]
            w = [self._w[i] for i in rest]
            out.extend(_exact_without_replacement(rest, w, k - len(out), rng))
        return [self.keys[i] for i in out]

    def empirical(self, draws: int, rng: random.Random) -> Dict[T, float]:
        """跑 `draws` 次统计经验频率（诊断用）。"""
        counts: Dict[T, int] = {}
        for _ in range(draws):
            key = self.draw(rng)
            counts[key] = counts.get(key, 0) + 1
        return {k: v / draws for k, v in counts.items()}


def _exact_without_replacement(
    items: Sequence[T], weights: Sequence[float], k: int, rng: random.Random
) -> List[T]:
    """精确的 O(k·n) 不放回加权抽样（只在退化路径上用）。"""
    pool = list(items)
    w = [float(x) for x in weights]
    k = min(k, len(pool))
    out: List[T] = []
    for _ in range(k):
        total = sum(w)
        if total <= 0:
            idx = rng.randrange(len(pool))
        else:
            r = rng.random() * total
            acc = 0.0
            idx = len(pool) - 1
            for i, wi in enumerate(w):
                acc += wi
                if r <= acc:
                    idx = i
                    break
        out.append(pool.pop(idx))
        w.pop(idx)
    return out
