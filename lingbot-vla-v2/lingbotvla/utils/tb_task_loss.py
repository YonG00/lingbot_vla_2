"""分任务训练 loss 的**跨 micro-batch（完整 GBS）**聚合 —— 纯日志工具。

背景：`tasks/vla/train_lingbotvla.py` 原先在梯度累积（GAS>1）时只把**最后一个**
micro-batch 的 `batch_mean_losses` 写进 TensorBoard（原代码注释自认
"we only log the last mini batch if grad acc is activated"）
⇒ MICRO=1/GAS=4 时 `detailed_loss/<task>` 只反映 1/4 的 GBS，
且没出现在最后一个 micro-batch 里的任务**完全没有点**（无法据此评估 Replay 样本）。

本模块只做**计数与求和**：不参与梯度、反向、优化器、采样或 RNG。
特地不 import torch/numpy —— 以便在无 GPU 环境下单测（张量只通过 `.item()` 取值）。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

__all__ = ["accumulate_task_losses", "mean_task_losses"]


def _as_float(value: Any) -> float:
    """兼容 0-dim tensor / numpy 标量 / python float。"""
    return float(value.item()) if hasattr(value, "item") else float(value)


def accumulate_task_losses(
    sums: MutableMapping[str, float],
    counts: MutableMapping[str, int],
    dataset_names: Sequence[Any],
    batch_mean_losses: Iterable[Any],
) -> None:
    """把**一个 micro-batch** 的逐样本 loss 累加进 ``(sums, counts)``。

    * ``dataset_names[i]`` 与 ``batch_mean_losses[i]`` 一一对应；
    * 同一 micro-batch 内允许出现多个任务（交错采样）；
    * 调用方应先把张量 ``detach().cpu()``（**每个 micro-batch 一次**），
      避免在这里逐样本触发 CUDA 同步。
    """
    for name, loss in zip(dataset_names, batch_mean_losses):
        key = str(name)
        sums[key] = sums.get(key, 0.0) + _as_float(loss)
        counts[key] = counts.get(key, 0) + 1


def mean_task_losses(
    sums: Mapping[str, float], counts: Mapping[str, int]
) -> Dict[str, float]:
    """``sum / count`` ⇒ 每个任务在**完整 GBS** 上的样本均值（计数为 0 的键跳过）。"""
    return {k: sums[k] / counts[k] for k in counts if counts[k] > 0}
