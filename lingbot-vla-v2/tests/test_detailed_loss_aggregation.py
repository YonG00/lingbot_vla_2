"""`detailed_loss/<task>` 跨 micro-batch（完整 GBS）聚合的回归测试。

对应审查发现：GAS>1 时原实现只统计**最后一个** micro-batch（GAS=4 ⇒ 只反映 1/4 的 GBS，
且未出现在最后一批的任务完全没有点）。本测试覆盖：
1. GAS=4、多个任务**交错**采样；
2. 各任务**样本数不等**（含 1 个样本、含 4 个样本）；
3. 与「旧实现（只取最后一个 micro-batch）」的差异可复现；
4. 计数守恒（Σcount == micro-batch 数 × micro_batch_size）与空输入；
5. 0-dim tensor / numpy 标量 / python float 混合输入；
6. 生产实现（真 torch 张量）—— 无 torch 时跳过。
"""
from __future__ import annotations

import pytest

from lingbotvla.utils.tb_task_loss import accumulate_task_losses, mean_task_losses


def _run(micro_batches):
    """micro_batches: [(dataset_names, losses), ...]，返回 (sums, counts, means)。"""
    sums, counts = {}, {}
    for names, losses in micro_batches:
        accumulate_task_losses(sums, counts, names, losses)
    return sums, counts, mean_task_losses(sums, counts)


def test_gas4_interleaved_tasks_with_unequal_counts():
    # GAS=4、MICRO=1：4 个 micro-batch，任务交错，A 出现 3 次、B 1 次、C 2 次（总 6 个样本）
    mbs = [
        (["A", "A"], [0.10, 0.30]),   # micro 1：两个样本都属于 A
        (["B"], [0.50]),              # micro 2
        (["A"], [0.60]),              # micro 3
        (["C", "C"], [0.20, 0.40]),   # micro 4
    ]
    sums, counts, means = _run(mbs)
    assert counts == {"A": 3, "B": 1, "C": 2}
    assert means["A"] == pytest.approx((0.10 + 0.30 + 0.60) / 3)
    assert means["B"] == pytest.approx(0.50)
    assert means["C"] == pytest.approx((0.20 + 0.40) / 2)
    # 计数守恒：每个样本恰好被计一次
    assert sum(counts.values()) == sum(len(losses) for _n, losses in mbs) == 6


def test_old_behaviour_would_have_been_wrong():
    """复现旧实现：只取最后一个 micro-batch ⇒ 丢样本 + 偏差。"""
    mbs = [(["A", "A"], [0.10, 0.30]), (["B"], [0.50]), (["A"], [0.60]), (["C", "C"], [0.20, 0.40])]
    _sums, _counts, means = _run(mbs)
    names_last, losses_last = mbs[-1]
    old = {}
    for n, v in zip(names_last, losses_last):
        old[n] = v
    assert old == {"C": 0.40}                      # 旧实现只剩最后一个样本
    assert "A" not in old and "B" not in old       # A/B 完全没有点
    assert means["A"] != pytest.approx(old.get("A", float("nan")))  # 旧值不可用


def test_single_micro_batch_equals_old_behaviour():
    """GAS=1（或只有 1 个 micro-batch）时新旧实现一致 —— 保证不回归。"""
    mbs = [(["A", "A", "B"], [0.1, 0.3, 0.5])]
    _s, _c, means = _run(mbs)
    assert means == {"A": pytest.approx(0.2), "B": pytest.approx(0.5)}


def test_empty_and_zero_count_keys_are_dropped():
    sums, counts = {}, {}
    accumulate_task_losses(sums, counts, [], [])
    assert mean_task_losses(sums, counts) == {}
    # 手工塞一个 0 计数的键 ⇒ 必须跳过（不能 0/0）
    assert mean_task_losses({"A": 0.0}, {"A": 0}) == {}


def test_mixed_scalar_types_item_float_numpy():
    np = pytest.importorskip("numpy")

    class T:                       # 0-dim tensor 替身
        def __init__(self, v): self.v = v
        def item(self): return self.v

    mbs = [(["A"], [T(0.2)]), (["A"], [np.float32(0.4)]), (["A"], [0.6])]
    _s, counts, means = _run(mbs)
    assert counts == {"A": 3}
    assert means["A"] == pytest.approx(0.4, rel=1e-6)


def test_with_real_torch_tensors():
    torch = pytest.importorskip("torch")
    names = ["A", "A", "B"]
    loss_log = {"batch_mean_losses": torch.tensor([0.2, 0.4, 0.8], dtype=torch.float64)}
    sums, counts = {}, {}
    for mb_names, mb_loss in ((["A", "A"], loss_log["batch_mean_losses"][:2]),
                              (["B"], loss_log["batch_mean_losses"][2:])):
        accumulate_task_losses(sums, counts, mb_names, mb_loss)
    means = mean_task_losses(sums, counts)
    assert counts == {"A": 2, "B": 1}
    assert means["A"] == pytest.approx(0.3) and means["B"] == pytest.approx(0.8)


def test_helper_module_has_no_torch_dependency():
    """纯函数模块不得 import torch/numpy ⇒ 无 GPU 环境也能单测（也是"不新增常驻张量"的保证）。

    用 AST 只看真正的 import 节点 —— docstring 里出现 "import torch" 字样不算。
    """
    import ast
    import inspect
    import lingbotvla.utils.tb_task_loss as m
    tree = ast.parse(inspect.getsource(m))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert "torch" not in mods, f"不得依赖 torch，实际 imports={sorted(mods)}"
    assert "numpy" not in mods, f"不得依赖 numpy，实际 imports={sorted(mods)}"
