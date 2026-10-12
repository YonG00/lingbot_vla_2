"""分片「协议内归并」的回归测试（2026-10-12 B1 修复）。

真机问题（al_v49/v50/v53/v55 实测）：
    分片模式下 `task/<t>/active_val_gmean_mse = None` ⇒ `check_pass` 返回 INVALID
    ⇒ **0 个免费 PASS、unit 末也判不了**（跑再久也不通关）。

根因（两步实测锤死）：
    ① 分片切分原本在「框架多卡协议**之上**」（backend 先切好 my_ids 再交给 adapter）；
    ② 而协议（`_multirank_eval_payload`）的语义是「各卡跑同一批 / **只有 rank0 结果算数** /
       广播 rank0 的结果」；
    ⇒ 每卡只交出"自己那 1 条" ⇒ 广播回到各卡的都是 rank0 那 1 条
      ⇒ 外层再 all_gather 收到 **7 份相同的 rank0 复制**（真机日志：
      `pre_dump=[{1109:0.0226} × 7]`、`per_len=1`）
      ⇒ 合并结果 `per_traj` 只有 1 条，而 `episode_ids` 被写成全集 4 条
      ⇒ `geometric_mse` 的 fail-closed 不变量（ids 必须与 per_traj 键一致）不成立
      ⇒ `gmean_mse = None` ⇒ 判定失效。

修法（B1）：把「切片」与「归并」都搬进**协议之内** ——
    各卡只算自己的切片，`merge_ranks` 在**广播之前**把各卡 raw 结果合并成**全集**，
    rank0 的返回值即全集 ⇒ 广播变成"把全集同步给所有卡"（协议语义不变、人人一致）。

本文件锁住四条：
  ① 逐轨迹合并：并集、按 episode id 去重、重复项以先出现者为准；
  ② `mse` 按**帧数加权**重算（与单卡 pooled mse 同口径）；缺帧数时等权退化；
  ③ **覆盖不全 ⇒ 抛错**（绝不静默产出会被 gmean fail-closed 的残缺结果）；
  ④ `set_shard_merge_strategy` 的注入/清除（`None` ⇒ 协议退回旧语义）。
"""

from __future__ import annotations

import pytest

from lingbotvla.utils.open_loop_validation import (
    _merge_ranks_callback,
    _merge_shard_raw_payloads,
    set_shard_merge_strategy,
    shard_merge_strategy,
)


def _raw(ids, mses, frames, *, n_chunks=1, mae=0.01):
    """造一个"某卡"的 raw payload（形状与 evaluator 返回一致）。"""
    return {
        "mse": float(sum(mses) / max(1, len(mses))),
        "n": len(ids), "n_chunks": int(n_chunks), "frames": int(sum(frames)),
        "mae": mae, "dims": 7, "action_keys": ["a"],
        "per_traj_mse": [float(v) for v in mses],
        "per_traj_ids": [int(i) for i in ids],
        "per_traj_frames": [int(f) for f in frames],
    }


# ---------------------------------------------------------------- ① 并集与去重

def test_merge_unions_all_ranks_and_dedups() -> None:
    """4 条轨迹分散在 3 张卡上（含 1 条回绕重复）⇒ 合并后恰好全集 4 条。"""
    results = [
        _raw([1109], [0.02], [100]),
        _raw([1110, 1111], [0.04, 0.08], [200, 300]),
        _raw([1116], [0.16], [400]),
        _raw([1109], [0.99], [100]),          # 回绕重复：**以先出现者为准**
    ]
    out = _merge_shard_raw_payloads(results, {"my_ids": [1109], "n_ids": 4})
    assert out["n"] == 4
    assert sorted(out["per_traj_ids"]) == [1109, 1110, 1111, 1116]
    # 重复项取先出现的 0.02（不是后到的 0.99）
    got = dict(zip(out["per_traj_ids"], out["per_traj_mse"]))
    assert got[1109] == pytest.approx(0.02)


# ---------------------------------------------------------------- ② 帧加权 mse

def test_merge_mse_is_frame_weighted() -> None:
    """pooled mse 必须是**帧数加权**（与单卡口径一致），不是各卡简单平均。"""
    results = [
        _raw([1], [0.01], [10]),
        _raw([2], [0.10], [90]),
    ]
    out = _merge_shard_raw_payloads(results, {"n_ids": 2})
    assert out["frames"] == 100
    # (0.01*10 + 0.10*90) / 100 = 0.091
    assert out["mse"] == pytest.approx(0.091, rel=1e-9)
    assert out["n_chunks"] == 2


def test_merge_mse_degrades_to_equal_weight_without_frames() -> None:
    """缺逐轨迹帧数（都为 0）⇒ 退化为等权平均，不产生 NaN/除零。"""
    results = [
        _raw([1], [0.02], [0]),
        _raw([2], [0.08], [0]),
    ]
    out = _merge_shard_raw_payloads(results, {"n_ids": 2})
    assert out["frames"] == 0
    assert out["mse"] == pytest.approx(0.05, rel=1e-9)


def test_merge_ignores_non_dict_entries() -> None:
    """汇总里混入 `None`（占位）不应让合并炸掉。"""
    out = _merge_shard_raw_payloads([None, _raw([3], [0.03], [50]), None], {"n_ids": 1})
    assert out["n"] == 1 and out["per_traj_ids"] == [3]
    assert out["mse"] == pytest.approx(0.03)


# ------------------------------------------------------- ③ 覆盖不全必须报错

def test_callback_raises_when_union_incomplete() -> None:
    """归并结果没覆盖全集 ⇒ **必须抛**（否则 gmean fail-closed 会静默变 None，判定失效）。"""
    set_shard_merge_strategy({"n_ids": 4})           # 全集要求 4 条
    try:
        with pytest.raises(RuntimeError) as ei:
            _merge_ranks_callback([_raw([1109], [0.02], [100])])   # 只给了 1 条
        assert "≠ 全集" in str(ei.value)
    finally:
        set_shard_merge_strategy(None)


def test_callback_accepts_complete_union() -> None:
    """覆盖全集 ⇒ 正常返回，且逐轨迹条数 == 全集条数（gmean 可算的前提）。"""
    set_shard_merge_strategy({"n_ids": 2})
    try:
        out = _merge_ranks_callback([_raw([1109], [0.02], [100]),
                                     _raw([1110], [0.04], [100])])
        assert out["n"] == 2
        from lingbotvla.auto_learning.decision.gmean import geometric_mse
        g = geometric_mse({i: v for i, v in zip(out["per_traj_ids"], out["per_traj_mse"])},
                          expected_count=out["n"], ids=out["per_traj_ids"])
        assert g is not None and g > 0      # ← 修复的核心目标：gmean **有值**
    finally:
        set_shard_merge_strategy(None)


# --------------------------------------------------- ④ 策略注入/清除（默认关）

def test_strategy_defaults_to_none_and_can_be_cleared() -> None:
    """未注入 ⇒ None（协议退回旧语义）；注入后能读回；清除后回到 None。"""
    set_shard_merge_strategy(None)
    assert shard_merge_strategy() is None
    set_shard_merge_strategy({"my_ids": [1], "n_ids": 4})
    assert shard_merge_strategy()["n_ids"] == 4
    set_shard_merge_strategy(None)
    assert shard_merge_strategy() is None
