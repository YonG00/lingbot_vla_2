"""AL 多卡评测分片：预检判据 + 自适应窗口（2026-10-12）。

真机背景（al_v47，从 2400 HF 起跑、`AL_EVAL_SHARD=1`）：
    训练起来了（池子导入成功、Step 2400），但**首个 active_val 评测**被
    "多卡评测一致性预检"拦下：

        ❌ 多卡评测一致性预检未通过（tag=al_move_can_pot_val_6711597a，world_size=7）
           - rank0: ids=[1109] n_starts=3 len_ds=143
           - rank1: ids=[1110] n_starts=4 len_ds=151

    两个独立缺陷叠在一起：
      ① **预检判据过严**：它要求 `ids/len_ds/n_starts` 逐一相同，而分片的**设计**
         就是让 `ids` 不同 ⇒ 必然误报；
      ② **推理次数真的不等**（`n_starts` 3 vs 4）：分片只对齐了"轨迹条数 k"，
         没有对齐"每条轨迹几个窗口" —— 窗口数 = `ceil(帧数/stride)`，
         帧数不同 ⇒ 次数不同 ⇒ **真跑下去会在 all-gather 上卡死**（预检拦对了）。

本文件锁住两条：
  · `adaptive_episode_starts`：**每条轨迹恰好 k 个窗口**（根治 ②）；
  · `_preflight_verdict`：分片模式不比 `ids`（修 ①），但**仍死守**
    `n_starts` 相等（防卡死）**并新增**"并集覆盖完整"（防漏评）；非分片模式逐字不变。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from lingbotvla.utils.open_loop_validation import (
    _preflight_verdict,
    adaptive_episode_starts,
    eval_shard_indices,
    per_episode_starts,
)

#: al_v47 真机实测的 move_can_pot val 帧数（日志 len_ds 与数据集 meta 一致）。
MOVE_CAN_POT_VAL_FRAMES = [(1109, 143), (1110, 151), (1111, 148), (1116, 157)]


def _ep_map(frames: List[tuple]) -> np.ndarray:
    ids = [ep for ep, _ in frames]
    counts = [n for _, n in frames]
    return np.repeat(np.array(ids, dtype=np.int64), counts)


def _counts_per_episode(starts: List[int], ep_map: np.ndarray) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for s in starts:
        ep = int(ep_map[s])
        out[ep] = out.get(ep, 0) + 1
    return out


# --------------------------------------------------------------------------- #
# ① adaptive_episode_starts：每条轨迹恰好 k 个窗口
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_adaptive_gives_exactly_k_windows_per_episode(k: int) -> None:
    ep_map = _ep_map(MOVE_CAN_POT_VAL_FRAMES)
    starts = adaptive_episode_starts(ep_map, k)
    per = _counts_per_episode(starts, ep_map)
    assert set(per) == {ep for ep, _ in MOVE_CAN_POT_VAL_FRAMES}
    assert set(per.values()) == {k}, f"k={k} 时各轨迹窗口数不一致: {per}"
    assert len(starts) == k * len(MOVE_CAN_POT_VAL_FRAMES)


def test_adaptive_starts_are_inside_their_episode() -> None:
    """起点必须落在本回合区间内（跨回合就变成别的评测集，见 per_episode_starts 的注释）。"""
    ep_map = _ep_map(MOVE_CAN_POT_VAL_FRAMES)
    starts = adaptive_episode_starts(ep_map, 2)
    for s in starts:
        assert 0 <= s < len(ep_map)
    # 每个回合的第一个起点 = 回合首帧
    firsts = {}
    for s in starts:
        firsts.setdefault(int(ep_map[s]), s)
    i = 0
    while i < len(ep_map):
        ep = int(ep_map[i])
        assert firsts[ep] == i, f"回合 {ep} 的首个起点不是首帧 {i}"
        j = i
        while j < len(ep_map) and int(ep_map[j]) == ep:
            j += 1
        i = j


def test_adaptive_non_shard_path_unchanged() -> None:
    """非分片路径（per_episode_starts）行为必须逐字不变：ceil(帧数/stride)。"""
    ep_map = _ep_map(MOVE_CAN_POT_VAL_FRAMES)
    starts = per_episode_starts(ep_map, 50)
    per = _counts_per_episode(starts, ep_map)
    assert per == {1109: 3, 1110: 4, 1111: 3, 1116: 4}      # 真机日志的 n_starts 来源


def test_adaptive_edge_cases() -> None:
    assert adaptive_episode_starts(np.array([], dtype=np.int64), 2) == []
    assert adaptive_episode_starts(np.array([7], dtype=np.int64), 3) == [0]      # 1 帧 ⇒ 1 个起点
    # 100 帧、k=3 ⇒ stride=ceil(100/3)=34 ⇒ 起点 0,34,68
    assert adaptive_episode_starts(np.zeros(100, dtype=np.int64), 3) == [0, 34, 68]


# --------------------------------------------------------------------------- #
# ② 分片下标：每卡条数恒等 + 并集覆盖完整
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,world", [(10, 7), (4, 7), (8, 7), (7, 7), (14, 7), (1, 3)])
def test_shard_indices_equal_counts_and_full_union(n: int, world: int) -> None:
    idx = [eval_shard_indices(n=n, world=world, rank=r) for r in range(world)]
    assert len({len(x) for x in idx}) == 1, f"各卡条数不等: {[len(x) for x in idx]}"
    union = {i for x in idx for i in x}
    assert union == set(range(n)), f"并集不完整: {sorted(union)}"


# --------------------------------------------------------------------------- #
# ③ 预检判据：分片模式（修误报 + 守死硬约束）
# --------------------------------------------------------------------------- #
def _report(rank: int, ids: List[int], *, n_starts: int, n_ids: Optional[int] = None,
            len_ds: int = 100, error: Optional[str] = None) -> Dict[str, Any]:
    """构造一个 rank 的预检报告。

    ⚠️ `len_ds` 默认**固定**：非分片模式要求五字段逐一相同，若默认就随 rank 变，
    那非分片用例会（正确地）报 len_ds 不一致 —— 想要那种场景请显式传不同值。
    """
    return {"rank": rank, "tag": "al_task_val_deadbeef", "ids": sorted(ids),
            "n_starts": n_starts, "len_ds": len_ds, "warmup": True,
            "n_ids": n_ids, "error": error}


def _sharded_reports(n: int, world: int, *, k_windows: int,
                     per_rank_tag: bool = False) -> List[Dict[str, Any]]:
    """分片模式下的各 rank 报告。

    ``per_rank_tag=True`` 模拟**真机形态**：`tag` 含 ids 指纹（`evaluator.py` 的
    「ids 指纹进 tag」）⇒ 各卡 tag **必然不同**，而分片判据**不该**因此报错。
    """
    out = []
    for r in range(world):
        rep = _report(r, eval_shard_indices(n=n, world=world, rank=r),
                      n_starts=k_windows, n_ids=n)
        if per_rank_tag:
            rep["tag"] = f"al_task_val_{r:08x}"
        out.append(rep)
    return out


def test_preflight_sharded_accepts_different_ids() -> None:
    """分片模式：ids 不同是**设计**（原来被误判成不一致）。"""
    reports = _sharded_reports(10, 7, k_windows=4)
    verdict = _preflight_verdict(7, reports, n_ids=10, sharded=True)
    assert verdict["problems"] == [], verdict["problems"]
    assert verdict["failures"] == []


def test_preflight_sharded_accepts_per_rank_tags_like_real_machine() -> None:
    """**真机形态回归**（al_v48 实测）：各卡 tag/ids/len_ds 都不同、n_starts 相等。

    背景：`tag` 带 ids 指纹（`evaluator.py`：ids 指纹进 tag，否则 scout/confirm 会共用
    同一份 episode_ids 文件）⇒ 分片下各卡 tag **必然不同**。曾把 tag 也列为"必须相同"
    ⇒ 真机误报 `[shard] rank1 与 rank0 的 tag 不一致`。
    """
    reports = _sharded_reports(4, 7, k_windows=1, per_rank_tag=True)
    for i, r in enumerate(reports):
        r["len_ds"] = 140 + i          # 各卡数据集长度也不同（真机 143/151/148/…）
    verdict = _preflight_verdict(7, reports, n_ids=4, sharded=True)
    assert verdict["problems"] == [], verdict["problems"]


def test_preflight_sharded_global_n_overrides_local_subset() -> None:
    """分片时"全集条数"必须取**上游公告的全局 n**，而报告自带的 `n_ids` 只是本卡子集长度。

    真机形态（al_v48 实测）：每卡 `ids` 只有 1 条 ⇒ 报告里 `n_ids=len(ids)=1`，
    而 `active_val` 全集是 4 条。若用报告值当全集，并集 4 会被判成"越界 4≠1"（假报）；
    用全局 n=4 才对。

    本用例把三种情形**都显式声明**，不依赖"环境是否干净"（曾因环境泄漏误判）：
      A. 报告值当全集（`n_ids=1`）⇒ 并集 4 ≠ 1 ⇒ 报并集错（说明"用错分母会误判"）
      B. 公告全局 n=4 ⇒ 并集 4 == 4 ⇒ 通过
      C. 公告全局 n=5（真漏一条）⇒ 报"漏评"
    """
    import os as _os
    _os.environ.pop("AL_EVAL_SHARD_CTX_N", None)
    _os.environ.pop("AL_EVAL_SHARD_EXPECT_STEPS", None)

    # 真机形态：7 卡各 1 条、并集恰好 4 条（= 全集）
    parts = [[1109], [1110], [1111], [1116], [1109], [1110], [1111]]
    reports = [_report(r, parts[r], n_starts=1, n_ids=len(parts[r])) for r in range(7)]
    assert len({i for r in reports for i in r["ids"]}) == 4

    try:
        # A：显式传入错误的全集（1）⇒ 应报错（证明分母用错会导致误判）
        va = _preflight_verdict(7, reports, n_ids=1, sharded=True)
        assert any("并集" in p for p in va["problems"]), va["problems"]

        # B：公告全局 n=4 ⇒ 通过
        _os.environ["AL_EVAL_SHARD_CTX_N"] = "4"
        vb = _preflight_verdict(7, reports, sharded=True)
        assert vb["problems"] == [], vb["problems"]

        # C：公告全局 n=5 ⇒ 真漏一条 ⇒ 报"漏评"
        _os.environ["AL_EVAL_SHARD_CTX_N"] = "5"
        vc = _preflight_verdict(7, reports, sharded=True)
        assert any("并集" in p and "漏评" in p for p in vc["problems"]), vc["problems"]
    finally:
        _os.environ.pop("AL_EVAL_SHARD_CTX_N", None)
        _os.environ.pop("AL_EVAL_SHARD_EXPECT_STEPS", None)

def test_preflight_sharded_still_rejects_unequal_n_starts() -> None:
    """**fail-closed 不退化**：推理次数不等 ⇒ 必须拦住（真跑会卡在 all-gather）。"""
    reports = _sharded_reports(10, 7, k_windows=4)
    reports[3]["n_starts"] = 3
    verdict = _preflight_verdict(7, reports, n_ids=10, sharded=True)
    assert verdict["problems"], "n_starts 不等却放行了 ⇒ 会静默卡死"
    assert any("n_starts" in p for p in verdict["problems"])


def test_preflight_sharded_rejects_missing_trajectory() -> None:
    """新增能力：并集缺一条轨迹（漏评）必须拦住 —— 旧判据发现不了这种。"""
    parts = [[0, 1], [2, 3], [4, 5], [6, 8], [9], [1, 0], [2, 3]]     # 缺 7
    reports = [_report(r, parts[r], n_starts=4, n_ids=10) for r in range(7)]
    verdict = _preflight_verdict(7, reports, n_ids=10, sharded=True)
    assert any("并集" in p for p in verdict["problems"]), verdict["problems"]


def test_preflight_sharded_rejects_out_of_range_id() -> None:
    parts = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [99], [2, 3]]     # 多出 99
    reports = [_report(r, parts[r], n_starts=4, n_ids=10) for r in range(7)]
    verdict = _preflight_verdict(7, reports, n_ids=10, sharded=True)
    assert any("并集" in p for p in verdict["problems"]), verdict["problems"]


def test_preflight_sharded_rejects_rank_error() -> None:
    reports = _sharded_reports(10, 7, k_windows=4)
    reports[2]["error"] = "RuntimeError: 数据集打不开"
    verdict = _preflight_verdict(7, reports, n_ids=10, sharded=True)
    assert verdict["failures"], "某 rank 报错时必须一起停（fail-closed）"


def test_preflight_non_shard_path_unchanged() -> None:
    """非分片模式：仍是"五字段逐一相同"的老判据，逐字保留。"""
    same = [_report(r, [1, 2, 3], n_starts=2) for r in range(7)]
    assert _preflight_verdict(7, same, sharded=False)["problems"] == []

    diff_ids = [_report(r, [1, 2, 3], n_starts=2) for r in range(7)]
    diff_ids[1]["ids"] = [1, 2, 9]
    assert any("ids" in p for p in
               _preflight_verdict(7, diff_ids, sharded=False)["problems"])

    diff_len = [_report(r, [1, 2, 3], n_starts=2) for r in range(7)]
    diff_len[1]["len_ds"] = 999
    assert any("len_ds" in p for p in
               _preflight_verdict(7, diff_len, sharded=False)["problems"])

    # 分片模式下 len_ds 不同**不应**报错（各卡数据集长度本就不同）
    assert _preflight_verdict(7, diff_len, n_ids=None, sharded=True)["problems"] == []


# --------------------------------------------------------------------------- #
# ④ 端到端一致性：分片 + 自适应 ⇒ 各 rank n_starts 恒等（真机崩点的回归测试）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("world", [1, 2, 4, 7, 8])
def test_end_to_end_n_starts_equal_across_ranks(world: int) -> None:
    """用真机 move_can_pot val 的帧数复现：修复后各 rank 推理次数必须相等。

    修复前（窗口数 = ceil(帧数/50)）：world=7 时 n_starts = 6/7 混杂 ⇒ 预检拦下。
    """
    frames = MOVE_CAN_POT_VAL_FRAMES
    n = len(frames)
    k = -(-n // world)
    n_starts_per_rank = []
    for r in range(world):
        idx = eval_shard_indices(n=n, world=world, rank=r)
        my = [frames[i] for i in idx]
        ep_map = _ep_map(my)
        starts = adaptive_episode_starts(ep_map, k)
        n_starts_per_rank.append(len(starts))
    assert len(set(n_starts_per_rank)) == 1, (
        f"world={world} 各卡推理次数不等: {n_starts_per_rank}")

    # 并集必须是全集（回绕重复由汇总去重）
    union = {i for r in range(world)
             for i in eval_shard_indices(n=n, world=world, rank=r)}
    assert union == set(range(n))
