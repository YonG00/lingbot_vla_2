"""hardness 多卡分片：噪声不变性 + 汇总完整性（CPU，无 torch 依赖）。

背景
----
`RealHardnessScorer.score()` 此前在**每个 rank 上各算一遍完全相同的全量样本**（7 卡浪费 7×），
而瓶颈恰恰是逐样本读帧（实测 ~5.0 s/样本）。改成分片（`ids[rank::world_size]`）+ `all_gather_object`
汇总后，**必须**同时保证"同一样本在不同批组成/不同分片下分数不变"，否则分片就是改口径。

本用例用**确定性桩**模拟真实 scorer 的两种噪声语义：
  * 传 `sample_ids` ⇒ 逐样本噪声（只由 (seed, sid) 决定）⇒ 批组成无关；
  * 不传        ⇒ 整批一个 generator ⇒ 换批即换噪声（正是要避免的旧语义）。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lingbotvla.auto_learning.real.backend import RealHardnessScorer


class RecLogger:
    """记录所有 info 行（含非 rank0 路径，便于断言分片日志）。"""

    def __init__(self):
        self.lines: list[str] = []

    def info(self, msg: str) -> None:
        self.lines.append(str(msg))

    def info_rank0(self, msg: str) -> None:
        self.lines.append(str(msg))

    def warning(self, msg: str) -> None:
        self.lines.append('WARN ' + str(msg))


class NoiseAwareStubCore:
    """模拟真实噪声语义的桩：per-sample 噪声 vs 整批噪声。"""

    def __init__(self, seed: int = 1234):
        self.seed = seed
        self.batches: list[list[int]] = []

    def score(self, items, sample_ids=None):  # noqa: ANN001
        sids = [int(it['idx']) for it in items] if sample_ids is None else [int(s) for s in sample_ids]
        self.batches.append(list(sids))
        vals = []
        if sample_ids is None:
            # 旧语义：整批一个 generator ⇒ 同一 sid 在不同批里得到不同的值
            rng = np.random.default_rng(self.seed + len(sids))
            noise = rng.standard_normal(len(sids))
            vals = [float(sid) + float(noise[k]) for k, sid in enumerate(sids)]
        else:
            # 新语义：逐样本 ⇒ 只由 (seed, sid) 决定
            for sid in sids:
                rng = np.random.default_rng(self.seed ^ (sid * 2654435761 % (1 << 32)))
                vals.append(float(sid) + float(rng.standard_normal()))
        return np.asarray(vals, dtype=float)


class StubDataset:
    def __getitem__(self, idx):  # noqa: ANN001
        return {'joint_mask': True, 'idx': int(idx)}


def _mk(max_batch=8, logger=None, core=None):
    core = core or NoiseAwareStubCore()
    return RealHardnessScorer(core, StubDataset(), max_batch=max_batch, logger=logger), core


def test_bind_ids_keeps_per_sample_scores_batch_invariant():
    """开 UNIFY_RNG（= 分片强制路径）后，批大小不得改变任何样本的分数。"""
    ids = list(range(40))
    import os
    os.environ['AL_HARDNESS_UNIFY_RNG'] = '1'
    try:
        s8, c8 = _mk(max_batch=8)
        s40, c40 = _mk(max_batch=40)
        a = s8.score('t', ids)
        b = s40.score('t', ids)
    finally:
        os.environ.pop('AL_HARDNESS_UNIFY_RNG', None)
    assert set(a) == set(b) == set(ids)
    for sid in ids:
        assert a[sid] == pytest.approx(b[sid], rel=0, abs=1e-12), \
            f'sid={sid} 的分数随批大小变化（{a[sid]} vs {b[sid]}）⇒ 分片不可比'
    assert c8.batches[0] == ids[:8], '第一批应为前 8 个'
    assert len(c40.batches) == 1 and c40.batches[0] == ids, 'batch=40 应一批做完'


def test_subset_score_matches_full_run_when_ids_bound():
    """分片的本质 = 只算子集 ⇒ 子集分数必须与全量里对应项**逐位相同**。"""
    import os
    os.environ['AL_HARDNESS_UNIFY_RNG'] = '1'
    try:
        ids = list(range(30))
        full, _ = _mk(max_batch=7)
        ref = full.score('t', ids)
        # 模拟 3 卡交错分片：rank0=0,3,6,… / rank1=1,4,7,… / rank2=2,5,8,…
        merged = {}
        for r in range(3):
            s, _ = _mk(max_batch=4)
            part = s.score('t', ids[r::3])
            assert set(part) == set(ids[r::3]), '子集必须只返回自己那批'
            merged.update(part)
        assert set(merged) == set(ref)
        for sid in ids:
            assert merged[sid] == ref[sid], f'sid={sid} 分片与全量不一致'
    finally:
        os.environ.pop('AL_HARDNESS_UNIFY_RNG', None)


def test_default_path_without_binding_is_batch_sensitive():
    """反向对照：**不**绑定 ID 时，批大小确实会改变分数（说明分片必须绑定）。"""
    ids = list(range(16))
    import os
    os.environ.pop('AL_HARDNESS_UNIFY_RNG', None)
    s4, _ = _mk(max_batch=4)
    s16, _ = _mk(max_batch=16)
    a = s4.score('t', ids)
    b = s16.score('t', ids)
    diff = sum(1 for sid in ids if abs(a[sid] - b[sid]) > 1e-12)
    assert diff > 0, '若这里为 0，说明桩没模拟出"整批噪声"，本测试失去意义'


def test_shard_env_without_dist_returns_full_set(caplog):
    """`AL_HARDNESS_SHARD=1` 但没有初始化分布式 ⇒ 退回单卡全量，且不报错。"""
    import os
    os.environ['AL_HARDNESS_SHARD'] = '1'
    try:
        log = RecLogger()
        s, core = _mk(max_batch=8, logger=log)
        ids = list(range(20))
        out = s.score('t', ids)
    finally:
        os.environ.pop('AL_HARDNESS_SHARD', None)
    assert set(out) == set(ids), '未初始化分布式时必须退回全量，不能只算子集'
    assert s.last_timing['sharded'] is False, '未初始化分布式 ⇒ 不得标记为分片'
    joined = '\n'.join(log.lines)
    assert '分片' not in joined, f'未分片时不得出现"分片"字样: {joined}'


def test_log_lines_carry_sample_counts():
    """日志必须自证"扫多少、多久、data/score 各占多少"。"""
    log = RecLogger()
    s, _ = _mk(max_batch=5, logger=log)
    s.score('mytask', list(range(23)))
    joined = '\n'.join(log.lines)
    assert '开始打分' in joined and 'samples' in joined or '样本' in joined
    assert '23' in joined, f'样本数必须出现在日志里: {joined}'
    assert '打分完成' in joined
    assert 'data ' in joined and 'score ' in joined
