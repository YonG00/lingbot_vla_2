"""hardness 逐样本磁盘缓存：CPU 契约测试（2026-10-10）。

覆盖：
  1. 首次全未命中 → 写入 → 再读全命中，且**数值逐位一致**；
  2. **部分复用**：请求更大 id 集合时，只缺新增的那些（不同 probe 设置互补）；
  3. 指纹不符 ⇒ 一律不命中（换权重/换口径后不得复用）；
  4. 噪声语义版本不符 ⇒ 不命中；
  5. 坏 JSON / 超大文件 ⇒ 视为未命中，不抛异常；
  6. **写失败不打断训练**（只读文件系统上 store 应静默返回）；
  7. 非法指纹（长度/字符）⇒ 构造时就报错（fail-fast，防手误）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lingbotvla.auto_learning.hardness_cache import (
    MAX_JSON_BYTES, NOISE_SEMANTIC_VERSION, VERSION, HardnessSampleCache,
)

FP = 'a' * 64


def _cache(tmp_path: Path, fp: str = FP, **kw) -> HardnessSampleCache:
    return HardnessSampleCache(tmp_path, fp, **kw)


def test_first_load_miss_then_store_then_hit_roundtrip(tmp_path):
    c = _cache(tmp_path)
    ids = [1, 5, 9]
    hits, miss = c.load('task_a', ids)
    assert hits == {} and miss == ids, '首次必须全未命中'

    c.store('task_a', {1: 0.11, 5: 0.55, 9: 0.99})

    hits2, miss2 = c.load('task_a', ids)
    assert miss2 == [], '写入后应全命中'
    assert hits2[1] == 0.11 and hits2[5] == 0.55 and hits2[9] == 0.99, '数值必须逐位一致'
    assert c.cached_count('task_a') == 3


def test_partial_reuse_across_different_probe_subsets(tmp_path):
    """不同 probe 设置扫的是不同子集 ⇒ 缓存按样本并集互补，而不是互相作废。"""
    c = _cache(tmp_path)
    c.store('t', {1: 1.0, 2: 2.0})
    hits, miss = c.load('t', [1, 2, 3, 4])
    assert hits == {1: 1.0, 2: 2.0} and miss == [3, 4], '只应缺新增样本'
    # 补扫 3/4 后再读：四个全命中
    c.store('t', {3: 3.0, 4: 4.0})
    hits2, miss2 = c.load('t', [1, 2, 3, 4])
    assert miss2 == [] and len(hits2) == 4
    assert c.cached_count('t') == 4, '并集应为 4（不能互相覆盖）'


def test_fingerprint_mismatch_is_a_miss(tmp_path):
    _cache(tmp_path, fp='b' * 64).store('t', {1: 1.0})
    other = _cache(tmp_path, fp='c' * 64)
    hits, miss = other.load('t', [1])
    assert hits == {} and miss == [1], '指纹不同不得复用'
    assert other.cached_count('t') == 0


def test_noise_semantic_version_mismatch_is_a_miss(tmp_path):
    c = _cache(tmp_path)
    c.store('t', {1: 1.0})
    p = c._task_path('t')
    raw = json.loads(p.read_text(encoding='utf-8'))
    raw['noise_semantic_version'] = NOISE_SEMANTIC_VERSION + 1
    p.write_text(json.dumps(raw), encoding='utf-8')
    hits, miss = c.load('t', [1])
    assert hits == {} and miss == [1], '噪声语义版本变了必须重扫（口径已不同）'


def test_corrupt_or_oversized_file_is_tolerated(tmp_path):
    c = _cache(tmp_path)
    c.store('bad', {1: 1.0})
    c._task_path('bad').write_text('{not json', encoding='utf-8')
    hits, miss = c.load('bad', [1])
    assert hits == {} and miss == [1], '坏文件应视为未命中而不是抛异常'

    c.store('big', {1: 1.0})
    p = c._task_path('big')
    raw = json.loads(p.read_text(encoding='utf-8'))
    # 直接按上限造：每条 `"123": 1.5` 约 15 字节 ⇒ 条目数 = 上限/12 保证超出
    raw['losses'] = {str(i): 1.5 for i in range(MAX_JSON_BYTES // 12)}
    p.write_text(json.dumps(raw), encoding='utf-8')
    assert p.stat().st_size > MAX_JSON_BYTES, '测试造数必须真的超过上限'
    hits2, miss2 = c.load('big', [1])
    assert hits2 == {} and miss2 == [1], '超过上限应视为未命中'


def test_store_failure_does_not_raise(tmp_path):
    """父路径是普通文件 ⇒ mkdir 必失败；store 必须静默返回（绝不打断训练）。"""
    blocker = tmp_path / 'blocked'
    blocker.write_text('not a dir', encoding='utf-8')
    c = HardnessSampleCache(blocker, FP)
    c.store('t', {1: 1.0})          # 不应抛异常
    hits, miss = c.load('t', [1])
    assert hits == {} and miss == [1]


def test_disabled_cache_is_pure_passthrough(tmp_path):
    c = _cache(tmp_path, enabled=False)
    c.store('t', {1: 1.0})          # enabled=False 不写
    hits, miss = c.load('t', [1])
    assert hits == {} and miss == [1]
    assert not (tmp_path / FP).exists()


def test_readonly_rank_does_not_write(tmp_path):
    writer = _cache(tmp_path, write_enabled=True)
    writer.store('t', {1: 1.0})
    reader = _cache(tmp_path, write_enabled=False)
    hits, miss = reader.load('t', [1])
    assert hits == {1: 1.0}, '非 rank0 也必须能读'
    reader.store('t', {2: 2.0})     # 非 rank0 不写
    assert writer.cached_count('t') == 1


@pytest.mark.parametrize('bad', ['', 'abc', 'A' * 64, 'z' * 64, 'a' * 63])
def test_invalid_fingerprint_fails_fast(tmp_path, bad):
    with pytest.raises(ValueError):
        HardnessSampleCache(tmp_path, bad)


def test_version_field_written(tmp_path):
    c = _cache(tmp_path)
    c.store('t', {1: 1.0})
    raw = json.loads(c._task_path('t').read_text(encoding='utf-8'))
    assert raw['version'] == VERSION
    assert raw['noise_semantic_version'] == NOISE_SEMANTIC_VERSION
    assert raw['fingerprint'] == FP and raw['task'] == 't' and raw['n'] == 1


# --------------------------------------------------------------------------- #
# 端到端：RealHardnessScorer 是否真的**跳过**已缓存样本（核心收益所在）
# --------------------------------------------------------------------------- #
class _CountingCore:
    """桩 scorer：记录被真正打分的 sample id（按 per-sample-id 语义，值可复现）。"""

    def __init__(self):
        self.scored: list[int] = []

    def score(self, items, sample_ids=None):  # noqa: ANN001
        import numpy as np
        sids = [int(it['idx']) for it in items] if sample_ids is None else [int(s) for s in sample_ids]
        self.scored.extend(sids)
        # 逐样本确定性：只由 sid 决定（模拟 per-sample-id RNG）
        return np.asarray([float(sid) * 0.001 + 0.5 for sid in sids], dtype=float)


class _StubDataset:
    def __getitem__(self, idx):  # noqa: ANN001
        return {'joint_mask': True, 'idx': int(idx)}


def test_scorer_skips_cached_samples_end_to_end(tmp_path):
    """第二轮扫描：已缓存样本**不得**再进模型；且返回值与第一轮逐位一致。"""
    from lingbotvla.auto_learning.real.backend import RealHardnessScorer

    os.environ['AL_HARDNESS_UNIFY_RNG'] = '1'      # 走逐样本绑定噪声（与分片同一语义）
    os.environ.pop('AL_HARDNESS_SHARD', None)
    try:
        # --- 第一轮：全部真扫，写入缓存 ---
        core1 = _CountingCore()
        c1 = HardnessSampleCache(tmp_path, FP)
        s1 = RealHardnessScorer(core1, _StubDataset(), max_batch=4, logger=None, cache=c1)
        ids = list(range(1, 13))
        out1 = s1.score('t', ids)
        assert sorted(core1.scored) == ids, '第一轮应真扫全部样本'
        assert c1.cached_count('t') == len(ids), '第一轮应把结果写进缓存'

        # --- 第二轮：全命中 ⇒ 一个样本都不该进模型 ---
        core2 = _CountingCore()
        c2 = HardnessSampleCache(tmp_path, FP)
        s2 = RealHardnessScorer(core2, _StubDataset(), max_batch=4, logger=None, cache=c2)
        out2 = s2.score('t', ids)
        assert core2.scored == [], f'已缓存样本不得再进模型，实际打了 {core2.scored}'
        assert out2 == out1, '复用结果必须与首轮逐位一致'

        # --- 第三轮：请求更大集合 ⇒ 只补缺失 ---
        core3 = _CountingCore()
        c3 = HardnessSampleCache(tmp_path, FP)
        s3 = RealHardnessScorer(core3, _StubDataset(), max_batch=4, logger=None, cache=c3)
        big = list(range(1, 13)) + [100, 101]
        out3 = s3.score('t', big)
        assert sorted(core3.scored) == [100, 101], f'只应补扫新增样本，实际 {core3.scored}'
        assert set(out3) == set(big) and out3[1] == out1[1]
        assert c3.cached_count('t') == len(big), '并集应累积'
    finally:
        os.environ.pop('AL_HARDNESS_UNIFY_RNG', None)


def test_scorer_without_cache_keeps_old_behavior(tmp_path):
    """不传 cache ⇒ 行为与改造前一致（每轮全扫）。"""
    from lingbotvla.auto_learning.real.backend import RealHardnessScorer

    os.environ['AL_HARDNESS_UNIFY_RNG'] = '1'
    os.environ.pop('AL_HARDNESS_SHARD', None)
    try:
        core = _CountingCore()
        s = RealHardnessScorer(core, _StubDataset(), max_batch=4, logger=None)   # 无 cache
        ids = list(range(1, 7))
        s.score('t', ids)
        assert sorted(core.scored) == ids, '无缓存时必须全扫（默认行为不变）'
    finally:
        os.environ.pop('AL_HARDNESS_UNIFY_RNG', None)
