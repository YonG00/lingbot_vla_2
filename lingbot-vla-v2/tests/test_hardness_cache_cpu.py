"""hardness 单文件缓存（显式文件 + 模型名）契约测试 —— 2026-10-10 重写版。

对应需求（用户 2026-10-10 定）：
  1. 缓存**文件由入参显式指定**，不做指纹推导 ⇒ 改代码不再让缓存失效；
  2. 缓存内记 `model`，**模型不同即视为未命中**（明确打印原因，不静默复用）。

覆盖：
  * 命中往返：首次全缺 → 写入 → 再读全命中，数值逐位一致；
  * 部分复用：不同 probe 子集按**并集**互补，不互相覆盖；
  * **模型不同 ⇒ 未命中**（核心新语义）；同模型跨代码改动仍命中（需求 1 的收益）；
  * 噪声语义版本不符 / schema 不符 ⇒ 未命中；
  * 坏 JSON / 超大文件 ⇒ 视为未命中，不抛异常；
  * 写失败（父路径是文件）⇒ 静默返回，绝不打断训练；
  * 非 rank0 只读；`store()` 写入前重读 ⇒ **不覆盖别处已写入的任务**；
  * 端到端：`RealHardnessScorer` 第二轮 **0 次模型调用**、数值逐位一致。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lingbotvla.auto_learning.hardness_cache import (
    MAX_JSON_BYTES, NOISE_SEMANTIC_VERSION, VERSION,
    HardnessSampleCache, resolve_model_name,
)

MODEL = 'robbyant_lingbot-vla-v2-6b-bf16'


def _cache(path: Path, model: str = MODEL, **kw) -> HardnessSampleCache:
    return HardnessSampleCache(path, model=model, **kw)


# --------------------------------------------------------------------------- #
# 基本往返与并集
# --------------------------------------------------------------------------- #
def test_miss_then_store_then_hit_roundtrip(tmp_path):
    f = tmp_path / 'hardness.json'
    c = _cache(f)
    ids = [1, 5, 9]
    hits, miss = c.load('task_a', ids)
    assert hits == {} and miss == ids, '首次必须全未命中'

    c.store('task_a', {1: 0.11, 5: 0.55, 9: 0.99})

    c2 = _cache(f)                       # 新实例、同一文件
    hits2, miss2 = c2.load('task_a', ids)
    assert miss2 == [], '写入后应全命中'
    assert hits2 == {1: 0.11, 5: 0.55, 9: 0.99}, '数值必须逐位一致'
    assert c2.cached_count('task_a') == 3


def test_partial_reuse_union_across_probe_subsets(tmp_path):
    f = tmp_path / 'hardness.json'
    c = _cache(f)
    c.store('t', {1: 1.0, 2: 2.0})
    hits, miss = c.load('t', [1, 2, 3, 4])
    assert hits == {1: 1.0, 2: 2.0} and miss == [3, 4], '只应缺新增样本'
    c.store('t', {3: 3.0, 4: 4.0})
    _, miss2 = _cache(f).load('t', [1, 2, 3, 4])
    assert miss2 == [], '并集后应全命中'
    assert _cache(f).cached_count('t') == 4


def test_store_preserves_other_task_written_elsewhere(tmp_path):
    """`store()` 必须**先重读文件**：否则会用它自己的陈旧副本盖掉别的任务。"""
    f = tmp_path / 'hardness.json'
    a = _cache(f)
    b = _cache(f)
    a.load('taskA', [1])                 # a 读到空文件（触发 _data 缓存）
    b.store('taskB', {7: 0.7})           # b 先写 taskB
    a.store('taskA', {1: 0.1})           # a 后写 —— 不得把 taskB 抹掉
    raw = json.loads(f.read_text(encoding='utf-8'))
    assert set(raw['tasks']) == {'taskA', 'taskB'}, f'任务被覆盖：{sorted(raw["tasks"])}'
    assert raw['tasks']['taskB']['losses'] == {'7': 0.7}


# --------------------------------------------------------------------------- #
# 模型名（核心新语义）
# --------------------------------------------------------------------------- #
def test_model_mismatch_is_a_miss_and_logged(tmp_path):
    f = tmp_path / 'hardness.json'
    _cache(f, model='model_A').store('t', {1: 1.0})

    other = _cache(f, model='model_B')
    hits, miss = other.load('t', [1])
    assert hits == {} and miss == [1], '模型不同不得复用'
    assert other.last_miss_reason == 'model_mismatch'
    assert other.cached_count('t') == 0


def test_same_model_across_code_change_reuses(tmp_path):
    """需求 1 的核心收益：**同一模型名**下，改代码不该影响命中（旧指纹方案会失效）。"""
    f = tmp_path / 'hardness.json'
    _cache(f).store('t', {1: 1.0, 2: 2.0})
    hits, miss = HardnessSampleCache(f, model=MODEL).load('t', [1, 2])
    assert miss == [] and len(hits) == 2


def test_model_mismatch_rebuilds_file_with_new_model(tmp_path):
    """换模型后写入：文件应以新模型名重建（旧模型数据不残留）。"""
    f = tmp_path / 'hardness.json'
    _cache(f, model='model_A').store('t', {1: 1.0})
    _cache(f, model='model_B').store('t', {2: 2.0})
    raw = json.loads(f.read_text(encoding='utf-8'))
    assert raw['model'] == 'model_B'
    assert raw['tasks']['t']['losses'] == {'2': 2.0}


@pytest.mark.parametrize('explicit,ckpt,expect', [
    ('my-model', '/anywhere/weights', 'my-model'),
    (None, '/models/robbyant_lingbot-vla-v2-6b-bf16', 'robbyant_lingbot-vla-v2-6b-bf16'),
    (None, None, 'unknown'),
])
def test_resolve_model_name(explicit, ckpt, expect):
    assert resolve_model_name(explicit, checkpoint_dir=ckpt) == expect


def test_empty_model_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='model'):
        HardnessSampleCache(tmp_path / 'h.json', model='   ')


# --------------------------------------------------------------------------- #
# 版本 / 容错
# --------------------------------------------------------------------------- #
def test_noise_semantic_version_mismatch_is_a_miss(tmp_path):
    f = tmp_path / 'hardness.json'
    _cache(f).store('t', {1: 1.0})
    raw = json.loads(f.read_text(encoding='utf-8'))
    raw['noise_semantic_version'] = NOISE_SEMANTIC_VERSION + 1
    f.write_text(json.dumps(raw), encoding='utf-8')
    hits, miss = _cache(f).load('t', [1])
    assert hits == {} and miss == [1], '噪声语义版本变了必须重扫'


def test_schema_version_mismatch_is_a_miss(tmp_path):
    f = tmp_path / 'hardness.json'
    f.write_text(json.dumps({'version': VERSION - 1, 'model': MODEL, 'tasks': {}}),
                 encoding='utf-8')
    hits, miss = _cache(f).load('t', [1])
    assert hits == {} and miss == [1]


def test_corrupt_or_oversized_file_is_tolerated(tmp_path):
    f = tmp_path / 'hardness.json'
    f.write_text('{not json', encoding='utf-8')
    hits, miss = _cache(f).load('t', [1])
    assert hits == {} and miss == [1], '坏文件应视为未命中而不是抛异常'

    big = {'version': VERSION, 'noise_semantic_version': NOISE_SEMANTIC_VERSION, 'model': MODEL,
           'tasks': {'t': {'losses': {str(i): 1.5 for i in range(MAX_JSON_BYTES // 12)}}}}
    f.write_text(json.dumps(big), encoding='utf-8')
    assert f.stat().st_size > MAX_JSON_BYTES, '测试造数必须真的超过上限'
    hits2, miss2 = _cache(f).load('t', [1])
    assert hits2 == {} and miss2 == [1]


def test_store_failure_does_not_raise(tmp_path):
    """父路径是普通文件 ⇒ mkdir 必失败；store 必须静默返回。"""
    blocker = tmp_path / 'blocked'
    blocker.write_text('not a dir', encoding='utf-8')
    c = HardnessSampleCache(blocker / 'h.json', model=MODEL)
    c.store('t', {1: 1.0})
    assert not (blocker / 'h.json').exists()


def test_disabled_cache_is_pure_passthrough(tmp_path):
    f = tmp_path / 'hardness.json'
    c = _cache(f, enabled=False)
    c.store('t', {1: 1.0})
    hits, miss = c.load('t', [1])
    assert hits == {} and miss == [1]
    assert not f.exists()


def test_readonly_rank_does_not_write(tmp_path):
    f = tmp_path / 'hardness.json'
    _cache(f, write_enabled=True).store('t', {1: 1.0})
    reader = _cache(f, write_enabled=False)
    hits, _ = reader.load('t', [1])
    assert hits == {1: 1.0}, '非 rank0 也必须能读'
    reader.store('t', {2: 2.0})
    assert _cache(f).cached_count('t') == 1, '非 rank0 不得写'


# --------------------------------------------------------------------------- #
# 端到端：RealHardnessScorer 真的跳过已缓存样本
# --------------------------------------------------------------------------- #
class _CountingCore:
    def __init__(self):
        self.scored: list[int] = []

    def score(self, items, sample_ids=None):  # noqa: ANN001
        import numpy as np
        sids = ([int(it['idx']) for it in items] if sample_ids is None
                else [int(s) for s in sample_ids])
        self.scored.extend(sids)
        return np.asarray([float(sid) * 0.001 + 0.5 for sid in sids], dtype=float)


class _StubDataset:
    def __getitem__(self, idx):  # noqa: ANN001
        return {'joint_mask': True, 'idx': int(idx)}


def _scorer(core, cache):
    from lingbotvla.auto_learning.real.backend import RealHardnessScorer
    return RealHardnessScorer(core, _StubDataset(), max_batch=4, logger=None, cache=cache)


def test_scorer_skips_cached_samples_end_to_end(tmp_path):
    f = tmp_path / 'hardness.json'
    os.environ['AL_HARDNESS_UNIFY_RNG'] = '1'
    os.environ.pop('AL_HARDNESS_SHARD', None)
    try:
        core1 = _CountingCore()
        ids = list(range(1, 13))
        out1 = _scorer(core1, _cache(f)).score('t', ids)
        assert sorted(core1.scored) == ids, '第一轮应真扫全部样本'
        assert _cache(f).cached_count('t') == len(ids)

        core2 = _CountingCore()
        out2 = _scorer(core2, _cache(f)).score('t', ids)
        assert core2.scored == [], f'已缓存样本不得再进模型，实际打了 {core2.scored}'
        assert out2 == out1, '复用结果必须与首轮逐位一致'

        core3 = _CountingCore()
        big = ids + [100, 101]
        out3 = _scorer(core3, _cache(f)).score('t', big)
        assert sorted(core3.scored) == [100, 101], f'只应补扫新增样本，实际 {core3.scored}'
        assert set(out3) == set(big) and out3[1] == out1[1]
        assert _cache(f).cached_count('t') == len(big)
    finally:
        os.environ.pop('AL_HARDNESS_UNIFY_RNG', None)


def test_scorer_without_cache_keeps_old_behavior():
    os.environ['AL_HARDNESS_UNIFY_RNG'] = '1'
    os.environ.pop('AL_HARDNESS_SHARD', None)
    try:
        core = _CountingCore()
        ids = list(range(1, 7))
        _scorer(core, None).score('t', ids)
        assert sorted(core.scored) == ids, '无缓存时必须全扫（默认行为不变）'
    finally:
        os.environ.pop('AL_HARDNESS_UNIFY_RNG', None)
