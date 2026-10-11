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
    raw = json.loads(a.path.read_text(encoding='utf-8'))   # 缓存文件按模型分名，读实例的真实路径
    assert set(raw['tasks']) == {'taskA', 'taskB'}, f'任务被覆盖：{sorted(raw["tasks"])}'
    assert raw['tasks']['taskB']['losses'] == {'7': 0.7}


# --------------------------------------------------------------------------- #
# 模型名（核心新语义）
# --------------------------------------------------------------------------- #
def test_model_change_uses_separate_file(tmp_path):
    """换模型 ⇒ **各用一份文件**（2026-10-11 起），不再靠"未命中后覆盖同一文件"。"""
    f = tmp_path / 'hardness.json'
    a = _cache(f, model='model_A')
    a.store('t', {1: 1.0})

    other = _cache(f, model='model_B')
    assert other.path != a.path, '不同模型必须落到不同文件'
    hits, miss = other.load('t', [1])
    assert hits == {} and miss == [1], '模型不同不得复用'
    assert other.cached_count('t') == 0
    # A 的数据仍在自己的文件里，未被 B 影响
    assert json.loads(a.path.read_text(encoding='utf-8'))['tasks']['t']['losses'] == {'1': 1.0}


def test_same_model_across_code_change_reuses(tmp_path):
    """需求 1 的核心收益：**同一模型名**下，改代码不该影响命中（旧指纹方案会失效）。"""
    f = tmp_path / 'hardness.json'
    _cache(f).store('t', {1: 1.0, 2: 2.0})
    hits, miss = HardnessSampleCache(f, model=MODEL).load('t', [1, 2])
    assert miss == [] and len(hits) == 2


def test_model_switch_keeps_both_files(tmp_path):
    """换模型后写入：**两份文件并存**（旧模型数据保留，可作为历史缓存）。"""
    f = tmp_path / 'hardness.json'
    ca = _cache(f, model='model_A'); ca.store('t', {1: 1.0})
    cb = _cache(f, model='model_B'); cb.store('t', {2: 2.0})
    assert ca.path != cb.path
    ra = json.loads(ca.path.read_text(encoding='utf-8'))
    rb = json.loads(cb.path.read_text(encoding='utf-8'))
    assert ra['model'] == 'model_A' and ra['tasks']['t']['losses'] == {'1': 1.0}
    assert rb['model'] == 'model_B' and rb['tasks']['t']['losses'] == {'2': 2.0}


@pytest.mark.parametrize('explicit,ckpt,expect', [
    ('my-model', '/anywhere/weights', 'my-model'),
    ('my-model', None, 'my-model'),
    # 路径**不足 4 段** ⇒ 按实际段数取（旧行为不变）
    (None, '/models/robbyant_lingbot-vla-v2-6b-bf16', 'models/robbyant_lingbot-vla-v2-6b-bf16'),
    (None, None, 'unknown'),
    (None, 'relative/dir', 'relative/dir'),
])
def test_resolve_model_name(explicit, ckpt, expect):
    assert resolve_model_name(explicit, checkpoint_dir=ckpt) == expect


# --------------------------------------------------------------------------- #
# 「模型标识 = checkpoint 尾部 4 段」—— 2026-10-11 用户定（方案 A）
# --------------------------------------------------------------------------- #
TRAIN_CKPT = ('/models/robotwin-persistent/al_runs/train_al_v37/'
              'checkpoints/global_step_500/hf_ckpt')


def test_model_name_keeps_run_and_step_tail4():
    """尾部 4 段必须保留 **run 名 + 步数 + 权重类型**（这正是修 bug 的目的）。"""
    assert resolve_model_name(None, checkpoint_dir=TRAIN_CKPT) == \
        'train_al_v37/checkpoints/global_step_500/hf_ckpt'


def test_base_and_trained_ckpt_get_different_names():
    """**核心回归**：base 权重与"训练 500 步后的权重"必须得到**不同**标识。

    实测踩过：二者都被算成 `robbyant_lingbot-vla-v2-6b-bf16` ⇒ base 的 scout/hardness
    缓存被错误地用在 500 步权重的训练上（而两者表现差异极大：nmse 0.79 vs 0.09）。
    """
    base = resolve_model_name(None, checkpoint_dir='/workspace/models/robbyant_lingbot-vla-v2-6b-bf16')
    trained = resolve_model_name(None, checkpoint_dir=TRAIN_CKPT)
    assert base != trained, f'base 与训练后权重不能同标识：{base!r} == {trained!r}'


@pytest.mark.parametrize('run_a,run_b', [('train_al_v37', 'train_al_v38')])
def test_different_runs_same_step_do_not_collide(run_a, run_b):
    """**核心回归**：不同 run 的同一 step（都叫 global_step_500）不得撞车。

    若只取尾部 3 段（`checkpoints/global_step_500/hf_ckpt`），这两个 run 会得到同一个标识
    ⇒ 缓存互串。尾部 4 段把 run 名保住，因此必须不同。
    """
    ca = f'/models/robotwin-persistent/al_runs/{run_a}/checkpoints/global_step_500/hf_ckpt'
    cb = f'/models/robotwin-persistent/al_runs/{run_b}/checkpoints/global_step_500/hf_ckpt'
    na, nb = (resolve_model_name(None, checkpoint_dir=c) for c in (ca, cb))
    assert na != nb, f'不同 run 撞车：{na!r}'


def test_different_steps_do_not_collide():
    """同 run 内不同步数必须得到不同标识（缓存与步数绑定）。"""
    n250 = resolve_model_name(None, checkpoint_dir=TRAIN_CKPT.replace('500', '250'))
    n500 = resolve_model_name(None, checkpoint_dir=TRAIN_CKPT)
    assert n250 != n500


def test_explicit_name_still_wins():
    """显式 `--model-name` 仍然是**唯一权威**（向后兼容，不改变既有语义）。"""
    assert resolve_model_name('explicit-name', checkpoint_dir=TRAIN_CKPT) == 'explicit-name'


def test_model_name_never_used_as_path(tmp_path):
    """标识含 `/` 也必须安全：缓存把它当**纯字符串**写入 JSON，不做路径拼接。"""
    model = resolve_model_name(None, checkpoint_dir=TRAIN_CKPT)
    assert '/' in model
    f = tmp_path / 'hardness.json'
    c = HardnessSampleCache(f, model=model)
    c.store('t', {1: 0.5})
    raw = json.loads(c.path.read_text(encoding='utf-8'))
    assert raw['model'] == model                       # 原样保存（不做路径拼接）
    # 文件路径由 `base + slug` 决定（slug 把 `/` 换成 `_`），不存在以标识为子目录的情况
    assert c.path.parent == tmp_path and '/' not in c.path.name
    # 标识不同 ⇒ 落到**另一个文件**（互不干扰）
    other = HardnessSampleCache(f, model='train_al_v38/checkpoints/global_step_500/hf_ckpt')
    assert other.path != c.path
    assert other.load('t', [1])[0] == {}


def test_file_level_model_guard_still_works(tmp_path):
    """文件里的 `model` 与当前不符时仍判未命中（防手工拷错/旧迁移带来的混用）。"""
    f = tmp_path / 'hardness.json'
    c = _cache(f, model='model_A')
    c.store('t', {1: 1.0})
    raw = json.loads(c.path.read_text(encoding='utf-8'))
    raw['model'] = 'model_B'                      # 手工把文件标成别的模型
    c.path.write_text(json.dumps(raw), encoding='utf-8')
    hits, miss = _cache(f, model='model_A').load('t', [1])
    assert hits == {} and miss == [1]
    assert _cache(f, model='model_A').load('t', [1])[0] == {}


def test_empty_model_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='model'):
        HardnessSampleCache(tmp_path / 'h.json', model='   ')


# --------------------------------------------------------------------------- #
# 版本 / 容错
# --------------------------------------------------------------------------- #
def test_noise_semantic_version_mismatch_is_a_miss(tmp_path):
    f = tmp_path / 'hardness.json'
    c0 = _cache(f); c0.store('t', {1: 1.0})
    raw = json.loads(c0.path.read_text(encoding='utf-8'))
    raw['noise_semantic_version'] = NOISE_SEMANTIC_VERSION + 1
    c0.path.write_text(json.dumps(raw), encoding='utf-8')
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
