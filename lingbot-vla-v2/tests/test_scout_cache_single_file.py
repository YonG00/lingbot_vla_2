"""scout 缓存「单文件 + 模型名」契约（2026-10-10 用户要求，与 hardness 侧同构）。

背景：旧实现是 `<root>/<指纹64位hex>/<scout_key>.json`，指纹 = 权重分片 hash +
14 个评测链源码语义 hash + 选项 ⇒ **改一行代码就让整份缓存不可见**（与 hardness 同病）。
现改为「**显式文件 + 模型名**」：

  * 文件由入参指定（`--scout-cache-file`），路径不参与哈希 ⇒ 改代码不再失效；
  * 文件里记 `model`，**模型不同即视为未命中**（`last_miss_reason='model_mismatch'`）；
  * 记录仍带完整结构自洽校验（沿用旧 `load()` 的全部条件）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lingbotvla.auto_learning.scout_cache import (
    MAX_JSON_BYTES, VERSION, BootstrapScoutCache, scout_key,
)

MODEL = 'robbyant_lingbot-vla-v2-6b-bf16'


def _metrics(task='click_bell', ids=(51, 52), mse=0.5, base=1.0):
    return {'task': task, 'episode_ids': list(ids), 'n_trajs': len(ids), 'metric_valid': True,
            'mse': mse, 'baseline_mse': base, 'nmse': mse / base,
            'per_traj_mse': {str(i): mse for i in ids}}


def _cache(f: Path, model: str = MODEL, **kw) -> BootstrapScoutCache:
    return BootstrapScoutCache(f, model=model, **kw)


def test_roundtrip_single_file(tmp_path):
    f = tmp_path / 'scout.json'
    c = _cache(f)
    assert c.load('click_bell', [51, 52]) is None
    assert c.last_miss_reason == 'missing'

    c.store('click_bell', [51, 52], _metrics())
    assert f.is_file(), '必须写成**单个**文件（不是目录里多条 JSON）'

    m = _cache(f).load('click_bell', [51, 52])
    assert m is not None and m['mse'] == 0.5
    doc = json.loads(f.read_text(encoding='utf-8'))
    assert doc['version'] == VERSION and doc['model'] == MODEL
    assert list(doc['records']) == [scout_key('click_bell', [51, 52])]


def test_model_mismatch_is_a_miss(tmp_path):
    f = tmp_path / 'scout.json'
    _cache(f, model='model_A').store('click_bell', [51, 52], _metrics())
    other = _cache(f, model='model_B')
    assert other.load('click_bell', [51, 52]) is None, '模型不同不得复用'
    assert other.last_miss_reason == 'model_mismatch'
    assert other.record_count() == 0


def test_same_model_across_code_change_reuses(tmp_path):
    """核心收益：同模型名 ⇒ 改代码不影响命中（旧指纹方案会失效）。"""
    f = tmp_path / 'scout.json'
    _cache(f).store('t', [1], _metrics(task='t', ids=(1,)))
    m = BootstrapScoutCache(f, model=MODEL).load('t', [1])
    assert m is not None and m['task'] == 't'


def test_store_merges_without_clobbering_other_records(tmp_path):
    """`store()` 先重读再合并 ⇒ 不得盖掉别的记录。"""
    f = tmp_path / 'scout.json'
    _cache(f).store('t1', [1], _metrics(task='t1', ids=(1,)))
    _cache(f).store('t2', [2], _metrics(task='t2', ids=(2,)))
    assert _cache(f).record_count() == 2
    assert _cache(f).load('t1', [1]) is not None
    assert _cache(f).load('t2', [2]) is not None


def test_model_switch_rebuilds_file(tmp_path):
    f = tmp_path / 'scout.json'
    _cache(f, model='model_A').store('t', [1], _metrics(task='t', ids=(1,)))
    _cache(f, model='model_B').store('t2', [2], _metrics(task='t2', ids=(2,)))
    doc = json.loads(f.read_text(encoding='utf-8'))
    assert doc['model'] == 'model_B'
    assert list(doc['records']) == [scout_key('t2', [2])], '换模型后应按新模型重建'


def test_readonly_rank_does_not_write(tmp_path):
    f = tmp_path / 'scout.json'
    _cache(f, write_enabled=False).store('t', [1], _metrics(task='t', ids=(1,)))
    assert not f.exists()


def test_ids_mismatch_is_a_miss(tmp_path):
    f = tmp_path / 'scout.json'
    _cache(f).store('t', [1, 2], _metrics(task='t', ids=(1, 2)))
    c = _cache(f)
    assert c.load('t', [1, 2, 3]) is None, '请求不同 id 集合不得命中'
    assert c.last_miss_reason == 'record_missing'
    assert _cache(f).load('other', [1, 2]) is None, '不同任务不得命中'


def test_invalid_metrics_are_rejected(tmp_path):
    """结构自洽校验（沿用旧 load 的条件）：nmse 与 mse/base 不符 ⇒ 不命中。"""
    f = tmp_path / 'scout.json'
    bad = _metrics(task='t', ids=(1, 2))
    bad['nmse'] = 123.0                       # 与 mse/base 不符
    _cache(f).store('t', [1, 2], bad)
    reader = _cache(f)                      # 同一实例上读，才能断言未命中原因
    assert reader.load('t', [1, 2]) is None
    assert reader.last_miss_reason == 'metrics_invalid'


def test_corrupt_or_oversized_file_tolerated(tmp_path):
    f = tmp_path / 'scout.json'
    f.write_text('{not json', encoding='utf-8')
    c = _cache(f)
    assert c.load('t', [1]) is None and c.last_miss_reason == 'unreadable'

    f.write_text('x' * (MAX_JSON_BYTES + 10), encoding='utf-8')
    c2 = _cache(f)
    assert c2.load('t', [1]) is None and c2.last_miss_reason == 'too_large'


def test_schema_version_mismatch_is_a_miss(tmp_path):
    f = tmp_path / 'scout.json'
    f.write_text(json.dumps({'version': VERSION - 1, 'model': MODEL, 'records': {}}),
                 encoding='utf-8')
    c = _cache(f)
    assert c.load('t', [1]) is None and c.last_miss_reason == 'schema'


def test_empty_model_rejected(tmp_path):
    with pytest.raises(ValueError, match='model'):
        BootstrapScoutCache(tmp_path / 's.json', model='  ')
