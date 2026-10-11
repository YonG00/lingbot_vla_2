"""缓存文件**按模型分名**（`scout.<model-slug>.json` / `hardness.<model-slug>.json`）的契约测试。

背景（2026-10-11 用户要求）：缓存是"固定文件名 + 单个 `model` 槽位"，
换模型时 `store()` 会重建并 `os.replace()` **覆盖同一个文件** ⇒ 上一份缓存被静默丢掉、
没有任何自动存档。改为把模型标识编进文件名后：每个模型一份文件、互不覆盖、旧文件自然留档。

本文件覆盖：
  * slug 生成（路径分隔符清洗、长度保护、空值）；
  * scoped 路径（幂等、`AL_CACHE_SHARED=1` 回退、后缀保留）；
  * **旧文件自动迁移**（只复制、不删除；非 rank0 不迁移）；
  * 两个缓存类内部都用 scoped 路径（写与读一致）；
  * 两个不同模型的缓存**不会互相覆盖**（核心回归）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lingbotvla.auto_learning.cache_paths import (
    MAX_SLUG_LEN,
    ensure_scoped_cache_file,
    model_slug,
    scoped_cache_path,
)
from lingbotvla.auto_learning.hardness_cache import HardnessSampleCache
from lingbotvla.auto_learning.scout_cache import BootstrapScoutCache

TRAIN_MODEL = 'train_al_v37/checkpoints/global_step_500/hf_ckpt'
BASE_MODEL = 'workspace/models/robbyant_lingbot-vla-v2-6b-bf16'


# --------------------------------------------------------------------------- #
# slug 与路径
# --------------------------------------------------------------------------- #
def test_slug_replaces_path_separators():
    assert model_slug(TRAIN_MODEL) == 'train_al_v37_checkpoints_global_step_500_hf_ckpt'


def test_slug_empty_for_blank():
    assert model_slug('') == ''
    assert model_slug('   ') == ''
    assert model_slug('///') == ''


def test_slug_truncates_with_hash_and_stays_unique():
    long_a = 'x' * 300 + '/aaa'
    long_b = 'x' * 300 + '/bbb'
    sa, sb = model_slug(long_a), model_slug(long_b)
    assert len(sa) <= MAX_SLUG_LEN and len(sb) <= MAX_SLUG_LEN
    assert sa != sb, '同名截断后必须靠 hash 区分'


def test_scoped_path_basic():
    p = scoped_cache_path('/workspace/al/scout_cache/scout.json', TRAIN_MODEL)
    assert p == Path('/workspace/al/scout_cache/scout.train_al_v37_checkpoints_'
                     'global_step_500_hf_ckpt.json')


def test_scoped_path_is_idempotent():
    once = scoped_cache_path('/a/scout.json', TRAIN_MODEL)
    twice = scoped_cache_path(once, TRAIN_MODEL)
    assert once == twice, '二次拼接必须幂等（否则路径会越滚越长）'


def test_scoped_path_keeps_non_json_suffix():
    p = scoped_cache_path('/a/scout.cache', TRAIN_MODEL)
    assert p.suffix == '.cache' and TRAIN_MODEL.replace('/', '_') in p.name


def test_scoped_path_falls_back_when_model_blank():
    assert scoped_cache_path('/a/scout.json', '') == Path('/a/scout.json')
    assert scoped_cache_path('/a/scout.json', None) == Path('/a/scout.json')


def test_scoped_path_shared_env_escape_hatch(monkeypatch):
    monkeypatch.setenv('AL_CACHE_SHARED', '1')
    assert scoped_cache_path('/a/scout.json', TRAIN_MODEL) == Path('/a/scout.json')


def test_different_models_get_different_files():
    a = scoped_cache_path('/a/scout.json', TRAIN_MODEL)
    b = scoped_cache_path('/a/scout.json', BASE_MODEL)
    assert a != b


# --------------------------------------------------------------------------- #
# 旧文件自动迁移
# --------------------------------------------------------------------------- #
def test_migration_copies_legacy_file(tmp_path):
    legacy = tmp_path / 'scout.json'
    legacy.write_text('{"version":2,"model":"m","records":{}}', encoding='utf-8')
    scoped = ensure_scoped_cache_file(legacy, 'model-A', write_enabled=True)
    assert scoped != legacy and scoped.is_file()
    assert legacy.is_file(), '迁移必须**只复制、不删除**旧文件'
    assert json.loads(scoped.read_text(encoding='utf-8'))['model'] == 'm'


def test_migration_skipped_when_not_writer(tmp_path):
    legacy = tmp_path / 'scout.json'
    legacy.write_text('{}', encoding='utf-8')
    scoped = ensure_scoped_cache_file(legacy, 'model-A', write_enabled=False)
    assert not scoped.is_file(), '非写盘 rank 不得迁移（否则可能误建空文件）'


def test_migration_does_not_overwrite_existing_scoped(tmp_path):
    legacy = tmp_path / 'scout.json'
    legacy.write_text('{"who":"legacy"}', encoding='utf-8')
    scoped = scoped_cache_path(legacy, 'model-A')
    scoped.write_text('{"who":"scoped"}', encoding='utf-8')
    ensure_scoped_cache_file(legacy, 'model-A', write_enabled=True)
    assert json.loads(scoped.read_text(encoding='utf-8'))['who'] == 'scoped'


# --------------------------------------------------------------------------- #
# 两个缓存类都用 scoped 路径
# --------------------------------------------------------------------------- #
def test_hardness_cache_uses_scoped_path(tmp_path):
    base = tmp_path / 'hardness.json'
    c = HardnessSampleCache(base, model=TRAIN_MODEL)
    assert c.path != base and model_slug(TRAIN_MODEL) in c.path.name
    c.store('t', {1: 0.5})
    assert c.path.is_file()
    assert json.loads(c.path.read_text(encoding='utf-8'))['model'] == TRAIN_MODEL


def test_scout_cache_uses_scoped_path(tmp_path):
    base = tmp_path / 'scout.json'
    c = BootstrapScoutCache(base, model=TRAIN_MODEL, write_enabled=True)
    assert c.path != base and model_slug(TRAIN_MODEL) in c.path.name
    c.store('task_a', [1, 2], {'task': 'task_a', 'episode_ids': [1, 2]})
    assert c.path.is_file()


def test_two_models_do_not_overwrite_each_other(tmp_path):
    """**核心回归**：base 与训练后权重的缓存必须各存一份、互不覆盖。"""
    base = tmp_path / 'hardness.json'
    c_base = HardnessSampleCache(base, model=BASE_MODEL)
    c_base.store('t', {1: 0.1})
    c_train = HardnessSampleCache(base, model=TRAIN_MODEL)
    c_train.store('t', {1: 0.9})

    assert c_base.path != c_train.path
    raw_base = json.loads(c_base.path.read_text(encoding='utf-8'))
    raw_train = json.loads(c_train.path.read_text(encoding='utf-8'))
    assert raw_base['model'] == BASE_MODEL and raw_base['tasks']['t']['losses'] == {'1': 0.1}
    assert raw_train['model'] == TRAIN_MODEL and raw_train['tasks']['t']['losses'] == {'1': 0.9}


def test_shared_env_restores_legacy_single_file(tmp_path, monkeypatch):
    monkeypatch.setenv('AL_CACHE_SHARED', '1')
    base = tmp_path / 'hardness.json'
    c = HardnessSampleCache(base, model=TRAIN_MODEL)
    assert c.path == base, 'AL_CACHE_SHARED=1 时必须回到旧的单文件行为'


# --------------------------------------------------------------------------- #
# launcher 与缓存类必须解析出**同一个文件**
# --------------------------------------------------------------------------- #
def test_launcher_uses_same_path_as_cache_class(tmp_path):
    """预检（`Coverage`）读的文件必须与训练（缓存类）写的文件**完全一致**。

    两条路径由 `al_launch.scout_cache_file_for` 与 `cache_paths.scoped_cache_path` 给出；
    这里不导入 launcher（它依赖 argparse 全局状态），而是断言二者**同规则**：
    相同的 (文件, 模型) ⇒ 相同路径。
    """
    base = tmp_path / 'scout.json'
    c = BootstrapScoutCache(base, model=TRAIN_MODEL, write_enabled=True)
    assert c.path == scoped_cache_path(base, TRAIN_MODEL)
