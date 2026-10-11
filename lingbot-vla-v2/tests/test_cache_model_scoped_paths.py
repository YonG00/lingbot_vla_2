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


# --------------------------------------------------------------------------- #
# launcher 侧解析（真机踩过的坑：解析后的模型名被**二次解析** ⇒ 退化成空串）
# --------------------------------------------------------------------------- #
def _load_launcher():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        '_al_launch_probe', root / 'experiment/robotwin/al_launch.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launcher_resolves_model_and_scopes_file():
    """`scout_cache_file_for` 必须一次算出【模型名 + scoped 文件】，两者同源。"""
    al = _load_launcher()
    base = '/workspace/al/scout_cache/scout.json'
    ckpt = ('/models/robotwin-persistent/al_runs/train_al_v37/'
            'checkpoints/global_step_500/hf_ckpt')
    path, model = al.scout_cache_file_for(base, '', ckpt)
    assert model == 'train_al_v37/checkpoints/global_step_500/hf_ckpt'
    assert model_slug(model) in path.name


def test_launcher_model_name_not_reparsed_by_coverage():
    """**核心回归**：把已解析的模型名交回 launcher 的 `Coverage`，不得再次解析。

    真机事故：`Coverage.__init__` 里对"已解析好的模型名"又调了一次
    `resolve_model_name()`，而它会把这串当成**路径**再取尾部 4 段
    ⇒ 模型名退化成 `''` ⇒ 与缓存里的模型全不匹配 ⇒ launcher 退出码 3。
    """
    al = _load_launcher()
    base = '/workspace/al/scout_cache/scout.json'
    ckpt = ('/models/robotwin-persistent/al_runs/train_al_v37/'
            'checkpoints/global_step_500/hf_ckpt')
    path, model = al.scout_cache_file_for(base, '', ckpt)

    cov = al.Coverage(path, model)          # ← 传"已解析"的值
    assert cov.model == model, f'模型名被二次解析：{cov.model!r} != {model!r}'
    assert cov.model, '模型名不得为空串'
    assert cov.cache_file == path, 'scoped 路径必须是幂等的（不得再拼一层）'


def test_launcher_coverage_accepts_checkpoint_kwarg():
    """`check_coverage`/`Coverage` 仍可接受 checkpoint（供一次性解析的调用方使用）。"""
    al = _load_launcher()
    cov = al.check_coverage(Path('/workspace/al/scout_cache/scout.json'),
                            'my-explicit-model', {'task_a': [1, 2]})
    assert cov.model == 'my-explicit-model'
    assert cov.missing == ['task_a'], '文件不存在 ⇒ 未覆盖'


def test_launcher_migrates_legacy_file_before_coverage(tmp_path):
    """**核心回归**（真机踩过，退出码 3）：预检解析出的 scoped 文件必须**已含旧文件内容**。

    旧实现这里用了纯路径的 `scoped_cache_path()` ⇒ 指向一个**不存在**的文件
    ⇒ 覆盖检查读到 `0 条记录` ⇒ 误判"覆盖不完整"、以退出码 3 拒绝启动。
    正确实现走 `ensure_scoped_cache_file()`：先把旧的无后缀文件**复制**成 scoped 名再检查。
    """
    al = _load_launcher()
    legacy = tmp_path / 'scout.json'
    legacy.write_text(json.dumps({'version': 2, 'model': BASE_MODEL,
                                  'records': {'k': {'task': 't'}}}), encoding='utf-8')
    path, model = al.scout_cache_file_for(legacy, '', '/ckpt/train_al_v37/checkpoints/global_step_500/hf_ckpt')
    assert path != legacy and path.is_file(), '必须已把旧文件迁移成 scoped 名'
    assert json.loads(path.read_text(encoding='utf-8'))['records'] == {'k': {'task': 't'}}
    assert legacy.is_file(), '旧文件必须保留'
    cov = al.Coverage(path, model)
    assert cov.record_count() == 1, '预检必须能看到旧文件的记录（不是 0）'


def test_base_env_model_name_matches_cache_ident():
    """**核心回归**（真机踩过，训练永远起不来）：`AL_MODEL_NAME` 必须与缓存路径同源。

    真机事故：`base_env()` 用旧口径（`Path(checkpoint).name` ⇒ `hf_ckpt`）写 `AL_MODEL_NAME`，
    而 launcher 自己算缓存路径用新口径（路径尾部 4 段）⇒
      launcher 读/等  scout.train_al_v37_checkpoints_global_step_500/hf_ckpt.json
      worker  写      scout.hf_ckpt.json
    ⇒ 覆盖率恒 0/50、训练永不启动（白等 14+ 分钟、6 卡空转）。
    """
    al = _load_launcher()
    ckpt = ('/models/robotwin-persistent/al_runs/train_al_v37/'
            'checkpoints/global_step_500/hf_ckpt')
    base = '/workspace/al/scout_cache/scout.json'
    # 缓存路径侧（launcher 门控读它）
    gate_path, gate_model = al.scout_cache_file_for(base, '', ckpt)
    # 环境变量侧（worker 写它）
    args = al.build_parser().parse_args([
        '--checkpoint', ckpt, '--scout-cache-file', base, '--min-free-gb', '0'])
    env = al.base_env(args=args, repo=Path('/tmp/repo'), python=Path('/tmp/py'),
                      al_cfg=Path('/tmp/al.yaml'), train_config=Path('/tmp/c.yaml'),
                      split_dir=Path('/tmp/split'), phases=Path('/tmp/phases'),
                      checkpoint=Path(ckpt), manifest=Path('/tmp/m.json'),
                      baseline=Path('/tmp/b.json'), norm=Path('/tmp/n.json'),
                      dtype='bfloat16')
    assert env['AL_MODEL_NAME'] == gate_model, (
        f"AL_MODEL_NAME={env['AL_MODEL_NAME']!r} 与门控标识 {gate_model!r} 不一致 ⇒ "
        'worker 会写到另一个文件、门控永远看不到')
    assert env['AL_MODEL_NAME'] != 'hf_ckpt', '不得退回 basename 口径'
    # worker 侧按 AL_MODEL_NAME 算出的路径必须与门控路径一致
    worker_path = scoped_cache_path(base, env['AL_MODEL_NAME'])
    assert worker_path == gate_path, f'worker 写 {worker_path} ≠ 门控读 {gate_path}'


def test_explicit_model_name_still_wins_in_env():
    """显式 `--model-name` 仍优先（且两边一致）。"""
    al = _load_launcher()
    base = '/workspace/al/scout_cache/scout.json'
    ckpt = '/any/where/hf_ckpt'
    gate_path, gate_model = al.scout_cache_file_for(base, 'my-model', ckpt)
    assert gate_model == 'my-model'
    args = al.build_parser().parse_args([
        '--checkpoint', ckpt, '--scout-cache-file', base,
        '--model-name', 'my-model', '--min-free-gb', '0'])
    env = al.base_env(args=args, repo=Path('/tmp/repo'), python=Path('/tmp/py'),
                      al_cfg=Path('/tmp/al.yaml'), train_config=Path('/tmp/c.yaml'),
                      split_dir=Path('/tmp/split'), phases=Path('/tmp/phases'),
                      checkpoint=Path(ckpt), manifest=Path('/tmp/m.json'),
                      baseline=Path('/tmp/b.json'), norm=Path('/tmp/n.json'),
                      dtype='bfloat16')
    assert env['AL_MODEL_NAME'] == 'my-model' == gate_model
