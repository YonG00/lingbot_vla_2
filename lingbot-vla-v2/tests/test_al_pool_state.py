"""池子独立存档（`pool_state.json`）的 CPU 契约测试 —— 2026-10-12 用户要求。

背景
----
Auto Learning 的池子（每任务 status/attempts/lp50/`sample_probs` …）原本只随 **DCP** 的
`extra_state` 走。从 **HF 权重**起新 run 时框架判 `_al_resume_mode == "fresh"`
⇒ 池子从零 ⇒ 已 PASS 的任务被重训、刚 DEFER 的被立刻重试，白花算力。

本文件锁住"池子独立导出/导入"的三类契约：
  A. **数据完整**：导出→读回逐字段相等（含 `sample_probs` 与 `sampler_rng`）；
  B. **不添乱**：文件缺失/损坏 → None（不抛）；导出绝不抛；原子写不留 `.tmp`；
     镜像目录按 keep 清理且不碰正本；
  C. **接线正确**（静态）：`build.py` 里"走 DCP 时跳过导入"的分支存在且被
     `enable_resume`/`load_checkpoint_path` 把守；`train_lingbotvla.py` 的导出挂在
     `save_hf_checkpoint_best_effort()` 内（⇒ 周期/轮末/收尾三个存档点全覆盖）。

⚠️ 静态断言（C 组）是有意为之：`build.py` / `train_lingbotvla.py` 顶层 import torch，
在本机（无 ROCm）无法导入 —— 用读源码的方式锁"接线"，把"行为"锁在 A/B 组的纯单元里。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lingbotvla.auto_learning.state import pool_state as ps

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# 夹具：真实形态的 al_state（sample_probs 用字符串键 —— 与 DCP 里一致）
# --------------------------------------------------------------------------- #
def _al_state() -> dict:
    return {
        'version': 1,
        'registry': {
            'order': ['task_a', 'task_b'],
            'records': {
                'task_a': {'task_name': 'task_a', 'status': 'PASS', 'attempt_count': 1,
                           'sample_probs': {'92715': 0.0002287, '92716': 0.0002313}},
                'task_b': {'task_name': 'task_b', 'status': 'DEFER', 'attempt_count': 1,
                           'lp50': -0.0265, 'sample_probs': {'93000': 0.5}},
            },
        },
        'sampler': {'batch_size': 35, 'ratio_layout': [1, 2]},
        'sampler_rng': [3, [1, 2, 3], None],
        'step_in_unit': 12, 'unit_steps': 50, 'steps_done': 2450, 'at_boundary': False,
    }


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """每个用例都从干净环境开始（镜像目录默认指向临时位置、不污染真机目录）。"""
    monkeypatch.delenv(ps.POOL_ENV, raising=False)
    monkeypatch.delenv(ps.NO_IMPORT_ENV, raising=False)
    monkeypatch.delenv(ps.EXPORT_ENV, raising=False)
    monkeypatch.delenv(ps.RESTORE_RNG_ENV, raising=False)
    monkeypatch.setenv(ps.MIRROR_ENV, '')
    monkeypatch.setenv(ps.PRUNE_ENV, '2')
    yield


# --------------------------------------------------------------------------- #
# A. 数据完整
# --------------------------------------------------------------------------- #
def test_export_then_load_roundtrip_is_field_exact(tmp_path):
    """导出→读回：registry / sampler / sampler_rng / 标量**逐字段相等**。"""
    al = _al_state()
    out = ps.export_pool_state(tmp_path, al, step=2400,
                               source_model_path='/workspace/al_v41_step801_hf/global_step_801/hf_ckpt',
                               source_checkpoint=str(tmp_path / 'global_step_2400'))
    assert out['path'] == str(tmp_path / ps.POOL_FILENAME)

    payload = ps.load_pool_state(out['path'])
    assert payload is not None
    back = payload['auto_learning']
    assert back['registry'] == al['registry']            # ← 含 sample_probs 4548 项量级的数据
    assert back['sampler'] == al['sampler']
    assert back['sampler_rng'] == al['sampler_rng']
    for key in ('version', 'step_in_unit', 'unit_steps', 'steps_done', 'at_boundary'):
        assert back[key] == al[key]
    # 元信息（用于"这是哪份权重/哪一步导出的"）
    assert payload['global_step'] == 2400
    assert payload['source_model_path'].endswith('hf_ckpt')
    assert payload['has_sampler_rng'] is True


def test_export_writes_hf_ckpt_copy(tmp_path):
    """`extra_dirs` 里的副本（HF 目录）要真的落盘 ⇒ 新 run 同目录即可读到。"""
    hf_dir = tmp_path / 'global_step_2400' / 'hf_ckpt'
    hf_dir.mkdir(parents=True)
    out = ps.export_pool_state(tmp_path, _al_state(), step=2400, extra_dirs=[hf_dir])
    assert (hf_dir / ps.POOL_FILENAME).is_file()
    assert str(hf_dir / ps.POOL_FILENAME) in out['mirrors']
    # 也支持直接给"文件名"（父目录已存在 —— 口径：已存在目录才当目录）
    out2 = ps.export_pool_state(tmp_path, _al_state(), step=2400,
                                extra_dirs=[hf_dir / 'second.json'])
    assert (hf_dir / 'second.json').is_file() and str(hf_dir / 'second.json') in out2['mirrors']
    # 目录**不存在**时按文件名处理并自动建父目录（避免"当目录"的歧义）
    out3 = ps.export_pool_state(tmp_path, _al_state(), step=2400,
                                extra_dirs=[tmp_path / 'newsub' / 'third.json'])
    assert (tmp_path / 'newsub' / 'third.json').is_file()


def test_summarize_pool_counts_status_and_probs():
    """摘要口径：状态计数 + 权重采样项数（日志里可核对）。"""
    s = ps.summarize_pool(_al_state())
    assert s['n_tasks'] == 2
    assert s['counts'] == {'PASS': 1, 'DEFER': 1}
    assert s['n_sample_probs'] == 3


# --------------------------------------------------------------------------- #
# B. 不添乱
# --------------------------------------------------------------------------- #
def test_load_missing_and_broken_return_none(tmp_path):
    """文件不存在 / 不是 JSON / 结构不对 ⇒ 一律 None（调用方据此从零开始）。"""
    assert ps.load_pool_state(None) is None
    assert ps.load_pool_state(tmp_path / 'nope.json') is None

    bad = tmp_path / 'bad.json'
    bad.write_text('{not json', encoding='utf-8')
    assert ps.load_pool_state(bad) is None

    wrong = tmp_path / 'wrong.json'
    wrong.write_text(json.dumps({'version': 1}), encoding='utf-8')   # 缺 auto_learning
    assert ps.load_pool_state(wrong) is None


def test_export_is_atomic_and_leaves_no_tmp(tmp_path):
    """原子写：写完后目录里**没有** `.tmp` 残留。"""
    ps.export_pool_state(tmp_path, _al_state(), step=1)
    assert list(tmp_path.glob('.pool_state.*.tmp')) == []
    assert (tmp_path / ps.POOL_FILENAME).is_file()


def test_export_safely_never_raises(monkeypatch, tmp_path):
    """`export_pool_state_safely` 在内部炸掉时**返回 error 而不抛**（存档路径硬要求）。"""
    def _boom(**kwargs):
        raise RuntimeError('disk full (simulated)')

    monkeypatch.setattr(ps, 'export_pool_state', _boom)
    res = ps.export_pool_state_safely(None, ckpt_dir=tmp_path, al_state=_al_state(), step=1)
    assert 'error' in res and 'disk full' in res['error']


def test_export_disabled_by_env(monkeypatch, tmp_path):
    """`AL_POOL_STATE_EXPORT=0` ⇒ 跳过导出（且不写文件）。"""
    monkeypatch.setenv(ps.EXPORT_ENV, '0')
    res = ps.export_pool_state_safely(None, ckpt_dir=tmp_path, al_state=_al_state(), step=1)
    assert res.get('skipped')
    assert not (tmp_path / ps.POOL_FILENAME).exists()


def test_export_skips_when_no_al_state(tmp_path):
    """非 AL run（`_al_hook is None`）⇒ `skipped=no_al_state`，不写文件。"""
    res = ps.export_pool_state(tmp_path, None, step=1)
    assert res['skipped'] == 'no_al_state'
    assert not (tmp_path / ps.POOL_FILENAME).exists()


def test_mirror_and_prune(tmp_path, monkeypatch):
    """镜像目录：按 step 命名；`keep=2` 时只留最近 2 份；**绝不碰正本目录**。"""
    mirror = tmp_path / 'mirror'
    monkeypatch.setenv(ps.MIRROR_ENV, str(mirror))
    ckpt = tmp_path / 'ckpts'
    for step in (100, 200, 300):
        ps.export_pool_state(ckpt / f'global_step_{step}', _al_state(), step=step,
                             source_model_path='/m/v41/global_step_801/hf_ckpt', prune_keep=2)
    kept = sorted(p.name for p in mirror.glob('*.json'))
    assert len(kept) == 2, kept
    assert any('step300' in n for n in kept) and any('step200' in n for n in kept)
    # 正本（每个存档旁）一个都不能少
    assert (ckpt / 'global_step_100' / ps.POOL_FILENAME).is_file()
    assert (ckpt / 'global_step_300' / ps.POOL_FILENAME).is_file()


def test_candidates_priority_and_find(tmp_path, monkeypatch):
    """查找优先级：显式 `AL_POOL_STATE_FILE` > model_path 同目录 > run 目录 > 镜像（按 step 降序）。"""
    model = tmp_path / 'hf_ckpt'
    model.mkdir()
    (model / ps.POOL_FILENAME).write_text(json.dumps({'auto_learning': {}}), encoding='utf-8')
    run = tmp_path / 'run'
    run.mkdir()
    (run / ps.POOL_FILENAME).write_text(json.dumps({'auto_learning': {}}), encoding='utf-8')

    cands = ps.pool_candidates(str(model), run_dir=run)
    assert cands[0] == model / ps.POOL_FILENAME          # model_path 同目录优先（池子跟着权重走）
    assert cands[1] == run / ps.POOL_FILENAME
    assert ps.find_pool_state_file(str(model), run_dir=run) == model / ps.POOL_FILENAME

    explicit = tmp_path / 'explicit.json'
    explicit.write_text(json.dumps({'auto_learning': {}}), encoding='utf-8')
    monkeypatch.setenv(ps.POOL_ENV, str(explicit))
    assert ps.pool_candidates(str(model), run_dir=run)[0] == explicit
    assert ps.find_pool_state_file(str(model), run_dir=run) == explicit


def test_candidates_mirror_prefers_highest_step(tmp_path, monkeypatch):
    """镜像目录有多份时 ⇒ 取 **step 最大** 的那份（= 最新池子）。"""
    mirror = tmp_path / 'mirror'
    mirror.mkdir()
    monkeypatch.setenv(ps.MIRROR_ENV, str(mirror))
    model = '/m/v41/global_step_801/hf_ckpt'
    for step in (400, 2400, 1200):
        ps.export_pool_state(tmp_path / f'c{step}', _al_state(), step=step, source_model_path=model)
    found = ps.find_pool_state_file(model)
    assert found is not None and 'step2400' in found.name


def test_describe_task_set_mismatch_is_readable():
    """差异文本必须**摊开**（多出来 / 缺少 / 共有），不能只说"不一致"（用户反馈读不懂）。"""
    assert ps.describe_task_set_mismatch(['a', 'b'], ['a', 'b']) is None
    msg = ps.describe_task_set_mismatch(['task_a', 'task_b', 'task_c'], ['task_a', 'task_d'])
    assert msg is not None
    assert 'task_b' in msg and 'task_c' in msg          # 池子里有、manifest 没有
    assert 'task_d' in msg                              # manifest 有、池子里没有
    assert '两侧都有：1 个' in msg
    assert ps.NO_IMPORT_ENV in msg                      # 给出"如何跳过导入"


def test_switch_readers(monkeypatch):
    """三个开关读取口径（都在进程启动时读一次）。"""
    assert ps.export_enabled() is True and ps.import_disabled() is False
    assert ps.restore_rng_enabled() is False            # sampler_rng **默认不恢复**
    monkeypatch.setenv(ps.NO_IMPORT_ENV, '1')
    monkeypatch.setenv(ps.RESTORE_RNG_ENV, '1')
    assert ps.import_disabled() is True
    assert ps.restore_rng_enabled() is True


# --------------------------------------------------------------------------- #
# C. 接线（静态断言 —— 两个文件顶层 import torch，本机无法导入）
# --------------------------------------------------------------------------- #
def test_build_py_hook_skips_when_resuming_from_dcp():
    """`build.py`：导入钩子必须被 `enable_resume / load_checkpoint_path` 把守。"""
    src = (REPO / 'lingbotvla' / 'auto_learning' / 'real' / 'build.py').read_text(encoding='utf-8')
    assert 'from ..state import pool_state as pool_state_mod' in src
    assert '_will_restore_from_dcp' in src
    assert re.search(r"_will_restore_from_dcp\s*=\s*bool\(", src)
    assert 'enable_resume' in src and 'load_checkpoint_path' in src
    # 必须"跳过"而不是"照旧导入"（否则走 DCP 时日志会误导成"丢状态"）
    assert '跳过 pool_state.json 导入' in src
    # 导入必须发生在 Scheduler 构造之后（registry 是它的属性）、Sampler 之前
    i_sched = src.index('scheduler = Scheduler(backend, cfg')
    i_pool = src.index('_pool_note: Optional[str] = None')
    assert i_sched < i_pool


def test_train_py_exports_inside_save_hf_helpers():
    """`train_lingbotvla.py`：导出挂在 `save_hf_checkpoint_best_effort()` 内 ⇒ 三个存档点全覆盖。"""
    src = (REPO / 'tasks' / 'vla' / 'train_lingbotvla.py').read_text(encoding='utf-8')
    assert 'def _export_pool_state_here(' in src
    assert '_export_pool_state_here(checkpoint_path, step)' in src
    # 必须在 save_hf_checkpoint_best_effort 的函数体里（缩进 8 空格 = 嵌套函数内部）
    i_hf = src.index('def save_hf_checkpoint_best_effort(')
    i_call = src.index('_export_pool_state_here(checkpoint_path, step)')
    i_next = src.index('def _export_pool_state_here(')
    assert i_hf < i_call < i_next, '导出调用必须落在 save_hf_checkpoint_best_effort 之后、辅助函数之前'
    assert '\n        _export_pool_state_here(checkpoint_path, step)' in src
    # 只 rank0 写、且失败只告警
    assert "args.train.global_rank != 0" in src
    assert '_export_pool_state_safely(' in src


def test_step_offset_auto_derive_present():
    """`train_lingbotvla.py`：未显式指定时从 `model_path` 推导 `step_offset`（编号接续）。"""
    src = (REPO / 'tasks' / 'vla' / 'train_lingbotvla.py').read_text(encoding='utf-8')
    # 只认 global_step_<N>（不认裸数字 ⇒ 不会把 `v41` 当 41 步）
    assert 'global_step_(\\d+)' in src, '必须按 global_step_<N> 解析'
    assert 're.search(r"global_step_' in src
    assert '自动从模型路径推导' in src
    # 推导必须在 total_train_steps 计算之前（否则 LR 调度步数会算错）
    i_off = src.index('自动从模型路径推导')
    i_tot = src.index('total_train_steps = args.train.train_steps * args.train.num_train_epochs')
    assert i_off < i_tot
