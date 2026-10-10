"""`al_launch.py --env` 与 `AL_*` 透传回归（CPU；2026-10-10 加）。

背景（真机踩坑）
----------------
启动器给扫描 worker 与训练**白名单构造**环境变量（有意为之：可审计、不受调用者 shell 影响）。
但因此父进程 `export AL_HARDNESS_SHARD=1` **到不了训练进程** —— 实测 rank 的
`/proc/<pid>/environ` 里根本没有该变量，于是"开了分片"其实没生效，白跑一轮。

本用例直接对 `--dry-run` 的环境变量段断言（走真实 `base_env()`），不启动任何进程。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'experiment' / 'robotwin' / 'al_launch.py'


def _dry_run(extra_args):
    env = dict(os.environ)
    env.setdefault('PYTHONPATH', str(ROOT))
    proc = subprocess.run([sys.executable, str(LAUNCHER), '--dry-run', *extra_args],
                          capture_output=True, text=True, errors='replace',
                          timeout=300, cwd=str(ROOT), env=env)
    return proc


def _env_values(text: str) -> dict:
    """从 dry-run 输出的 `KEY=VALUE` 行里抓环境变量。"""
    out = {}
    for line in text.splitlines():
        m = re.match(r'^\s{2,}([A-Z][A-Z0-9_]*)=(.*)$', line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_env_flag_reaches_worker_and_train_plans():
    proc = _dry_run(['--env', 'AL_HARDNESS_SHARD=1', '--env', 'AL_HARDNESS_LOG_SEC=7'])
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    blob = proc.stdout + proc.stderr
    vals = _env_values(blob)
    assert vals.get('AL_HARDNESS_SHARD') == '1', f'--env 未进入环境变量段:\n{blob[-1500:]}'
    assert vals.get('AL_HARDNESS_LOG_SEC') == '7', f'--env 未进入环境变量段:\n{blob[-1500:]}'


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_parent_al_env_is_passed_through():
    """父进程里已存在的 AL_* 必须自动透传（不被白名单吞掉）。"""
    env = dict(os.environ, AL_HARDNESS_SHARD='1', PYTHONPATH=str(ROOT))
    proc = subprocess.run([sys.executable, str(LAUNCHER), '--dry-run'],
                          capture_output=True, text=True, errors='replace',
                          timeout=300, cwd=str(ROOT), env=env)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    vals = _env_values(proc.stdout + proc.stderr)
    assert vals.get('AL_HARDNESS_SHARD') == '1', '父进程的 AL_* 没有透传到子进程环境'


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_bad_env_flag_is_rejected():
    proc = _dry_run(['--env', '=novalue'])
    blob = proc.stdout + proc.stderr
    assert proc.returncode != 0, f'非法 --env 应被拒绝:\n{blob[-800:]}'
    assert 'env' in blob.lower()


# --------------------------------------------------------------------------- #
# 启动器空间检查：hardness-cache **不得**被 20G 下限拦死（2026-10-10 自锁事故）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_hardness_cache_is_not_blocked_by_min_free_gb(tmp_path):
    """真机教训：把体积只有几十 KB 的 hardness 缓存塞进「20G 下限」检查后，
    `/workspace` 只剩 19.7G ⇒ **任何启用缓存的启动都被自己拦死**（dry-run 退出码 2）。

    这里用 dry-run 断言：即使把下限抬到远高于该目录可用空间，hardness-cache 也不报错。
    """
    hc = tmp_path / 'hardness_cache'
    hc.mkdir()
    proc = _dry_run(['--hardness-cache', str(hc), '--min-free-gb', '100000'])
    blob = proc.stdout + proc.stderr
    # 其它三个目录会因 100000G 下限而报错 ⇒ 只看"有没有点名 hardness-cache"
    assert 'hardness-cache' not in blob, (
        f'hardness-cache 仍被空间检查拦下（不应参与 20G 下限）：\n{blob[-1200:]}')


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_hardness_cache_missing_dir_is_allowed(tmp_path):
    """缓存目录不存在**允许**（`store()` 会 `mkdir(parents=True)` 自建）——不应因此拦启动。"""
    proc = _dry_run(['--hardness-cache', str(tmp_path / 'nope' / 'hc'), '--min-free-gb', '0'])
    blob = proc.stdout + proc.stderr
    assert 'hardness-cache 不可写' not in blob and '上层路径不是目录' not in blob, blob[-1200:]


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_hardness_cache_default_is_persistent_volume():
    """默认必须是持久卷路径（用户 2026-10-10 定：持久化）。"""
    proc = _dry_run(['--min-free-gb', '0'])
    blob = proc.stdout + proc.stderr
    assert '/workspace/al/hardness_cache' in blob, blob[-1200:]


# --------------------------------------------------------------------------- #
# 指纹计算默认必须**关闭**（2026-10-10：缓存改为显式文件 + 模型名后，指纹不再参与寻址）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_fingerprint_computation_is_skipped_by_default():
    """默认不得去哈希 11.9 GiB 权重分片：dry-run 应打印「跳过计算」且不出现「正在哈希」。"""
    proc = _dry_run(['--min-free-gb', '0'])
    blob = proc.stdout + proc.stderr
    assert '正在哈希' not in blob, (
        '默认仍在哈希权重分片（11.9 GiB，每轮多花 1–3 分钟 I/O）——'
        '指纹已不参与缓存寻址，应改为 --compute-fingerprint 显式开启')
    assert 'fingerprint] 跳过计算' in blob or '跳过计算' in blob, blob[-1200:]


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_explicit_compute_fingerprint_flag_is_accepted():
    """显式开关必须被接受（不报 argparse 错）——dry-run 下因路径不齐会跳过，但不得崩。"""
    proc = _dry_run(['--compute-fingerprint', '--min-free-gb', '0'])
    blob = proc.stdout + proc.stderr
    assert 'unrecognized arguments' not in blob, blob[-800:]
    assert '--compute-fingerprint' not in blob or 'error' not in blob.lower()


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_deprecated_fingerprint_env_is_not_exported():
    """已废弃的 AL_SCOUT_CACHE_FINGERPRINT 不得再出现在子进程 env 里。"""
    proc = _dry_run(['--min-free-gb', '0'])
    blob = proc.stdout + proc.stderr
    assert 'AL_SCOUT_CACHE_FINGERPRINT=' not in blob, '仍在透传已废弃的指纹 env'
    # 新 env 必须都在
    for key in ('AL_SCOUT_CACHE_FILE=', 'AL_HARDNESS_CACHE_FILE=', 'AL_MODEL_NAME='):
        assert key in blob, f'{key} 未透传'


@pytest.mark.skipif(not LAUNCHER.is_file(), reason='缺 al_launch.py')
def test_scout_and_hardness_cache_files_are_distinct_paths():
    """两个缓存文件必须分属不同目录（避免互相覆盖）。"""
    proc = _dry_run(['--min-free-gb', '0'])
    blob = proc.stdout + proc.stderr
    assert '/workspace/al/scout_cache/scout.json' in blob
    assert '/workspace/al/hardness_cache/hardness.json' in blob
