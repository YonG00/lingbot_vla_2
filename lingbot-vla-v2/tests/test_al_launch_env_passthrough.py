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
