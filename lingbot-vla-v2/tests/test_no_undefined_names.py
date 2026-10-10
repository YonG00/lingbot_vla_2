"""静态审计：本仓库**本次改造涉及的**文件不得有"未定义名"这类真机才会炸的错误。

为什么单独写这条
----------------
2026-10-10 连续三次真机启动失败，全是同一类问题——**改名/删方法后的残留引用**：
  1. `NameError: name 'cache_dir' is not defined`（缓存目录→单文件改造漏改 14 处）；
  2. `AttributeError: Coverage has no attribute 'json_files'`（方法改名漏改调用点）；
  3. `undefined name '_scout_model'`（定义在使用点之后、且跨函数作用域）；
这些分支**本地 dry-run 走不到**（本机没有 /workspace 下的 manifest/split 数据，dry-run 提前返回），
单元测试也看不见 ⇒ 只能等真机炸。用 `pyflakes` 做源码级审计，能在本地一次拦住全部同类问题。

审计范围**只包含本次改造动过的文件**（存量告警不在本次清理范围，避免把无关噪音变成红灯）；
`undefined name` 与 `undefined local` 视为**硬失败**，其余（未用导入等）仅打印提示。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: 本次改造涉及的文件（改动这些文件时请同步维护本清单）
AUDITED = [
    'experiment/robotwin/al_launch.py',
    'lingbotvla/auto_learning/hardness_cache.py',
    'lingbotvla/auto_learning/scout_cache.py',
    'lingbotvla/auto_learning/config.py',
    'lingbotvla/auto_learning/sampling/hardness_scan.py',
    'lingbotvla/auto_learning/real/build.py',
    'lingbotvla/auto_learning/real/backend.py',
    'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py',
    'lingbotvla/models/vla/lingbot_vla/flex_attention.py',
    'lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py',
]

HARD_FAIL_MARKERS = ('undefined name', 'undefined local')


def _pyflakes(paths: list[str]) -> str:
    proc = subprocess.run([sys.executable, '-m', 'pyflakes', *paths],
                          cwd=ROOT, capture_output=True, text=True)
    if proc.returncode not in (0, 1):        # 0=干净, 1=有告警, 其他=工具本身出错
        pytest.skip(f'pyflakes 不可用：{proc.stderr[:200]}')
    return proc.stdout


@pytest.mark.parametrize('rel', AUDITED)
def test_no_undefined_names_in_audited_file(rel):
    p = ROOT / rel
    assert p.is_file(), f'审计清单里的文件不存在：{rel}（清单需同步维护）'
    out = _pyflakes([rel])
    hard = [l for l in out.splitlines() if any(m in l for m in HARD_FAIL_MARKERS)]
    assert not hard, ('存在"未定义名"（真机必然 NameError / AttributeError）：\n'
                      + '\n'.join(hard))


def test_audited_list_matches_real_files():
    """清单本身必须有效（防止改名后审计悄悄失效）。"""
    missing = [r for r in AUDITED if not (ROOT / r).is_file()]
    assert not missing, f'审计清单里有不存在的文件：{missing}'
