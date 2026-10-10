"""启动程序并发占用检查的**自匹配**回归（CPU；2026-10-10 实测坑）。

背景
----
`al_launch.py` 的"是否已有训练在跑"检查（`list_other_runs`）是拿工具名去匹配
`/proc/<pid>/cmdline`。而**调用它的那条 shell 的命令行里也含 `al_launch.py`**：

    bash -c 'cd … && python experiment/robotwin/al_launch.py --dry-run | tail'

旧实现只排除了 `os.getpid()`（python 自己），于是把**包装 shell** 当成"已有训练在跑"，
dry-run 直接以退出码 2 拒绝启动。这与仓库铁律里 `pgrep -f` 自匹配是同一个坑。

本用例在子进程里验证（避免 pytest 自己的命令行里含该字面量而污染判断）：
  * 包装 shell（cmdline 含 `al_launch.py`）**不得**被匹配；
  * 真的在跑 `al_launch.py` 的 python 进程**必须**被匹配（护栏没被削弱）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOKEN = 'al_launch' + '.py'          # 拆开写：本文件命令行里不出现该字面量


@pytest.mark.skipif(sys.platform != 'linux', reason='依赖 /proc')
def test_busy_check_ignores_wrapper_shell_but_still_sees_real_run(tmp_path):
    child = Path(__file__).with_name('_self_match_child.py')
    proc = subprocess.run([sys.executable, str(child), str(tmp_path)],
                          capture_output=True, text=True, errors='replace', timeout=120)
    assert proc.returncode == 0, f'子进程失败\nstdout={proc.stdout}\nstderr={proc.stderr}'
    data = json.loads((tmp_path / 'result.json').read_text(encoding='utf-8'))

    # 1) 包装 shell 不能被当成"已有训练在跑"（旧实现在这里会失败）
    assert data['matched_wrapper'] == [], (
        '包装 shell 被误判为在跑的训练/启动器（自匹配）: '
        f"{data['matched_wrapper']}")
    # 2) 直接祖先也不能被误判
    assert data['matched_parent'] == [], f"父进程被误判: {data['matched_parent']}"
    # 3) 护栏没被削弱：真在跑的 al_launch.py 进程必须被抓到
    assert data['matched_real'], '真在跑的启动器进程没有被检出（并发护栏失效）'
    # 4) 自己与祖先确实进了跳过集合
    assert data['skip_pids'] and data['parent_pid'] in data['skip_pids']
