#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""子进程端：验证 `list_other_runs` 的自匹配行为。

    python _self_match_child.py <临时目录>

流程：
  1. 起一个**命令行里含 `al_launch.py` 字面量**的 bash 进程（模拟"调用启动器的 shell"）；
  2. 调 `list_other_runs(['al_launch'+'.py'])`，期望**不匹配**（它只是祖先进程/包装 shell）；
  3. 再把一个**真的在跑 `al_launch.py`** 的 python 进程拉起来，期望**匹配**（护栏没被削弱）。

结果写进 <临时目录>/result.json。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = 'al_launch' + '.py'          # 拆开写：本文件自身命令行里不出现该字面量
tmp = Path(sys.argv[1])
tmp.mkdir(parents=True, exist_ok=True)


def load_launcher():
    spec = importlib.util.spec_from_file_location(
        'al_launch_probe', ROOT / 'experiment' / 'robotwin' / ('al_launch' + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load_launcher()
result = {'skip_pids': sorted(mod._self_skip_pids()) if hasattr(mod, '_self_skip_pids') else None}

# ---- 1) 包装 shell：命令行里含 al_launch.py（模拟 `bash -c "... al_launch.py ..."`）----
wrapper = subprocess.Popen(
    ['bash', '-c', f'echo running {TOKEN} ; sleep 30'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.0)
hits1 = mod.list_other_runs([TOKEN])
result['wrapper_pid'] = wrapper.pid
result['matched_wrapper'] = [h for h in hits1 if h['pid'] == wrapper.pid]
# 也检查直接祖先（本进程的父进程）没被匹配
ppid = mod.proc_ppid(__import__('os').getpid())
result['parent_pid'] = ppid
result['matched_parent'] = [h for h in hits1 if h['pid'] == ppid]

# ---- 2) 真在跑启动器的进程：期望被抓到 ----
real = subprocess.Popen([sys.executable, '-c',
                         f'import time; time.sleep(30)  # {TOKEN}'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.0)
hits2 = mod.list_other_runs([TOKEN])
result['real_pid'] = real.pid
result['matched_real'] = [h for h in hits2 if h['pid'] == real.pid]

# 清理：只按自己记录的 PID 精确终止
for p in (wrapper, real):
    try:
        p.terminate()
        p.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass

(tmp / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
