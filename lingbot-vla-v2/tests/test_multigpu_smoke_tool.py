"""把多进程体检工具纳入测试套件（真起 2 进程 + gloo，抓"真死锁/真广播错"）。

工具本身在 `tools/multigpu_eval_smoke.py`；这里只做"能不能跑通 + 是否返回 0"的守门。
分布式代码的坑（pickle 不了、rank0 失败导致对端死等、事件流每 rank 各写一份）用 mock 测不出来，
必须真跑进程；本测试超时即判失败。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools/multigpu_eval_smoke.py"


@pytest.mark.parametrize("procs", [2, 4])
def test_multigpu_smoke_tool_passes(procs):
    out = subprocess.run([sys.executable, str(TOOL), "--procs", str(procs),
                          "--timeout", "180"],
                         cwd=str(REPO), capture_output=True, text=True, timeout=300)
    tail = "\n".join((out.stdout or "").splitlines()[-12:])
    assert out.returncode in (0, 3), f"rc={out.returncode}\n{tail}\n{out.stderr[-500:]}"
    if out.returncode == 3:                      # 本机无 torch.distributed ⇒ 跳过
        pytest.skip("本机 torch.distributed 不可用")
    assert "✅" in tail, tail
    # ⚠️ 不能用 `"死锁" not in tail`：成功信息里含"失败不死锁"字样；
    #    死锁的判据是工具打印的"疑似"与失败标记 ❌。
    assert "疑似" not in tail and "❌" not in tail, tail
