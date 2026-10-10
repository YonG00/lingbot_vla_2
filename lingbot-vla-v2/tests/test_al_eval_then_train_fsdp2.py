"""评测体张量上下文回归：**必须 no_grad，绝不能 inference_mode**（2026-10-10 真机 hang）。

现象（AMD 8×W7900D / ROCm / 2 卡 FSDP2，AutoLearning bootstrap）
--------------------------------------------------------------
`_select()` 的 train/val 两次开环评测都正常出数，随后**完全静默**：
进程存活、显存不释放、`Step` 停在 0。而那次评测之后紧接着跑的就是
`HardnessScanner.scan()`（`orchestration/scheduler.py:644`）—— 它跑在**任何训练前向之前**。

根因（CPU 2 进程 gloo + 真 FSDP2 逐字复现，见 `tools/al_bootstrap_order_fsdp2_repro.py`）
--------------------------------------------------------------------------------------
* 评测体包在 `torch.inference_mode()` 里（旧代码 `open_loop_validation.py` 的
  `_run()` / `_evaluate_run()`）；
* FSDP2 只对**根单元**显式 `unshard()`（在 inference_mode 之外，安全），
  而**嵌套单元**（每层 decoder 各一个 FSDP2）是在**模型前向内部**被各自的
  pre-forward hook all-gather 的 ⇒ 这一步**发生在 inference_mode 之内**；
* `torch.inference_mode()` 里建出来的张量**永久**带 inference 标记 ⇒ 嵌套单元的
  `fsdp_params[i].unsharded_param` 被污染、并被 FSDP2 缓存复用；
* 紧接着的 Hardness 扫描 / 第一个训练步（InferenceMode **之外**）在
  `all_gather_copy_out` 里 `split_with_sizes_copy(..., out=unsharded_param)` 做 inplace 写 ⇒::

      RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.

  （两个 rank 同时踩到时是报错；只有一个 rank 先踩到、另一个还等在集合通信里时就是**静默 hang**。）

守门（本文件）
--------------
1. 单元级：`_eval_tensor_context()` 必须「关 grad + **不**开 inference mode」；
2. 源码级：`OpenLoopValidator._run` / `._evaluate_run` 体内**不得**出现
   `torch.inference_mode()` 调用；
3. 端到端（真 2 进程 gloo + 真 FSDP2）：
   * `--eval-mode repo`（当前实现）⇒ 全链路通过；
   * `--eval-mode inference`（修复前的写法）⇒ **必须**复现上面那条 RuntimeError
     且诊断里点名被污染的 `unsharded_param` ⇒ 证明第 1/2 条不是装饰性的。
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch", reason="本测试需要 torch")

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "lingbotvla/utils/open_loop_validation.py"
TOOL = REPO / "tools/al_bootstrap_order_fsdp2_repro.py"


# --------------------------------------------------------------------------- #
# 1) 单元级：`_eval_tensor_context()` 的语义
# --------------------------------------------------------------------------- #
def _load_eval_tensor_context():
    """AST 编译 `_eval_tensor_context`（本机可能缺 torchdata/lerobot ⇒ 不 import 整模块）。"""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    body = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_eval_tensor_context"]
    assert body, "open_loop_validation.py 里找不到 _eval_tensor_context"
    ast.fix_missing_locations(tree)
    ns = {"torch": torch}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return ns["_eval_tensor_context"]


def test_eval_tensor_context_is_no_grad_and_not_inference_mode():
    fn = _load_eval_tensor_context()
    assert torch.is_grad_enabled() is True, "前置条件：本测试必须从 grad 开启状态开始"
    with fn():
        assert torch.is_grad_enabled() is False, "评测体必须关 grad（no_grad）"
        assert torch.is_inference_mode_enabled() is False, (
            "评测体**不能**开 inference_mode：嵌套 FSDP2 单元的 unsharded_param 会被打成 "
            "inference 张量，紧接着的 Hardness/训练前向就报 "
            "`Inplace update to inference tensor outside InferenceMode`（真机 hang 的根因）")
    # 退出后必须完全恢复
    assert torch.is_grad_enabled() is True
    assert torch.is_inference_mode_enabled() is False


def _inference_mode_calls(node: ast.AST) -> list:
    hits = []
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "inference_mode"):
            hits.append(getattr(sub, "lineno", -1))
    return hits


@pytest.mark.parametrize("method", ["_run", "_evaluate_run"])
def test_eval_bodies_never_call_inference_mode(method):
    """源码级守门：两个评测执行体里不得出现 `torch.inference_mode()`。"""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == method:
                    found = sub
    assert found is not None, f"找不到 OpenLoopValidator.{method}"
    hits = _inference_mode_calls(found)
    assert not hits, (
        f"OpenLoopValidator.{method} 第 {hits} 行又用了 torch.inference_mode() ⇒ "
        "FSDP2 多卡下评测之后的训练/Hardness 前向会报 "
        "`Inplace update to inference tensor outside InferenceMode`（真机表现为静默 hang）")


# --------------------------------------------------------------------------- #
# 2) 端到端：真 2 进程 gloo + 真 FSDP2（评测窗口 → Hardness → 训练步）
# --------------------------------------------------------------------------- #
def _run_repro(eval_mode: str, timeout: float = 300.0):
    return subprocess.run(
        [sys.executable, str(TOOL), "--ws", "2", "--timeout", "180",
         "--eval-mode", eval_mode],
        cwd=str(REPO), capture_output=True, text=True, timeout=timeout)


def test_al_eval_then_train_fsdp2_passes_with_repo_eval_context():
    """当前实现（no_grad 评测体）：评测 → Hardness → 训练步 → 单元末评测全部通过。"""
    if not torch.distributed.is_available():
        pytest.skip("本机 torch.distributed 不可用")
    out = _run_repro("repo")
    tail = "\n".join((out.stdout or "").splitlines()[-14:])
    assert out.returncode == 0, f"rc={out.returncode}\n{tail}\n{out.stderr[-800:]}"
    assert "✅" in tail, tail
    assert "❌" not in tail, tail


def test_al_eval_then_train_fsdp2_repro_detects_inference_mode_bug():
    """修复前的写法（inference_mode 评测体）必须复现那条 RuntimeError（守门非装饰性）。"""
    if not torch.distributed.is_available():
        pytest.skip("本机 torch.distributed 不可用")
    out = _run_repro("inference")
    blob = (out.stdout or "") + (out.stderr or "")
    assert out.returncode != 0, "inference_mode 版本竟然通过了 —— 复现脚本已失效？"
    assert "Inplace update to inference tensor outside InferenceMode" in blob, blob[-1200:]
    assert "unsharded_param" in blob, (
        "诊断里应点名被 inference 标记污染的 FSDP2 张量（unsharded_param）\n" + blob[-1200:])
