"""多卡开环评测的「rank 不对称 ⇒ 集合通信错配 ⇒ NCCL 看门狗 600 s 后 abort」守门测试。

真机故障（2026-10-10，AMD 8×W7900D / ROCm / 8 卡 FSDP2 正式 AL 训练）
--------------------------------------------------------------------
* bootstrap 的**第一个** scout 评测（`al_adjust_bottle_val_e10f709e` = `adjust_bottle`
  前 2 条 val 回合的指纹）里整体崩掉，日志只剩::

      [Rank 1] Some NCCL operations have failed or timed out.
      [PG ID 0 PG GUID 0(default_pg) Rank 1] Process group watchdog thread terminated …
      [open_loop] ⚠️ [al_adjust_bottle_val_e10f709e] 评测失败: DistBackendError: NCCL communicator was aborted on rank 1.

* 判读：卡住的集合通信在 **default_pg**（= 评测末尾的 `broadcast_object_list`；
  FSDP2 的 all-gather 走 device mesh 自己的 process group）⇒ **有人没到那次广播**。
* 两种可复现成因（本文件的守门对象）：
  ① 某 rank 在评测**准备阶段**（建数据集 / 解码 / 磁盘）失败 ⇒ 旧代码在
     `_multirank_eval_payload` 里把它**静默吞掉**（`except BaseException: pass`），
     该 rank 去等末尾广播、其余 rank 卡在 all-gather ⇒ 死锁（真机 = 600 s 看门狗）；
  ② 各 rank 的**推理次数**不同（回合集合 / 索引口径分叉）⇒ 集合通信次数错配 ⇒ 同样死锁。

本文件断言（新代码）
--------------------
1. **预检在所有集合通信之前**：`_eval_prelude` 在**真实 unshard 窗口之外**依次做
   本地准备 → 一致性预检 A → **对称预热** → 一致性预检 B，且 `validate()` /
   `evaluate_ids()` 两条路径都在 `_sharded_eval_context()`（= `unshard()`）**之前**
   调它（源码级守门 —— 真机日志里 `一致性预检通过` 出现 0 次、`unshard()` 出现了，
   说明旧实现的预检在窗口**之内**，护不到真正出事的集合点）；
2. **窗口前对称预热**：所有 rank 都做、输入一致、无副作用（no_grad / RNG 与
   `_noise_gen` 还原 / 不留梯度 / 不进 inference_mode），`AL_EVAL_WARMUP=0` 可关，
   且**预热的「计划」逐 rank 一致**（预热窗口本身是集合通信，一个 rank 进、
   另一个不进就是死锁 ⇒ 不一致必须由预检 A 在任何集合通信之前拦下）；
3. 非 0 rank 在评测体内失败**绝不再静默**：异常 + rank + 栈先打到 stderr；
   `AL_EVAL_FAIL_FAST_NONZERO=1`（且已进集合通信区）时**立刻抛**（不再等广播）；
4. 每 rank 阶段面包屑 + 停滞告警（`AL_EVAL_PHASE_LOG` / `AL_EVAL_STALL_SEC`）：
   走**逐 rank 的 stderr + 独立轨迹文件**（真机 rank4 一行都没有 = 该机制不成立），
   且「▶ 进入评测」一定写在进窗口之前；
5. 端到端（真 2 进程 gloo + 真 FSDP2 + **真 `OpenLoopValidator`**，
   `tools/al_eval_multirank_preflight_repro.py`）：
   * 健康路径全绿、各 rank 集合通信次数一致、各 rank 恰好预热 1 次、无副作用审计通过；
   * 注入「某 rank 准备阶段失败」/「某 rank 预热报告失败」/「某 rank **不进**预热窗口」
     ⇒ **快速一起失败**（不是死锁），且都**在 unshard 之前**被点名；
   * 注入「某 rank 在**窗口之前**慢一拍」（真机 rank4 形态）⇒ 不是死锁，
     其余 rank 停在**窗口前的集合点**、停滞告警指名阶段与耗时；
   * 关掉预检（旧的逐字行为）⇒ **必须**复现死锁（证明守门不是装饰性的）。
"""
from __future__ import annotations

import ast
import contextlib
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "lingbotvla/utils/open_loop_validation.py"
TOOL = REPO / "tools/al_eval_multirank_preflight_repro.py"

_WANTED_FUNCS = (
    "_multirank_eval_payload",
    "_env_flag", "_preflight_enabled", "_fail_fast_nonzero", "_stall_seconds",
    "_stall_poll_seconds", "_warmup_enabled",
    "_eval_rank_report", "_preflight_verdict", "_format_preflight_error",
    "_multirank_eval_preflight", "_EvalPhaseTracker",
)   # ⚠️ `_eval_warmup_plan` / `_eval_warmup` 是 OpenLoopValidator 的**方法** ⇒ 见 `_warmup_class`
_WANTED_CONSTS = ("EVAL_PREFLIGHT_ENV", "EVAL_PHASE_LOG_ENV", "EVAL_STALL_SEC_ENV",
                  "EVAL_FAIL_FAST_ENV", "EVAL_WARMUP_ENV", "EVAL_PHASE_DIR_ENV",
                  "EVAL_STALL_POLL_SEC_ENV")


def _mod() -> SimpleNamespace:
    """AST 编译被测函数（本机 import 不了整模块：缺 torchdata / lerobot 等）。"""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    body: List[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED_FUNCS:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in _WANTED_FUNCS:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", None) in _WANTED_CONSTS for t in targets):
                body.append(node)
    got = {n.name for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert got == set(_WANTED_FUNCS), f"源码里找不到这些定义: {sorted(set(_WANTED_FUNCS) - got)}"
    ast.fix_missing_locations(tree)
    ns: Dict[str, Any] = {
        "Any": Any, "List": list, "Dict": dict, "Callable": object, "Iterator": object,
        "Tuple": tuple, "Optional": Optional, "Sequence": list, "os": os,
        "contextlib": contextlib, "time": time, "sys": sys, "print": print,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return SimpleNamespace(**{k: ns[k] for k in list(_WANTED_FUNCS) + list(_WANTED_CONSTS)})


# --------------------------------------------------------------------------- #
# 1) 预检的纯逻辑
# --------------------------------------------------------------------------- #
def test_preflight_healthy_reports_no_problem():
    """各 rank 的 tag/ids/推理次数/长度全一致 ⇒ 无 failures、无 problems。"""
    m = _mod()
    rpt = m._eval_rank_report(tag="t", ids=[3, 1], rank=0, n_starts=6, len_ds=10)
    assert rpt["ids"] == [1, 3], "ids 必须排序（顺序无关的身份）"
    v = m._preflight_verdict(3, [dict(rpt, rank=r) for r in range(3)])
    assert v["failures"] == [] and v["problems"] == []


@pytest.mark.parametrize("field,value,needle", [
    ("n_starts", 9, "n_starts 不一致"),
    ("len_ds", 15, "len_ds 不一致"),
    ("ids", [1, 5, 20], "ids 不一致"),
    ("tag", "other_eval", "tag 不一致"),
])
def test_preflight_detects_rank_asymmetry(field, value, needle):
    """任一根 rank 的评测身份不同 ⇒ problems 里点名 rank 与字段（真机「各 rank 集合不同」）。"""
    m = _mod()
    reps = [m._eval_rank_report(tag="t", ids=[1, 5], rank=r, n_starts=6, len_ds=10)
            for r in range(3)]
    reps[2] = dict(reps[2], **{field: value})
    v = m._preflight_verdict(3, reps)
    assert v["failures"] == []
    assert any(needle in p and "rank2" in p for p in v["problems"]), v["problems"]
    msg = m._format_preflight_error(v, tag="t")
    assert "一致性预检未通过" in msg and "rank2" in msg
    assert m.EVAL_PREFLIGHT_ENV in msg, "报错里必须写清怎么关掉预检（可回滚）"


def test_preflight_detects_rank_local_failure():
    """某 rank 准备阶段失败 ⇒ failures 带着它的错误文本（并让所有 rank 一起抛）。"""
    m = _mod()
    reps = [m._eval_rank_report(tag="t", ids=[1, 5], rank=r, n_starts=6, len_ds=10)
            for r in range(4)]
    reps[3] = m._eval_rank_report(tag="t", ids=[1, 5], rank=3,
                                  error="RuntimeError: 解码失败")
    v = m._preflight_verdict(4, reps)
    assert len(v["failures"]) == 1 and v["failures"][0]["rank"] == 3
    msg = m._format_preflight_error(v, tag="t")
    assert "rank3" in msg and "解码失败" in msg


def test_preflight_skips_collective_on_single_rank():
    """ws<=1 ⇒ **不做任何集合通信**（单卡行为逐字不变，注入的 all_gather 不许被调用）。"""
    m = _mod()
    called = []
    v = m._multirank_eval_preflight(
        ws=1, rank=0, tag="t",
        local_report=m._eval_rank_report(tag="t", ids=[1], rank=0, n_starts=1, len_ds=5),
        all_gather=lambda out, obj: called.append(obj))
    assert called == [] and v["problems"] == [] and v["failures"] == []


def test_preflight_all_gathers_and_verdicts():
    """ws>1 ⇒ 用注入的 all_gather 交换各 rank 报告；不一致 ⇒ problems。"""
    m = _mod()
    reports = [m._eval_rank_report(tag="t", ids=[1, 5], rank=r, n_starts=6, len_ds=10)
               for r in range(2)]
    reports[1] = dict(reports[1], n_starts=7)

    def _fake_gather(out, obj):
        out[:] = reports

    v = m._multirank_eval_preflight(ws=2, rank=0, tag="t",
                                    local_report=reports[0], all_gather=_fake_gather)
    assert any("n_starts 不一致" in p for p in v["problems"])


# --------------------------------------------------------------------------- #
# 2) 非 0 rank 失败：不再静默 + 可选 fail-fast
# --------------------------------------------------------------------------- #
def _simulate(m, *, ranks, all_ranks_run, run_by_rank, fail_fast=None):
    """忠实模拟 `broadcast_object_list`（rank0 写、其余读 rank0 的）与逐 rank 调用。"""
    box = {"rank0": None}
    out = {}
    for r in sorted(ranks):
        def _run(rr=r):
            return run_by_rank[rr]()

        def _bc(payload, rr=r):
            if rr == 0:
                box["rank0"] = payload[0]
            else:
                payload[0] = box["rank0"]

        try:
            out[r] = m._multirank_eval_payload(run=_run, ws=len(ranks), rank=r,
                                               broadcast=_bc, all_ranks_run=all_ranks_run,
                                               fail_fast=fail_fast)
        except BaseException as exc:  # noqa: BLE001
            out[r] = exc
    return out


def test_nonzero_rank_failure_is_never_silent(capsys):
    """非 0 rank 失败**必须**在 stderr 留证据（旧代码 `except BaseException: pass` 什么都没有）。"""
    m = _mod()

    def _ok():
        return {"mse": 1.0}

    def _boom():
        raise RuntimeError("rank1 的死因-证据")

    out = _simulate(m, ranks=[0, 1], all_ranks_run=True,
                    run_by_rank={0: _ok, 1: _boom})
    err = capsys.readouterr().err
    assert "rank1" in err and "rank1 的死因-证据" in err and "RuntimeError" in err, err
    # 协议语义保持不变：仍用 rank0 的结果（默认不 fail-fast）
    assert out[0] == {"mse": 1.0} and out[1] == {"mse": 1.0}


def test_nonzero_rank_failure_fail_fast_raises_immediately(capsys):
    """`fail_fast(exc)=True`（= 集合通信区内失败）⇒ 该 rank 立刻抛，不再等广播。"""
    m = _mod()

    def _boom():
        raise RuntimeError("集合通信区内失败")

    out = _simulate(m, ranks=[0, 1], all_ranks_run=True,
                    run_by_rank={0: lambda: {"mse": 1.0}, 1: _boom},
                    fail_fast=lambda exc: True)
    assert isinstance(out[1], RuntimeError) and "集合通信区内失败" in str(out[1])
    assert out[0] == {"mse": 1.0}
    assert "rank1" in capsys.readouterr().err, "fail-fast 之前也必须先留证据"


def test_rank0_failure_still_broadcast_to_everyone():
    """rank0 失败 ⇒ 广播错误标记、所有 rank 一起抛（原有语义不变）。"""
    m = _mod()

    def _boom():
        raise ValueError("rank0 挂了")

    out = _simulate(m, ranks=[0, 1], all_ranks_run=True,
                    run_by_rank={0: _boom, 1: lambda: {"mse": 9.0}})
    assert all(isinstance(v, RuntimeError) and "rank0 挂了" in str(v) for v in out.values()), out


# --------------------------------------------------------------------------- #
# 3) 源码级守门：预检必须在集合通信段之前
# --------------------------------------------------------------------------- #
def _method_node(name: str) -> ast.FunctionDef:
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "OpenLoopValidator":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == name:
                    return sub
    raise AssertionError(f"找不到 OpenLoopValidator.{name}")


def _call_lines(fn: ast.FunctionDef, names) -> Dict[str, int]:
    """函数体里第一次出现 `self.<name>(...)` / `<name>(...)` 调用的行号。"""
    out: Dict[str, int] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name in names:
            out.setdefault(name, node.lineno)
    return out


def test_eval_prelude_orders_preflight_before_warmup_before_window():
    """`_eval_prelude` 的顺序：本地准备 → 预检 → 预热 → 再预检（都在窗口之外）。"""
    fn = _method_node("_eval_prelude")
    lines = _call_lines(fn, ("_eval_prepare", "_eval_preflight", "_eval_warmup",
                             "_phase_tracker", "_sharded_eval_context"))
    assert "_eval_prepare" in lines, "`_eval_prelude` 里找不到 `_eval_prepare(...)`"
    assert lines["_eval_prepare"] < lines["_eval_preflight"], lines
    assert lines["_eval_preflight"] < lines["_eval_warmup"], (
        "预检必须在**预热之前**：预热窗口本身是集合通信，"
        "「谁进谁不进」必须由预检先确认一致（否则死锁）", lines)
    preflights = [n.lineno for n in ast.walk(fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "_eval_preflight"]
    assert len(preflights) >= 2 and max(preflights) > lines["_eval_warmup"], preflights
    assert "_sharded_eval_context" not in lines, (
        "`_eval_prelude` 自己**不**开评测窗口：真实窗口由入口方法在预检/预热之后开")


@pytest.mark.parametrize("entry", ["validate", "evaluate_ids"])
def test_entry_paths_preflight_and_warmup_before_unshard_window(entry):
    """两条入口都必须在 `_sharded_eval_context()`（= `unshard()` 集合通信）**之前**
    调 `_eval_prelude`（预检 + 预热）。

    真机证据：日志里 `一致性预检通过` 出现 0 次、`评测窗口：unshard()` 出现了
    ⇒ 旧实现的预检在窗口**里面**，`unshard()` 这个真正出事的集合点根本没被护到。
    """
    fn = _method_node(entry)
    lines = _call_lines(fn, ("_eval_prelude", "_sharded_eval_context",
                             "_multirank_eval_payload"))
    assert "_eval_prelude" in lines, f"`{entry}` 里找不到 `_eval_prelude(...)`"
    assert "_sharded_eval_context" in lines, f"`{entry}` 里应保留评测窗口"
    assert lines["_eval_prelude"] < lines["_sharded_eval_context"], (
        f"预检/预热必须在 unshard 窗口之前：{lines}")
    assert lines["_eval_prelude"] < lines["_multirank_eval_payload"], lines


def test_validate_preludes_both_eval_sets():
    """`validate()` 的 train_monitor 与 val **各自**都要走窗口前阶段。"""
    src = ast.get_source_segment(SRC.read_text(encoding="utf-8"),
                                 _method_node("validate")) or ""
    assert "train_monitor" in src and '"val"' in src, src[-800:]
    assert "_eval_prelude" in src and "preps" in src


def test_evaluate_ids_reuses_prep_and_raises_collected_error():
    """`_evaluate_ids` 复用窗口外算好的 prep，并且**不**自己开窗口。"""
    fn = _method_node("_evaluate_ids")
    lines = _call_lines(fn, ("_eval_prepare", "_eval_preflight", "_prediction_groups",
                             "_sharded_eval_context"))
    src = ast.get_source_segment(SRC.read_text(encoding="utf-8"), fn) or ""
    assert "_sharded_eval_context" not in lines, (
        "`_evaluate_ids` 体内不应再有 FSDP 窗口（窗口在入口方法里、预检/预热之后开）")
    assert "_prediction_groups" in lines and "prep_error" in src
    assert lines["_eval_prepare"] < lines["_prediction_groups"], lines
    if "_eval_preflight" in lines:      # 只有「直接调用」的兜底路径才会补预检
        assert lines["_eval_preflight"] < lines["_prediction_groups"], lines


def test_prepare_phase_error_is_collected_not_raised_inline():
    """准备阶段的异常必须**先收进 prep 字典**（否则失败 rank 会脱离集合点 ⇒ 死锁）。"""
    src = ast.get_source_segment(SRC.read_text(encoding="utf-8"),
                                 _method_node("_eval_prepare")) or ""
    assert 'prep["prep_error"] = exc' in src, "准备阶段异常必须收集进 prep_error（不能就地抛）"
    src_eval = ast.get_source_segment(SRC.read_text(encoding="utf-8"),
                                      _method_node("_evaluate_ids")) or ""
    assert "raise prep_error" in src_eval, "收集到的准备错误要在推理段原样抛（旧语义）"


def test_warmup_is_side_effect_free_and_plan_guarded():
    """预热：no_grad（**不是** inference_mode）、安全上下文、噪声 generator 还原、
    由 `_eval_warmup_plan` 决定进不进窗口（对称性由预检保证）。"""
    src = ast.get_source_segment(SRC.read_text(encoding="utf-8"),
                                 _method_node("_eval_warmup")) or ""
    assert "_eval_tensor_context()" in src and "torch.inference_mode()" not in src, (
        "预热必须走 `_eval_tensor_context()`（no_grad）；inference 张量会污染 FSDP2 参数")
    assert "_eval_context()" in src, "预热必须包在 safe_eval_context 里（RNG/标志快照+恢复+审计）"
    assert "_noise_gen" in src, "预热不得改变评测噪声序列起点（_noise_gen 还原）"
    assert "_eval_warmup_plan" in src, "预热必须先问 `_eval_warmup_plan`（逐 rank 对称）"
    plan = ast.get_source_segment(SRC.read_text(encoding="utf-8"),
                                  _method_node("_eval_warmup_plan")) or ""
    assert "_warmup_enabled()" in plan and "_warmup_done" in plan and "prep_error" in plan
    assert "warmup_item" in plan, "预热要用的那条样本必须在**窗口前**就读好（本地失败不脱队）"


def test_warmup_plan_is_part_of_preflight_report():
    """预热计划必须随预检交换出去（否则「一 rank 进窗口、另一 rank 不进」= 死锁）。"""
    src = ast.get_source_segment(SRC.read_text(encoding="utf-8"),
                                 _method_node("_eval_prelude")) or ""
    assert "warmup=warm_plan" in src, "预检报告里必须带 warmup 计划"
    verdict_src = SRC.read_text(encoding="utf-8")
    assert '"warmup"' in verdict_src, "_preflight_verdict 必须把 warmup 纳入一致性比对"


def test_collective_region_flag_and_fail_fast_env():
    """`_nonzero_failure_is_fatal` 必须同时要求「env 开关」与「已进集合通信区」。"""
    fn = _method_node("_nonzero_failure_is_fatal")
    src = ast.get_source_segment(SRC.read_text(encoding="utf-8"), fn) or ""
    assert "_fail_fast_nonzero()" in src and "_eval_in_collective_region" in src


# --------------------------------------------------------------------------- #
# 4) 阶段面包屑 + 停滞告警
# --------------------------------------------------------------------------- #
class _CapLogger:
    """（保留）旧接口的 logger —— 新实现**默认不经过它**（见下面的测试）。"""

    def __init__(self):
        self.lines: List[str] = []

    def info(self, msg, *a, **k):
        self.lines.append(f"INFO {msg}")

    def warning(self, msg, *a, **k):
        self.lines.append(f"WARN {msg}")

    def info_rank0(self, msg, *a, **k):
        self.lines.append(f"INFO0 {msg}")


def test_phase_tracker_logs_breadcrumbs_and_stall_warning(capsys, tmp_path):
    """阶段面包屑走 **stderr + 逐 rank 轨迹文件**（真机 rank4 一行都没有 = 该机制不成立）。

    ⚠️ 关键回归点：**不再依赖 logger**（logger 级别、`LOCAL_RANK` 门控、
    stdout 重定向都可能吞掉某个 rank 的输出）。所以这里给一个「什么都不收」的
    logger，输出仍然必须出现在 stderr 与 rank3.log 里。
    """
    m = _mod()
    tr = m._EvalPhaseTracker(None, rank=3, ws=8, stall_sec=0.15, poll_sec=0.05,
                             trace_dir=str(tmp_path))
    tr.begin("al_adjust_bottle_val_e10f709e")
    tr.phase("本地准备（数据集/索引）")
    time.sleep(0.45)                     # 让停滞线程抓到一次
    tr.phase("推理进度 1/6")
    tr.end(ok=True)
    err = capsys.readouterr().err
    trace = (tmp_path / "rank3.log").read_text(encoding="utf-8")
    for blob in (err, trace):
        assert "rank3/8" in blob and "进入评测" in blob and "推理进度 1/6" in blob
        assert "仍停在" in blob, f"停滞告警没打出来：{blob}"
        assert "✅ 评测结束" in blob
    # 「▶ 进入评测」必须在**进入任何阶段之前**就已落盘（真机 rank4 的判据）
    lines = trace.splitlines()
    assert lines[0].startswith("[open_loop][phase][rank3/8] ▶ 进入评测"), lines[0]
    assert lines.index([ln for ln in lines if "▶ 本地准备" in ln][0]) > 0


def test_phase_tracker_writes_one_file_per_rank(capsys, tmp_path):
    """逐 rank **各写各的文件**：8 个 rank 的结果不会互相覆盖，谁缺一眼可见。"""
    m = _mod()
    for r in range(8):
        tr = m._EvalPhaseTracker(None, rank=r, ws=8, trace_dir=str(tmp_path))
        tr.begin(f"t{r}")
        tr.end()
    files = sorted(p.name for p in tmp_path.glob("rank*.log"))
    assert files == [f"rank{r}.log" for r in range(8)], files
    assert all((tmp_path / f).read_text(encoding="utf-8").strip() for f in files)


def test_phase_tracker_disabled_on_single_rank(capsys, tmp_path):
    """单卡 ⇒ 一条阶段日志都不打、也不建轨迹文件（单卡日志与以前逐字一致）。"""
    m = _mod()
    log = _CapLogger()
    tr = m._EvalPhaseTracker(log, rank=0, ws=1, trace_dir=str(tmp_path))
    tr.begin("t")
    tr.phase("x")
    tr.end()
    assert log.lines == [] and capsys.readouterr().err == ""
    assert list(tmp_path.glob("rank*.log")) == []


def test_stall_poll_derived_from_stall_sec(monkeypatch):
    """告警阈值调小时轮询也要跟着变小（否则排障时又要多等 15 s）。"""
    m = _mod()
    monkeypatch.delenv(m.EVAL_STALL_POLL_SEC_ENV, raising=False)
    assert m._stall_poll_seconds(120.0) == 15.0
    assert m._stall_poll_seconds(4.0) <= 2.0
    monkeypatch.setenv(m.EVAL_STALL_POLL_SEC_ENV, "0.25")


# --------------------------------------------------------------------------- #
# 4b) 窗口前对称预热：计划判定（纯逻辑，不需要分布式）
# --------------------------------------------------------------------------- #
def _warmup_class(m):
    """把 `_eval_warmup_plan` + `_eval_warmup` 装进一个最小类（只测这两个方法）。"""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    body = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("_env_flag", "_warmup_enabled")]
    for name in ("_eval_warmup_plan", "_eval_warmup"):
        body.append(_method_node(name))              # 这两个是 OpenLoopValidator 的方法
    mod = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: Dict[str, Any] = {"Any": Any, "List": list, "Dict": dict, "Optional": Optional,
                          "Sequence": list, "os": os, "time": time,
                          "EVAL_WARMUP_ENV": m.EVAL_WARMUP_ENV,
                          # 三个模块级小工具的桩（本测试只测预热逻辑本身）
                          "_global_rank": lambda: 2, "_world_size": lambda: 8,
                          "traceback": traceback,
                          "_eval_tensor_context": lambda: contextlib.nullcontext()}
    exec(compile(mod, str(SRC), "exec"), ns)
    # 基类提供 `_eval_context` / `_sharded_eval_context` / `_infer_one` / logger 等假件
    return type("_W", (_FakeWarmSelf,), {k: ns[k] for k in ("_eval_warmup_plan", "_eval_warmup")})


class _FakeWarmSelf:
    """最小 self：记录调用顺序，供断言「预热到底做了什么」。

    ⚠️ `_eval_warmup_plan` / `_eval_warmup` 由 `_warmup_class()` 从**真实源码**里
    取出来装到这个类上（保证测的是真代码，不是复制品）。
    """

    def __init__(self, tmp_trace=None):
        self.model = object()
        self._warmup_done = False
        self._noise_gen = "GEN"
        self.calls: List[str] = []
        self.logger = _CapLogger()

    # 三个上下文/推理入口都记一笔
    @contextlib.contextmanager
    def _eval_context(self):
        self.calls.append("eval_context")
        yield None

    @contextlib.contextmanager
    def _sharded_eval_context(self):
        self.calls.append("sharded_window")
        yield ""

    def _infer_one(self, item, ft):
        self.calls.append("infer_one")
        return {"mse": 0.0}


def _good_prep():
    return {"prep_error": None, "ds": [1], "ft": object(), "warmup_item": {"a": 1},
            "starts": [0, 5]}


def test_warmup_plan_requires_everything_local_and_consistent(monkeypatch):
    """计划的判定条件：开关开、本进程没做过、有模型、prep 完好、样本已读好。"""
    m = _mod()
    cls = _warmup_class(m)
    s = cls()
    monkeypatch.delenv(m.EVAL_WARMUP_ENV, raising=False)
    assert s._eval_warmup_plan(_good_prep()) is True
    monkeypatch.setenv(m.EVAL_WARMUP_ENV, "0")
    assert s._eval_warmup_plan(_good_prep()) is False, "AL_EVAL_WARMUP=0 必须能关掉预热"
    monkeypatch.delenv(m.EVAL_WARMUP_ENV, raising=False)
    s._warmup_done = True
    assert s._eval_warmup_plan(_good_prep()) is False, "一次/进程"
    s._warmup_done = False
    for bad in ({"prep_error": RuntimeError("x"), "ds": 1, "ft": 1, "warmup_item": 1,
                 "starts": [0]},
                dict(_good_prep(), warmup_item=None),
                dict(_good_prep(), starts=[])):
        assert s._eval_warmup_plan(bad) is False, bad
    s.model = None
    assert s._eval_warmup_plan(_good_prep()) is False


def test_warmup_skips_entirely_when_plan_is_false(monkeypatch):
    """计划为 False ⇒ **一步都不做**（尤其不进那个集合通信窗口）。"""
    m = _mod()
    cls = _warmup_class(m)
    s = cls()
    monkeypatch.setenv(m.EVAL_WARMUP_ENV, "0")
    assert s._eval_warmup(_good_prep(), "t", None) is None
    assert s.calls == [] and s._warmup_done is False, s.calls


def test_warmup_runs_one_inference_inside_safe_contexts_and_restores_noise(monkeypatch, capsys):
    """计划为 True ⇒ 恰好 1 次推理；顺序 = 安全上下文 → no_grad → 全参数窗口；
    `_noise_gen` 还原、`_warmup_done` 置位、返回 None、并逐 rank 打一行含耗时。"""
    m = _mod()
    cls = _warmup_class(m)
    s = cls()
    monkeypatch.delenv(m.EVAL_WARMUP_ENV, raising=False)
    tr = m._EvalPhaseTracker(None, rank=2, ws=8, trace_dir=None)
    tr.begin("t")
    assert s._eval_warmup(_good_prep(), "t", tr) is None
    assert s.calls == ["eval_context", "sharded_window", "infer_one"], s.calls
    assert s._warmup_done is True and s._noise_gen == "GEN"
    # 逐 rank 一行日志（含 rank 与耗时）—— 走 stderr（不看 logger 配置）
    err = capsys.readouterr().err
    assert any("预热完成" in ln and "rank2/8" in ln and "+" in ln
               for ln in err.splitlines()), err


def test_warmup_failure_is_reported_not_raised(monkeypatch):
    """预热失败 ⇒ 返回错误文本（交给集合点 B 一起抛），**不**在窗口里就地抛。"""
    m = _mod()
    cls = _warmup_class(m)
    s = cls()

    def _boom(item, ft):
        raise RuntimeError("INJECTED warmup boom")

    s._infer_one = _boom
    monkeypatch.delenv(m.EVAL_WARMUP_ENV, raising=False)
    err = s._eval_warmup(_good_prep(), "t", None)
    assert err and "INJECTED warmup boom" in err, err
    assert s._noise_gen == "GEN"


# --------------------------------------------------------------------------- #
# 5) 端到端：真 2 进程 gloo + 真 FSDP2 + 真 OpenLoopValidator（含故障注入）
# --------------------------------------------------------------------------- #
pytest.importorskip("torch", reason="本测试需要 torch")


def _run_tool(*args: str, timeout: float = 300.0):
    return subprocess.run([sys.executable, str(TOOL), *args],
                          cwd=str(REPO), capture_output=True, text=True, timeout=timeout)


def test_tool_healthy_multirank_eval_no_deadlock():
    """健康路径（2 进程 gloo + 真 FSDP2）：评测 → 训练步 → 再评测，全绿、集合通信次数一致。"""
    out = _run_tool("--ws", "2", "--timeout", "90")
    blob = (out.stdout or "") + (out.stderr or "")
    assert out.returncode == 0, blob[-2000:]
    assert "✅ 全链路无死锁" in blob, blob[-2000:]
    m = re.search(r"集合通信调用数（逐 rank）：\{([^}]*)\}", blob)
    assert m, blob[-1500:]
    nums = [int(x) for x in re.findall(r":\s*(\d+)", m.group(1))]
    assert len(nums) == 2, f"应有两个 rank 的计数：{m.group(0)}"
    assert len(set(nums)) == 1, f"两个 rank 的集合通信次数必须相同：{m.group(0)}"


def test_tool_rank_local_failure_fails_fast_instead_of_hanging():
    """注入「rank1 准备阶段失败 + 预检开」⇒ 两个 rank **一起**快速失败，绝不死锁。"""
    out = _run_tool("--ws", "2", "--timeout", "45", "--inject", "1:prepare")
    blob = (out.stdout or "") + (out.stderr or "")
    assert out.returncode == 0, blob[-3000:]
    assert "一致性预检未通过" in blob and "INJECTED prepare failure on rank1" in blob, blob[-3000:]
    assert "符合预期" in blob and "疑似**死锁**" not in blob, blob[-1500:]


def test_tool_skew_detected_by_preflight():
    """注入「rank1 的评测回合集合多一条」⇒ 预检点名 n_starts 不一致（而不是死锁）。"""
    out = _run_tool("--ws", "2", "--timeout", "45", "--inject", "1:skew")
    blob = (out.stdout or "") + (out.stderr or "")
    assert out.returncode == 0, blob[-3000:]
    assert "n_starts 不一致" in blob and "符合预期" in blob, blob[-3000:]


def test_tool_preflight_off_reproduces_deadlock():
    """关掉预检（旧的逐字行为）⇒ **必须**复现死锁 —— 证明守门不是装饰性的。"""
    out = _run_tool("--ws", "2", "--timeout", "12", "--inject", "1:prepare",
                    "--preflight", "off")
    blob = (out.stdout or "") + (out.stderr or "")
    assert out.returncode == 2, blob[-3000:]
    assert "疑似**死锁**" in blob, blob[-2000:]


# --------------------------------------------------------------------------- #
# 6) 端到端：窗口前预检 / 预热 / 逐 rank 阶段日志（真 gloo + 真 FSDP2）
# --------------------------------------------------------------------------- #
def test_tool_healthy_asserts_warmup_symmetry_side_effects_and_per_rank_trace():
    """健康路径：**每个 rank** 都有阶段日志（且『进入评测』写在窗口之前）、
    各 rank 恰好预热 1 次、且预热/评测不产生任何副作用（RNG/参数/梯度/标志/inference 张量）。"""
    out = _run_tool("--ws", "2", "--timeout", "90")
    blob = (out.stdout or "") + (out.stderr or "")
    assert out.returncode == 0, blob[-3000:]
    assert "✅ 逐 rank 阶段日志" in blob, blob[-3000:]
    assert "✅ 窗口前对称预热" in blob, blob[-2000:]
    assert "✅ 无副作用审计通过" in blob, blob[-2000:]
    assert "预热完成" in blob and "'warmup': True" in blob, blob[-2500:]
    # 预热的日志里必须带 rank 与耗时
    assert re.search(r"预热完成（rank\d/2，1 次真实推理，\+\d", blob), blob[-2500:]


def test_tool_warmup_can_be_turned_off():
    """`AL_EVAL_WARMUP=0`（--warmup off）⇒ 不预热、流程逐字回旧（仍然全绿、无死锁）。"""
    out = _run_tool("--ws", "2", "--timeout", "90", "--warmup", "off")
    blob = (out.stdout or "") + (out.stderr or "")
    assert out.returncode == 0, blob[-3000:]
    assert "窗口前预热调用（计划=False）" in blob, blob[-2000:]
    assert "✅ 全链路无死锁" in blob and "❌" not in blob, blob[-2000:]


def test_tool_warmup_local_failure_named_before_window():
    """注入「rank1 预热报告本地失败」⇒ 由**预检 B**（真实 unshard 窗口之前）点名，
    两个 rank 一起快速失败、**不是**死锁（旧顺序里这种失败会把全组拖到看门狗超时）。"""
    out = _run_tool("--ws", "2", "--timeout", "45", "--inject", "1:warmup")
    blob = (out.stdout or "") + (out.stderr or "")
    assert out.returncode == 0, blob[-3000:]
    assert "INJECTED warmup failure on rank1" in blob, blob[-3000:]
    assert "一致性预检未通过" in blob and "预热后（真实 unshard 窗口之前）" in blob, blob[-3000:]
    assert "符合预期" in blob and "疑似**死锁**" not in blob, blob[-1500:]


def test_tool_warmup_plan_asymmetry_caught_by_preflight_a():
    """注入「rank1 **不进**预热窗口」⇒ 预检 A 的 `warmup` 字段点名。

    这是本设计里最容易踩的死锁：预热窗口是集合通信，一个 rank 进、另一个不进
    必然挂死 ⇒ 必须在**任何集合通信之前**（预检 A）就拦下。
    """
    out = _run_tool("--ws", "2", "--timeout", "45", "--inject", "1:warmup_skip")
    blob = (out.stdout or "") + (out.stderr or "")
    assert out.returncode == 0, blob[-3000:]
    assert "warmup 不一致" in blob and "rank1" in blob, blob[-3000:]
    assert "符合预期" in blob and "疑似**死锁**" not in blob, blob[-1500:]


def test_tool_prewindow_delay_is_not_a_deadlock_and_is_named():
    """注入「某 rank 在**窗口之前**慢一拍」（= 真机 rank4 形态）⇒ 不是死锁：
    其余 rank 停在**窗口前的集合点**，逐 rank 阶段日志 + 停滞告警把位置与耗时写清楚。"""
    out = _run_tool("--ws", "2", "--timeout", "60", "--inject", "1:delay",
                    "--delay-sec", "7", "--stall-sec", "3")
    blob = (out.stdout or "") + (out.stderr or "")
    assert out.returncode == 0, blob[-3000:]
    assert "符合预期（窗口前延迟）" in blob, blob[-3000:]
    assert "仍停在「一致性预检 A" in blob, blob[-3000:]
    assert "疑似**死锁**" not in blob, blob[-1500:]
