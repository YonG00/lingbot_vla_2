#!/usr/bin/env python3
"""多卡评测协议的**多进程体检**（CPU + gloo，**不需要 GPU**）。

为什么不能只做单元测试：分布式代码的坑大多是"**真跑起来才出现**"的 —— 广播签名错、
pickle 不了的结果、rank0 失败导致对端**永久死等**、事件流被每个 rank 各写一份……
这些用假函数 mock 是测不出来的。本脚本用 2 个真实进程 + gloo + `broadcast_object_list`
把协议整体跑一遍，并带**超时判定**（超时即判"疑似死锁"，非零退出）。

用法：
    python tools/multigpu_eval_smoke.py            # 2 进程
    python tools/multigpu_eval_smoke.py --procs 4  # 3 卡以上同协议
退出码：0 全过 / 2 失败（含超时/死锁）/ 3 部分跳过（无 torch.distributed）
"""
from __future__ import annotations

import argparse
import ast
import json
import contextlib
import os
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402


def _helpers() -> Dict[str, Any]:
    """AST 编译 `open_loop_validation` 的多卡工具（该模块本地无法 import：transformers 版本差异）。"""
    src = REPO / "lingbotvla/utils/open_loop_validation.py"
    wanted = ("_world_size", "_global_rank", "_data_parallel_mode", "_ddp_replicated",
              "_is_sharded_mode", "_fsdp_eval_unsupported", "_multirank_eval_supported",
              "_all_ranks_must_run_eval", "_multirank_batch_allowed",
              "_fsdp1_class", "_fsdp2_mixin", "_sharded_model_kind",
              # `_fsdp_full_params_context` 依赖这两个（FSDP2 评测前必须先做根单元惰性初始化）
              "_fsdp2_state", "_fsdp2_root_lazy_init",
              "_fsdp_full_params_context",
              "_broadcast_object", "_multirank_eval_payload")
    consts = ("SHARDED_EVAL_MODES", "MULTIRANK_EVAL_MODES",
              "FSDP_EVAL_UNSUPPORTED_ENV", "MULTIRANK_BATCH_ENV")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", None) in consts for t in targets):
                body.append(node)
    got = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    assert got == set(wanted), sorted(set(wanted) - got)
    ast.fix_missing_locations(tree)
    ns: Dict[str, Any] = {"Any": object, "List": list, "Dict": dict, "Callable": object,
                          "Iterator": object, "Tuple": tuple, "Optional": object,
                          "os": os, "contextlib": contextlib}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(src), "exec"), ns)
    return {k: ns[k] for k in wanted}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


#: 广播用的"结果"要贴近真实：嵌套 dict / list / float / None / 中文
_FAKE_RESULT = {"mse": 0.065688, "mae": 0.1, "nmse": None,
                "per_traj_mse": [0.01, 0.02, 0.03],
                "per_traj_ids": [51, 53, 60],
                "note": "中文与 emoji ✅", "nested": {"a": [1, 2], "b": {"c": 3}}}


def _worker(rank: int, world_size: int, port: int, tmp: str, fail_first: bool,
            out: Any) -> None:
    """一个 rank 的全部检查；结果写进共享 dict（不抛异常，便于父进程统一判定）。"""
    res: Dict[str, Any] = {}
    tmpdir = Path(tmp)
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world_size,
                                init_method=f"tcp://127.0.0.1:{port}")
        H = _helpers()
        res["rank"] = rank
        res["global_rank"] = int(H["_global_rank"]())
        res["world_size"] = int(H["_world_size"]())      # ⚠️ 存**结果**，不能存函数对象（不可 pickle）

        # ---- ① 正常路径：rank0 算 + 广播，所有 rank 必须拿到**完全相同**的结果 ----
        calls = {"run": 0}

        def _run() -> Dict[str, Any]:
            calls["run"] += 1
            return json.loads(json.dumps(_FAKE_RESULT))

        got = H["_multirank_eval_payload"](run=_run, ws=world_size, rank=rank,
                                           broadcast=H["_broadcast_object"])
        res["normal_result_ok"] = (got == _FAKE_RESULT)
        res["normal_local_calls"] = calls["run"]           # 非 0 rank 应为 0

        # ---- ①b 分片并行：all_ranks_run=True ⇒ **每个 rank 都跑**，但只用 rank0 的结果 ----
        #  （FSDP 的 all-gather 是集合通信：任何一个 rank 不跑就挂死；而各 rank 自己的
        #    结果是各自视角的 ⇒ 必须统一用 rank0 的，否则决策分叉。）
        calls2 = {"run": 0}

        def _run2() -> Dict[str, Any]:
            calls2["run"] += 1
            return dict(_FAKE_RESULT, who=rank)

        got2 = H["_multirank_eval_payload"](run=_run2, ws=world_size, rank=rank,
                                            broadcast=H["_broadcast_object"],
                                            all_ranks_run=True)
        res["allranks_local_calls"] = calls2["run"]        # 每个 rank 都应为 1
        res["allranks_result_is_rank0"] = (got2.get("who") == 0)
        res["allranks_result_ok"] = (got2 == dict(_FAKE_RESULT, who=0))

        # 非 0 rank 自己失败 ⇒ 不得影响协议（用 rank0 的结果）
        def _run2_nonzero_boom() -> Dict[str, Any]:
            if rank != 0:
                raise RuntimeError("synthetic-nonzero-local-failure")
            return {"who": 0}

        got3 = H["_multirank_eval_payload"](run=_run2_nonzero_boom, ws=world_size,
                                            rank=rank, broadcast=H["_broadcast_object"],
                                            all_ranks_run=True)
        res["allranks_nonzero_failure_tolerated"] = (got3 == {"who": 0})

        # rank0 失败 ⇒ **所有 rank 一起抛**（带原因），不能有人死等
        def _run2_rank0_boom() -> Dict[str, Any]:
            if rank == 0:
                raise RuntimeError("synthetic-rank0-failure-in-allranks")
            return {"who": rank}

        try:
            H["_multirank_eval_payload"](run=_run2_rank0_boom, ws=world_size, rank=rank,
                                         broadcast=H["_broadcast_object"], all_ranks_run=True)
            res["allranks_rank0_failure_raised"] = False
            res["allranks_rank0_failure_msg"] = ""
        except Exception as exc:  # noqa: BLE001
            res["allranks_rank0_failure_raised"] = True
            res["allranks_rank0_failure_msg"] = f"{type(exc).__name__}: {exc}"

        # 失败路径之后通信组仍然可用
        again2 = H["_multirank_eval_payload"](run=lambda: {"ok": 2}, ws=world_size,
                                              rank=rank, broadcast=H["_broadcast_object"],
                                              all_ranks_run=True)
        res["allranks_group_reusable"] = (again2 == {"ok": 2})

        # ---- ② 失败路径：rank0 抛 ⇒ 所有 rank 一起抛（**不能有人死等**）----
        def _boom() -> Dict[str, Any]:
            raise RuntimeError("synthetic-rank0-failure")

        raised = None
        try:
            H["_multirank_eval_payload"](run=_boom, ws=world_size, rank=rank,
                                         broadcast=H["_broadcast_object"])
        except Exception as exc:  # noqa: BLE001
            raised = f"{type(exc).__name__}: {exc}"
        res["fail_raised"] = raised is not None
        res["fail_is_runtime_error"] = bool(raised and raised.startswith("RuntimeError"))
        res["fail_mentions_cause"] = bool(raised and "synthetic-rank0-failure" in raised)

        # ---- ③ 失败之后通信组仍可用（协议没把组搞坏）----
        again = H["_multirank_eval_payload"](run=lambda: {"ok": 1}, ws=world_size,
                                             rank=rank, broadcast=H["_broadcast_object"])
        res["group_reusable"] = (again == {"ok": 1})

        # ---- ④ 事件流：只有 rank0 写（本 rank 自己写自己的）----
        from lingbotvla.auto_learning.real.build import SchedulerLoggerAdapter

        class _L:
            def info_rank0(self, *a, **k):
                pass
            info = warning = info_rank0

        ev = tmpdir / "events.jsonl"
        log = SchedulerLoggerAdapter(_L(), writer=None, event_path=str(ev))
        log.log_event({"kind": "event", "action": "select", "task": f"t{rank}"})
        log.log_text(1, "text/x", f"rank{rank}")

        # ---- ⑤ scout 缓存：只有 rank0 写 ----
        from lingbotvla.auto_learning.scout_cache import BootstrapScoutCache

        cache = BootstrapScoutCache(tmpdir / "scout.json", model="smoke")
        cache.store("click_bell", [1, 2],
                    {"task": "click_bell", "episode_ids": [1, 2], "mse": 0.5})
    except Exception as exc:  # noqa: BLE001
        import traceback
        res["exception"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-800:]
    finally:
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
        out[rank] = res


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="多卡评测协议多进程体检（CPU/gloo）")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=180.0, help="整体超时（秒）；超时判死锁")
    a = ap.parse_args(argv)
    if not dist.is_available():
        print("[smoke] ⚠️ 本机 torch.distributed 不可用 ⇒ 跳过（退出码 3）")
        return 3

    n = max(2, int(a.procs))
    port = _free_port()
    tmp = tempfile.mkdtemp(prefix="multigpu_smoke_")
    print(f"[smoke] 进程数={n} 端口={port} 临时目录={tmp}")
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    out = mgr.dict()
    procs = [ctx.Process(target=_worker, args=(r, n, port, tmp, False, out), daemon=True)
             for r in range(n)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=a.timeout)
    hung = [i for i, p in enumerate(procs) if p.is_alive()]
    if hung:
        print(f"[smoke] ❌ 疑似**死锁**：rank {hung} 在 {a.timeout}s 内未结束 ⇒ 杀掉并判失败")
        for p in procs:
            p.terminate()
        return 2
    elapsed = time.perf_counter() - t0

    results = {int(k): dict(v) for k, v in dict(out).items()}
    problems: List[str] = []
    for r in range(n):
        res = results.get(r)
        if res is None:
            problems.append(f"rank{r}: 无结果（进程可能崩溃）")
            continue
        if res.get("exception"):
            problems.append(f"rank{r}: 异常 {res['exception']}")
            continue
        if res.get("global_rank") != r:
            problems.append(f"rank{r}: _global_rank()={res.get('global_rank')} 不符")
        if not res.get("normal_result_ok"):
            problems.append(f"rank{r}: 广播结果不一致")
        if r != 0 and res.get("normal_local_calls", 1) != 0:
            problems.append(f"rank{r}: 非 0 rank 竟然自己算了 {res.get('normal_local_calls')} 次")
        if not (res.get("fail_raised") and res.get("fail_is_runtime_error")
                and res.get("fail_mentions_cause")):
            problems.append(f"rank{r}: rank0 失败时本 rank 没有正确抛（{res}）")
        if not res.get("group_reusable"):
            problems.append(f"rank{r}: 失败路径后通信组不可再用")
        # ---- 分片并行（all_ranks_run=True）----
        if res.get("allranks_local_calls") != 1:
            problems.append(f"rank{r}: all_ranks_run=True 时本 rank 应恰好跑 1 次，"
                            f"实际 {res.get('allranks_local_calls')}")
        if not res.get("allranks_result_is_rank0") or not res.get("allranks_result_ok"):
            problems.append(f"rank{r}: all_ranks_run=True 的结果必须统一用 rank0 的")
        if not res.get("allranks_nonzero_failure_tolerated"):
            problems.append(f"rank{r}: 非 0 rank 自己失败不应影响协议结果")
        if not res.get("allranks_rank0_failure_raised"):
            problems.append(f"rank{r}: all_ranks_run 下 rank0 失败时本 rank 必须抛")
        elif "synthetic-rank0-failure-in-allranks" not in str(res.get("allranks_rank0_failure_msg")):
            problems.append(f"rank{r}: rank0 失败原因没广播过来: {res.get('allranks_rank0_failure_msg')}")
        if not res.get("allranks_group_reusable"):
            problems.append(f"rank{r}: all_ranks_run 失败路径后通信组不可再用")

    ev = Path(tmp) / "events.jsonl"
    lines = [l for l in ev.read_text(encoding="utf-8").splitlines() if l.strip()] if ev.is_file() else []
    if len(lines) != 2:
        problems.append(f"事件流行数={len(lines)}，应为 2（只有 rank0 写；每个 rank 写 2 条会变成 {2 * n}）")
    scout = sorted((Path(tmp) / "scout").rglob("*.json"))
    if len(scout) != 1:
        problems.append(f"scout 缓存文件数={len(scout)}，应为 1（只有 rank0 写）")

    print(f"[smoke] 用时 {elapsed:.1f}s；每 rank 结果：")
    for r in range(n):
        res = results.get(r, {})
        print(f"   rank{r}: 广播一致={res.get('normal_result_ok')} 本地计算次数={res.get('normal_local_calls')} "
              f"失败路径抛错={res.get('fail_raised')} 组可复用={res.get('group_reusable')}")
        print(f"   rank{r}: [分片] 本地计算次数={res.get('allranks_local_calls')} "
              f"结果=rank0:{res.get('allranks_result_is_rank0')} "
              f"非0失败可容忍={res.get('allranks_nonzero_failure_tolerated')} "
              f"rank0失败一起抛={res.get('allranks_rank0_failure_raised')} "
              f"组可复用={res.get('allranks_group_reusable')}")
    print(f"[smoke] 事件流行数={len(lines)}（期望 2）  scout 缓存文件数={len(scout)}（期望 1）")
    if problems:
        print("[smoke] ❌ 发现问题：")
        for p in problems:
            print("   -", p)
        for r in range(n):
            if results.get(r, {}).get("traceback"):
                print(f"   --- rank{r} traceback ---\n{results[r]['traceback']}")
        return 2
    print("[smoke] ✅ 多卡评测协议体检通过（广播一致 / 非 0 rank 不自算 / 失败不死锁 / 组可复用 / "
          "分片模式所有 rank 一起跑且统一用 rank0 结果 / 写入守卫）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
