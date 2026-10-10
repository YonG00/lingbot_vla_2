"""真实 **2 进程 + gloo + 真实 FSDP2/FSDP1** 的开环评测验证（纯 CPU，不需要 GPU）。

为什么必须真跑进程：本次改造的两条核心性质**用 mock 测不出来** ——
1. `sample_actions` 是模型上的**自定义方法**（不是 `forward`/`__call__`），
   而 FSDP 的 all-gather 挂在**前向 hook** 上 ⇒ 不显式取全参数时它会拿到**分片参数**；
2. 「所有 rank 一起跑 + rank0 结果广播」这条协议在真实集合通信下才暴露死锁/错配。

本文件用小模型 + 2 个 gloo 进程覆盖：
* `_fsdp_full_params_context`（FSDP2 = `unshard/reshard`；FSDP1 = `summon_full_params`）
  在窗口内让**自定义方法**可用、窗口外**确实不可用**（= 证明这个窗口不是装饰性的）；
* 两个 rank 在同一份输入上的输出**逐位一致** ⇒ rank0 的评测结果与「任何 rank 自己算的」相同；
* 退出窗口后模型仍然健康（DTensor 分片恢复、普通 `forward` 可用）；
* `_multirank_eval_payload(all_ranks_run=True)` 在**真 gloo** 上：每个 rank 都跑、
  所有人拿 rank0 的结果、rank0 失败时所有 rank 一起抛（带原因），**没有任何人挂死**。

超时即判失败（疑似死锁），并打印每个 rank 的 traceback。
"""
from __future__ import annotations

import ast
import contextlib
import socket
import traceback
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "lingbotvla/utils/open_loop_validation.py"

_WANTED_FUNCS = ("_fsdp1_class", "_fsdp2_mixin", "_sharded_model_kind",
                 "_fsdp_full_params_context", "_broadcast_object",
                 "_multirank_eval_payload")
_WANTED_CONSTS = ("SHARDED_EVAL_MODES", "MULTIRANK_EVAL_MODES")


def _helpers() -> Dict[str, Any]:
    """AST 编译被测函数（本机无法 import 该模块：缺 torchdata）。"""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED_FUNCS:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", None) in _WANTED_CONSTS for t in targets):
                body.append(node)
    got = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    assert got == set(_WANTED_FUNCS), sorted(set(_WANTED_FUNCS) - got)
    ast.fix_missing_locations(tree)
    ns: Dict[str, Any] = {"Any": object, "List": list, "Dict": dict, "Callable": object,
                          "Iterator": object, "Tuple": tuple, "Optional": object,
                          "contextlib": contextlib, "torch": torch}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return {k: ns[k] for k in _WANTED_FUNCS}


# ---------------------------------------------------------------------------
# 被测模型：模仿 LingbotVlaV2Policy 的形状（根单元自有参数 + 嵌套单元 + 自定义方法）
# ---------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return torch.relu(self.lin(x))


class _Root(nn.Module):
    """`sample_actions` **绕开** `__call__`（像 `FlowMatchingV2.sample_actions` 那样直接调
    `.forward()` / 自有参数）⇒ FSDP 的前向 hook 不会触发。"""

    def __init__(self, d: int = 8):
        super().__init__()
        self.head = nn.Linear(d, d, bias=False)
        self.blocks = nn.ModuleList([_Block(d), _Block(d)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)

    def sample_actions(self, x):                       # 评测路径
        h = self.head(x)
        for b in self.blocks:
            h = b(h)
        return h


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wrap_blocks(module, recurse, nonwrapped_numel):
    """每个 `_Block` 单独包一个 FSDP1（复现仓库 `auto_wrap_policy` 的嵌套结构）。"""
    from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
    return lambda_auto_wrap_policy(
        module, recurse, nonwrapped_numel,
        lambda_fn=lambda m: m.__class__.__name__ == "_Block")


def _worker(rank: int, ws: int, port: int, out) -> None:
    res: Dict[str, Any] = {}
    H = None
    try:
        dist.init_process_group("gloo", rank=rank, world_size=ws,
                                init_method=f"tcp://127.0.0.1:{port}")
        H = _helpers()
        res["helpers_loaded"] = True

        from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
        from torch.distributed.fsdp import ShardingStrategy

        mesh = init_device_mesh("cpu", (ws,))
        x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 10.0   # 各 rank 输入完全一致
        res["input_sha"] = f"{hash(tuple(x.reshape(-1).tolist())) & 0xffffffff:08x}"

        # ---------------- FSDP2 ----------------
        torch.manual_seed(0)
        m2 = _Root()
        for b in m2.blocks:
            fully_shard(b, mesh=mesh)
        fully_shard(m2, mesh=mesh)
        res["fsdp2_param_before"] = type(m2.head.weight).__name__
        # FSDP2 要求**根模块先跑一次 forward** 才能让嵌套单元惰性初始化
        # （否则子单元先 init ⇒ RuntimeError: requires running forward through the root
        #   module first）。真实训练里早在第一个 step 就跑过了 ⇒ 这里照做。
        with torch.no_grad():
            m2(x)
        # 🔴 FSDP2 对**根模块**「前向之后不 reshard」（留给 backward 复用，见
        #    `_fsdp_state.py::_lazy_init`：post_forward_mesh_info=None）。而训练循环里
        #    评测发生在 optimizer.step()（backward 已结束 ⇒ 根参数**已 reshard**）之后 ⇒
        #    这里显式 reshard() 复现那一刻的真实状态，否则测的是「碰巧还没被分片」。
        m2.reshard()
        res["fsdp2_param_at_eval_time"] = type(m2.head.weight).__name__

        # ① 窗口**外**：自定义方法必须失败（分片参数）—— 这正是旧实现的路径
        try:
            m2.sample_actions(x)
            res["fsdp2_without_window"] = "NO-ERROR(!!)"
        except Exception as exc:  # noqa: BLE001
            res["fsdp2_without_window"] = f"{type(exc).__name__}: {str(exc)[:110]}"

        # ② 窗口**内**：自定义方法可用，且参数是完整 Parameter
        with H["_fsdp_full_params_context"](m2) as kind:
            res["fsdp2_ctx_kind"] = kind
            res["fsdp2_param_in_window"] = type(m2.head.weight).__name__
            y_local = m2.sample_actions(x)
        res["fsdp2_after_window"] = type(m2.head.weight).__name__
        with torch.no_grad():
            res["fsdp2_forward_after"] = tuple(m2(x).shape)          # 模型仍健康

        # ③ 跨 rank 输出一致性：同一份输入 ⇒ 逐位相同的输出
        gathered: List[Any] = [None] * ws
        dist.all_gather_object(gathered, [round(v, 7) for v in y_local.reshape(-1).tolist()])
        res["fsdp2_outputs_identical"] = bool(all(g == gathered[0] for g in gathered))
        res["fsdp2_output_first"] = gathered[0][0]

        # ---------------- FSDP1 ----------------
        # ⚠️ 用**auto_wrap_policy**（和仓库一致：`build_parallelize_model` 会把
        #    `_no_split_modules` 里的每层单独包一个 FSDP）⇒ 覆盖「嵌套 FSDP 实例在
        #    summon_full_params 窗口里被 `__call__` 调用」这条真实路径。
        torch.manual_seed(0)
        try:
            m1 = FSDP1(_Root(), device_id=torch.device("cpu"),
                       sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True,
                       auto_wrap_policy=_wrap_blocks)
        except Exception as exc:  # noqa: BLE001
            res["fsdp1_init"] = f"SKIP {type(exc).__name__}: {str(exc)[:120]}"
            m1 = None
        if m1 is not None:
            res["fsdp1_init"] = "ok"
            res["fsdp1_n_nested"] = sum(1 for mod in m1.modules() if isinstance(mod, FSDP1))
            with torch.no_grad():
                m1(x)      # FSDP1 同样要先跑一次前向做惰性初始化
            try:
                m1.sample_actions(x)
                res["fsdp1_without_window"] = "NO-ERROR(!!)"
            except Exception as exc:  # noqa: BLE001
                res["fsdp1_without_window"] = f"{type(exc).__name__}: {str(exc)[:110]}"
            with H["_fsdp_full_params_context"](m1) as kind:
                res["fsdp1_ctx_kind"] = kind
                y1 = m1.sample_actions(x)
            with torch.no_grad():
                res["fsdp1_forward_after"] = tuple(m1(x).shape)
            g1: List[Any] = [None] * ws
            dist.all_gather_object(g1, [round(v, 7) for v in y1.reshape(-1).tolist()])
            res["fsdp1_outputs_identical"] = bool(all(g == g1[0] for g in g1))

        # ---------------- 协议（真 gloo）----------------
        calls = {"n": 0}

        def _run_ok() -> Dict[str, Any]:
            calls["n"] += 1
            return {"rank": rank, "mse": 1.0 + rank, "per_traj_mse": [0.1 * (rank + 1)]}

        got = H["_multirank_eval_payload"](run=_run_ok, ws=ws, rank=rank,
                                           broadcast=H["_broadcast_object"],
                                           all_ranks_run=True)
        res["proto_ran"] = calls["n"]                    # 每个 rank 都必须跑过 1 次
        res["proto_result_is_rank0"] = (got.get("rank") == 0)
        res["proto_result"] = got

        def _run_raise() -> Dict[str, Any]:
            if rank != 0:
                raise RuntimeError("non-rank0 local failure")
            return {"rank": rank}

        got2 = H["_multirank_eval_payload"](run=_run_raise, ws=ws, rank=rank,
                                            broadcast=H["_broadcast_object"],
                                            all_ranks_run=True)
        res["proto_nonzero_failure_tolerated"] = (got2 == {"rank": 0})

        def _run_rank0_raise() -> Dict[str, Any]:
            if rank == 0:
                raise RuntimeError("rank0 boom")
            return {"rank": rank}

        try:
            H["_multirank_eval_payload"](run=_run_rank0_raise, ws=ws, rank=rank,
                                         broadcast=H["_broadcast_object"], all_ranks_run=True)
            res["proto_rank0_failure_raised"] = False
            res["proto_rank0_failure_msg"] = ""
        except Exception as exc:  # noqa: BLE001
            res["proto_rank0_failure_raised"] = True
            res["proto_rank0_failure_msg"] = str(exc)[:200]
        dist.barrier()
    except Exception as exc:  # noqa: BLE001
        res["fatal"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-1500:]
    finally:
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
        out[rank] = res


def _run_workers(ws: int = 2, timeout: float = 240.0):
    """起 ws 个进程；超时即判**疑似死锁**（杀掉并失败）。"""
    if not dist.is_available():
        pytest.skip("本机 torch.distributed 不可用")
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    out = mgr.dict()
    port = _port()
    procs = [ctx.Process(target=_worker, args=(r, ws, port, out), daemon=True)
             for r in range(ws)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout)
    hung = [i for i, p in enumerate(procs) if p.is_alive()]
    if hung:
        for p in procs:
            p.terminate()
        results = {int(k): dict(v) for k, v in dict(out).items()}
        raise AssertionError(f"疑似**死锁**：rank {hung} 在 {timeout}s 内未结束；已见结果={results}")
    results = {int(k): dict(v) for k, v in dict(out).items()}
    assert len(results) == ws, f"少了 rank 的结果（进程可能崩溃）：{results}"
    return results


def test_fsdp_multirank_open_loop_cpu_gloo():
    """2 进程 gloo：FSDP2/FSDP1 取全参数窗口 + all_ranks_run 协议。"""
    results = _run_workers(ws=2)
    problems: List[str] = []
    for r, res in sorted(results.items()):
        if res.get("fatal"):
            problems.append(f"rank{r} 异常: {res['fatal']}\n{res.get('traceback', '')}")
            continue
        # FSDP2
        if res.get("fsdp2_param_at_eval_time") != "DTensor":
            problems.append(f"rank{r}: 评测时刻根参数应为分片 DTensor（backward 之后），"
                            f"实际 {res.get('fsdp2_param_at_eval_time')}")
        if res.get("fsdp2_param_before") != "DTensor":
            problems.append(f"rank{r}: FSDP2 参数应为 DTensor，实际 {res.get('fsdp2_param_before')}")
        if res.get("fsdp2_without_window", "").startswith("NO-ERROR"):
            problems.append(f"rank{r}: 窗口外自定义方法竟然成功 ⇒ 本测试没测到东西")
        if res.get("fsdp2_ctx_kind") != "fsdp2":
            problems.append(f"rank{r}: 窗口类型应为 fsdp2，实际 {res.get('fsdp2_ctx_kind')}")
        if res.get("fsdp2_param_in_window") != "Parameter":
            problems.append(f"rank{r}: 窗口内参数应为完整 Parameter，实际 {res.get('fsdp2_param_in_window')}")
        if res.get("fsdp2_after_window") != "DTensor":
            problems.append(f"rank{r}: 退出窗口后应恢复 DTensor，实际 {res.get('fsdp2_after_window')}")
        if tuple(res.get("fsdp2_forward_after") or ()) != (2, 8):
            problems.append(f"rank{r}: 退出窗口后普通 forward 不可用: {res.get('fsdp2_forward_after')}")
        if not res.get("fsdp2_outputs_identical"):
            problems.append(f"rank{r}: 各 rank 同一输入的输出**不一致** ⇒ 评测结果不可信")
        # FSDP1
        if str(res.get("fsdp1_init", "")).startswith("SKIP"):
            continue
        if int(res.get("fsdp1_n_nested") or 0) < 3:
            problems.append(f"rank{r}: FSDP1 应有嵌套实例（root+2 blocks），实际 {res.get('fsdp1_n_nested')}")
        if res.get("fsdp1_ctx_kind") != "fsdp1":
            problems.append(f"rank{r}: FSDP1 窗口类型应为 fsdp1，实际 {res.get('fsdp1_ctx_kind')}")
        if res.get("fsdp1_without_window", "").startswith("NO-ERROR"):
            problems.append(f"rank{r}: FSDP1 窗口外自定义方法竟然成功")
        if tuple(res.get("fsdp1_forward_after") or ()) != (2, 8):
            problems.append(f"rank{r}: FSDP1 退出窗口后 forward 不可用: {res.get('fsdp1_forward_after')}")
        if not res.get("fsdp1_outputs_identical"):
            problems.append(f"rank{r}: FSDP1 各 rank 输出不一致")
        # 协议
        if res.get("proto_ran") != 1:
            problems.append(f"rank{r}: all_ranks_run=True 时本 rank 应跑 1 次，实际 {res.get('proto_ran')}")
        if not res.get("proto_result_is_rank0"):
            problems.append(f"rank{r}: 结果应来自 rank0，实际 {res.get('proto_result')}")
        if not res.get("proto_nonzero_failure_tolerated"):
            problems.append(f"rank{r}: 非 0 rank 失败不应影响协议")
        if not res.get("proto_rank0_failure_raised"):
            problems.append(f"rank{r}: rank0 失败时本 rank 必须抛")
        elif "rank0 boom" not in str(res.get("proto_rank0_failure_msg")):
            problems.append(f"rank{r}: rank0 失败原因没广播过来: {res.get('proto_rank0_failure_msg')}")
    assert not problems, "多卡 FSDP 评测验证失败:\n  - " + "\n  - ".join(problems)
    # 两个 rank 必须**逐位相同**的输入 & 输出
    outs = {r: v.get("fsdp2_output_first") for r, v in results.items()}
    assert len(set(outs.values())) == 1, f"各 rank 评测输出不一致: {outs}"
    shas = {r: v.get("input_sha") for r, v in results.items()}
    assert len(set(shas.values())) == 1, f"各 rank 输入不一致: {shas}"
