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
  所有人拿 rank0 的结果、rank0 失败时所有 rank 一起抛（带原因），**没有任何人挂死**；
* 🆕 `test_fsdp2_open_loop_before_any_root_forward_cpu_gloo`：**根模块一次前向都没跑**
  （= AutoLearning bootstrap 的真实时序）就直接评测 ⇒ 修复前随后训练前向报
  ``FSDP requires running forward through the root module first``，修复后全流程（评测 →
  训练前向 → backward → 再评测）正常。

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
#: 本次修复新增的「根单元惰性初始化」。**故意不放进 `_WANTED_FUNCS`**：
#: 老代码里没有它们 ⇒ 修复前只有「单元级」断言降级，而**真症状**（评测后训练前向报
#: `FSDP requires running forward through the root module first`、子单元被当成根）
#: 仍然照常断言 ⇒ 修复前的失败证据是「真 bug」而不是「函数不存在」。
_OPTIONAL_FUNCS = ("_fsdp2_state", "_fsdp2_root_lazy_init")
_WANTED_CONSTS = ("SHARDED_EVAL_MODES", "MULTIRANK_EVAL_MODES")


def _helpers() -> Dict[str, Any]:
    """AST 编译被测函数（本机无法 import 该模块：缺 torchdata）。"""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    names = set(_WANTED_FUNCS) | set(_OPTIONAL_FUNCS)
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", None) in _WANTED_CONSTS for t in targets):
                body.append(node)
    got = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    assert set(_WANTED_FUNCS) <= got, sorted(set(_WANTED_FUNCS) - got)
    ast.fix_missing_locations(tree)
    ns: Dict[str, Any] = {"Any": object, "List": list, "Dict": dict, "Callable": object,
                          "Iterator": object, "Tuple": tuple, "Optional": object,
                          "contextlib": contextlib, "torch": torch}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return {k: ns[k] for k in names if k in ns}


def _fsdp_state_of(module) -> Any:
    """直接向模型要 FSDPState（**不依赖**本次新增的函数 ⇒ 修复前也能断言 `_is_root` 归属）。"""
    getter = getattr(module, "_get_fsdp_state", None)
    return getter() if callable(getter) else None


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


def _run_workers(ws: int = 2, timeout: float = 240.0, target=None):
    """起 ws 个进程；超时即判**疑似死锁**（杀掉并失败）。"""
    if not dist.is_available():
        pytest.skip("本机 torch.distributed 不可用")
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    out = mgr.dict()
    port = _port()
    procs = [ctx.Process(target=target or _worker, args=(r, ws, port, out), daemon=True)
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


def test_root_lazy_init_runs_before_unshard_in_source():
    """源码顺序护栏：`_fsdp_full_params_context` 里**根 init 必须在 unshard 之前**。

    顺序反了（先 unshard/前向、后 init 根）正是本次真机 crash 的成因；行为测试
    `test_fsdp2_open_loop_before_any_root_forward_cpu_gloo` 已经覆盖，这条只是把
    「顺序」这个不变量显式钉在源码上，避免将来重构时被无声调换。
    """
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_fsdp_full_params_context")
    nodes = list(ast.walk(fn))
    init_calls = [n for n in nodes if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == "_fsdp2_root_lazy_init"]
    assert init_calls, "`_fsdp_full_params_context` 里没有 `_fsdp2_root_lazy_init` 调用 ⇒ 评测前没有根 init"
    unshard_call = next(n for n in nodes if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                        and isinstance(n.value.func, ast.Name) and n.value.func.id == "unshard")
    assert init_calls[0].lineno < unshard_call.lineno, "根单元惰性初始化必须排在 unshard() 之前"
    # unshard 在 try 之外、reshard 在 finally 里（窗口成对 ⇒ rank0 失败也能一起退出）
    try_node = next(n for n in nodes if isinstance(n, ast.Try))
    assert unshard_call.lineno < try_node.lineno, "unshard() 必须在 try 之前"
    assert try_node.finalbody, "reshard() 必须在 finally 里"


# ---------------------------------------------------------------------------
# 🆕 根模块**一次前向都没跑**就评测（= AutoLearning bootstrap 的真实时序）
# ---------------------------------------------------------------------------
def _worker_fresh_root(rank: int, ws: int, port: int, out) -> None:
    """复现真机 crash 的时序：**没有任何训练前向**，开局直接进评测窗口（真 FSDP2/FSDP1）。

    真机（ROCm 2 卡/8 卡）表现：
    * 2 卡：评测那一步不报错，但**紧随其后的训练第一步前向**在根 ``_lazy_init`` 里报
      ``RuntimeError: FSDP state has already been lazily initialized for <layer>
      / FSDP requires running forward through the root module first``；
    * 8 卡：更早一步 ``HIP error: an illegal memory access was encountered`` ⇒ SIGABRT。
    同一根因：评测走的 ``sample_actions`` 绕过根模块的前向 hook，子单元各自被当成「根」。
    """
    res: Dict[str, Any] = {}
    try:
        dist.init_process_group("gloo", rank=rank, world_size=ws,
                                init_method=f"tcp://127.0.0.1:{port}")
        H = _helpers()

        from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
        from torch.distributed.fsdp import ShardingStrategy

        mesh = init_device_mesh("cpu", (ws,))
        x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 10.0   # 各 rank 完全一致

        # ---------------- FSDP2：开局直接评测（不跑任何根前向）----------------
        torch.manual_seed(0)
        ref = _Root()                      # 未分片的参考模型：验证 warm-up 不改参数数值
        torch.manual_seed(0)
        m2 = _Root()
        for b in m2.blocks:
            fully_shard(b, mesh=mesh)
        fully_shard(m2, mesh=mesh)
        res["before_root_is_root"] = _fsdp_state_of(m2)._is_root            # 必须还是 None
        res["before_child_is_root"] = [_fsdp_state_of(b)._is_root for b in m2.blocks]
        res["before_param_type"] = type(m2.head.weight).__name__
        res["before_local_shard"] = [round(v, 7) for v in
                                     m2.head.weight._local_tensor.reshape(-1).tolist()]

        # ① 生产入口（`_fsdp_full_params_context` = validate/evaluate_ids 走的那个）：
        #    修复前它只 unshard，不做根 init ⇒ 下面每一行断言都会暴露问题。
        rng_before = torch.get_rng_state().clone()
        with H["_fsdp_full_params_context"](m2) as kind:
            res["ctx_kind"] = kind
            res["in_window_param_type"] = type(m2.head.weight).__name__
            res["param_not_inference"] = not bool(m2.head.weight.is_inference())
            y_local = m2.sample_actions(x)
        res["after_window_param_type"] = type(m2.head.weight).__name__
        res["rng_untouched"] = bool(torch.equal(rng_before, torch.get_rng_state()))
        res["no_grad_after_eval"] = all(p.grad is None for p in m2.parameters())
        res["after_root_is_root"] = _fsdp_state_of(m2)._is_root
        res["after_child_is_root"] = [_fsdp_state_of(b)._is_root for b in m2.blocks]
        res["after_local_shard_same"] = (res["before_local_shard"] ==
                                         [round(v, 7) for v in
                                          m2.head.weight._local_tensor.reshape(-1).tolist()])

        # ② 窗口内取到的是**完整且正确**的权重：根单元自有参数逐位等于未分片参考模型；
        #    自定义方法的输出也与参考模型逐位一致（嵌套单元由各自的前向 hook 取全）。
        #    ⚠️ 不能直接比 `b.lin.weight`：unshard() 只作用于**根单元**，嵌套单元在窗口里
        #    仍是 DTensor（要等它自己的前向 hook 才取全 —— 这正是 FSDP2 的设计）。
        with H["_fsdp_full_params_context"](m2):
            res["head_param_matches_ref"] = bool(torch.equal(m2.head.weight, ref.head.weight))
            y_ref = m2.sample_actions(x)
        res["eval_matches_reference"] = bool(torch.equal(y_ref, ref.sample_actions(x)))

        # ③ 修复前的失败点：评测之后**训练前向**必须正常（不再报 root module first）
        try:
            with torch.no_grad():
                m2(x)
            res["train_forward"] = "OK"
        except Exception as exc:  # noqa: BLE001
            res["train_forward"] = f"FAIL {type(exc).__name__}: {str(exc)[:160]}"
        # ④ backward 也必须正常（参数要回到 DTensor、梯度也要是 DTensor）
        try:
            m2(x).sum().backward()
            res["train_backward"] = "OK"
            res["grad_type"] = type(m2.head.weight.grad).__name__
        except Exception as exc:  # noqa: BLE001
            res["train_backward"] = f"FAIL {type(exc).__name__}: {str(exc)[:160]}"

        # ⑤ 再来一次评测（训练之后）仍然可用，且两个 rank 输出逐位一致
        with H["_fsdp_full_params_context"](m2):
            y2 = m2.sample_actions(x)
        g2: List[Any] = [None] * ws
        dist.all_gather_object(g2, [round(v, 7) for v in y2.reshape(-1).tolist()])
        res["second_eval_outputs_identical"] = bool(all(v == g2[0] for v in g2))
        res["eval_output_first"] = g2[0][0]

        # ⑥ 单元级：`_fsdp2_root_lazy_init` 自己（新模型 = 又没跑过前向）
        #    ⚠️ 老代码里没有这个函数（本次修复新增）⇒ 缺了就只跳过这一节，
        #       上面的真症状断言（评测后训练前向 / _is_root 归属）照常生效。
        res["unit_api_available"] = ("_fsdp2_root_lazy_init" in H and "_fsdp2_state" in H)
        if res["unit_api_available"]:
            torch.manual_seed(0)
            m3 = _Root()
            for b in m3.blocks:
                fully_shard(b, mesh=mesh)
            fully_shard(m3, mesh=mesh)
            res["unit_before_is_root"] = H["_fsdp2_state"](m3)._is_root
            rng_u = torch.get_rng_state().clone()
            res["unit_ret_first"] = H["_fsdp2_root_lazy_init"](m3)
            res["unit_rng_untouched"] = bool(torch.equal(rng_u, torch.get_rng_state()))
            res["unit_ret_second"] = H["_fsdp2_root_lazy_init"](m3)     # 幂等
            res["unit_root_is_root"] = H["_fsdp2_state"](m3)._is_root
            res["unit_child_is_root"] = [H["_fsdp2_state"](b)._is_root for b in m3.blocks]
            res["unit_not_fsdp"] = H["_fsdp2_root_lazy_init"](torch.nn.Linear(2, 2))  # '' 未分片
            with torch.no_grad():
                m3(x)                              # 补一次真前向：状态必须仍然可用
            res["unit_train_forward"] = "OK"

        # ---------------- FSDP1：同样开局直接评测 ----------------
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
            with H["_fsdp_full_params_context"](m1) as kind1:
                res["fsdp1_ctx_kind"] = kind1
                res["fsdp1_in_window_param"] = type(m1.head.weight).__name__
                y1 = m1.sample_actions(x)
            try:
                with torch.no_grad():
                    m1(x)
                m1(x).sum().backward()
                res["fsdp1_train"] = "OK"
            except Exception as exc:  # noqa: BLE001
                res["fsdp1_train"] = f"FAIL {type(exc).__name__}: {str(exc)[:160]}"
            g1: List[Any] = [None] * ws
            dist.all_gather_object(g1, [round(v, 7) for v in y1.reshape(-1).tolist()])
            res["fsdp1_outputs_identical"] = bool(all(v == g1[0] for v in g1))
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


def test_fsdp2_open_loop_before_any_root_forward_cpu_gloo():
    """2 进程 gloo：**根模块还没跑过任何前向**就直接评测（AutoLearning bootstrap 时序）。

    修复前这里必挂：`_fsdp_full_params_context` 只做 unshard，评测的 `sample_actions`
    绕过根 hook ⇒ 子单元各自当根 ⇒ 紧随其后的 `m2(x)` 报
    ``FSDP requires running forward through the root module first``。
    修复后：评测 → 训练前向 → backward → 再评测全绿，且根/子单元的 `_is_root` 归属正确。
    """
    results = _run_workers(ws=2, target=_worker_fresh_root)
    problems: List[str] = []
    for r, res in sorted(results.items()):
        if res.get("fatal"):
            problems.append(f"rank{r} 异常: {res['fatal']}\n{res.get('traceback', '')}")
            continue
        # 前提：确实是在「根模块没跑过前向」的状态下评测的（否则这个用例没意义）
        if res.get("before_root_is_root") is not None:
            problems.append(f"rank{r}: 前置条件不成立 —— 根单元 _is_root="
                            f"{res.get('before_root_is_root')!r}（应还是 None）")
        if any(v is not None for v in (res.get("before_child_is_root") or [None])):
            problems.append(f"rank{r}: 前置条件不成立 —— 子单元已 init: "
                            f"{res.get('before_child_is_root')}")
        if res.get("before_param_type") != "DTensor":
            problems.append(f"rank{r}: 分片参数应为 DTensor，实际 {res.get('before_param_type')}")
        # ① 窗口
        if res.get("ctx_kind") != "fsdp2":
            problems.append(f"rank{r}: 窗口类型应为 fsdp2，实际 {res.get('ctx_kind')}")
        if res.get("in_window_param_type") != "Parameter":
            problems.append(f"rank{r}: 窗口内参数应为完整 Parameter，实际 "
                            f"{res.get('in_window_param_type')}")
        if res.get("after_window_param_type") != "DTensor":
            problems.append(f"rank{r}: 退出窗口后应恢复 DTensor，实际 "
                            f"{res.get('after_window_param_type')}")
        if not res.get("rng_untouched"):
            problems.append(f"rank{r}: 评测窗口（含根 init）动了 torch RNG ⇒ 会影响训练随机流")
        if not res.get("param_not_inference"):
            problems.append(f"rank{r}: 窗口内的参数带 inference 标记（在 inference_mode 里建的？)"
                            f"⇒ 训练 backward 会炸")
        if not res.get("no_grad_after_eval"):
            problems.append(f"rank{r}: 评测（含根 init）产生了梯度 ⇒ 污染训练状态")
        if not res.get("after_local_shard_same"):
            problems.append(f"rank{r}: 根 init 改了参数数值（本地分片前后不一致）")
        # ② 根 init 之后必须是**一棵树**：根 = True、所有子单元 = False
        if res.get("after_root_is_root") is not True:
            problems.append(f"rank{r}: 评测后根单元 _is_root 应为 True，实际 "
                            f"{res.get('after_root_is_root')!r}")
        bad = [i for i, v in enumerate(res.get("after_child_is_root") or []) if v is not False]
        if bad:
            problems.append(f"rank{r}: 子单元 {bad} 被当成根单元（_is_root 应为 False）："
                            f"{res.get('after_child_is_root')}")
        if not res.get("head_param_matches_ref"):
            problems.append(f"rank{r}: 窗口内根参数与未分片参考模型不一致")
        if not res.get("eval_matches_reference"):
            problems.append(f"rank{r}: 窗口内自定义方法的输出与未分片参考模型不一致"
                            f"（= 取全的参数不对）")
        # ③④ 修复前的失败点
        if res.get("train_forward") != "OK":
            problems.append(f"rank{r}: 评测之后的训练前向失败 ⇒ {res.get('train_forward')}")
        if res.get("train_backward") != "OK":
            problems.append(f"rank{r}: 评测之后的 backward 失败 ⇒ {res.get('train_backward')}")
        elif res.get("grad_type") != "DTensor":
            problems.append(f"rank{r}: 梯度应为 DTensor，实际 {res.get('grad_type')}")
        # ⑤ 二次评测
        if not res.get("second_eval_outputs_identical"):
            problems.append(f"rank{r}: 训练后再次评测，各 rank 输出不一致")
        # ⑥ 单元级（本次修复新增的 API）
        if not res.get("unit_api_available"):
            problems.append(f"rank{r}: 源码里没有 `_fsdp2_root_lazy_init`/`_fsdp2_state`"
                            f"（本次修复新增的根单元惰性初始化）⇒ 评测前没有根 init")
        else:
            if res.get("unit_before_is_root") is not None:
                problems.append(f"rank{r}: 单元级前置条件不成立 {res.get('unit_before_is_root')!r}")
            if res.get("unit_ret_first") != "lazy_init":
                problems.append(f"rank{r}: 首次根 init 应返回 'lazy_init'，实际 "
                                f"{res.get('unit_ret_first')!r}")
            if res.get("unit_ret_second") != "already":
                problems.append(f"rank{r}: 根 init 必须幂等（第二次应 'already'），实际 "
                                f"{res.get('unit_ret_second')!r}")
            if not res.get("unit_rng_untouched"):
                problems.append(f"rank{r}: 根 init 动了 torch RNG")
            if res.get("unit_root_is_root") is not True or any(
                    v is not False for v in (res.get("unit_child_is_root") or [True])):
                problems.append(f"rank{r}: 根 init 后 _is_root 归属不对: root="
                                f"{res.get('unit_root_is_root')!r} children="
                                f"{res.get('unit_child_is_root')}")
            if res.get("unit_not_fsdp") != "":
                problems.append(f"rank{r}: 未分片模型应返回空串（单卡/DDP 路径不变），实际 "
                                f"{res.get('unit_not_fsdp')!r}")
            if res.get("unit_train_forward") != "OK":
                problems.append(f"rank{r}: 根 init 之后普通前向失败")
        # FSDP1
        if str(res.get("fsdp1_init", "")).startswith("SKIP"):
            problems.append(f"rank{r}: FSDP1 建不起来（本用例要求真 FSDP1）：{res.get('fsdp1_init')}")
        else:
            if res.get("fsdp1_ctx_kind") != "fsdp1":
                problems.append(f"rank{r}: FSDP1 窗口类型应为 fsdp1，实际 {res.get('fsdp1_ctx_kind')}")
            if res.get("fsdp1_train") != "OK":
                problems.append(f"rank{r}: FSDP1 未跑前向就评测后，训练/backward 失败 ⇒ "
                                f"{res.get('fsdp1_train')}")
            if not res.get("fsdp1_outputs_identical"):
                problems.append(f"rank{r}: FSDP1 各 rank 输出不一致")
    assert not problems, "「根模块未前向就评测」验证失败:\n  - " + "\n  - ".join(problems)
    # 各 rank 的评测输出必须逐位相同（评测结果只信 rank0 的前提）
    outs = {r: v.get("eval_output_first") for r, v in results.items()}
    assert len(set(outs.values())) == 1, f"各 rank 评测输出不一致: {outs}"
