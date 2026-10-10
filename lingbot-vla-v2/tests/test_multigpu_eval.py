"""多卡评测支持的 CPU 测试。

背景：AL 的 hook 在**所有 rank** 无条件驱动调度器 ⇒ 评测也必须在所有 rank 拿到**同一份**结果，
否则各 rank 决策分叉（各训各的任务）。设计：
* 所有 rank 一起进入评测（保持对称）；
* **复制并行（ddp）**：rank0 计算并广播结果（广播即同步点）；
* **分片并行（fsdp1/fsdp2/fsdp2-vescale）**：**每个 rank 都真跑前向**（FSDP 的 all-gather
  是集合通信，少一个 rank 就挂死），但只有 rank0 的结果算数，广播后所有 rank 用同一份；
  评测窗口内用 `_fsdp_full_params_context` 把分片参数 all-gather 回完整权重；
* rank0 失败时**广播错误标记**，让所有 rank 一起抛 —— 不会永久等在广播上。

（真实 FSDP2/FSDP1 的 all-gather 行为在 `tests/test_openloop_multirank_fsdp.py`
用**两个真进程 + gloo** 验证 —— 本文件只测纯逻辑。）
"""
from __future__ import annotations

import ast
import contextlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "lingbotvla/utils/open_loop_validation.py"

#: 被编译的函数会引用这些模块级常量 ⇒ 一并从**源码**里取（不复制字面量，避免测试与实现漂移）
_WANTED_CONSTS = ("SHARDED_EVAL_MODES", "MULTIRANK_EVAL_MODES",
                  "FSDP_EVAL_UNSUPPORTED_ENV", "MULTIRANK_BATCH_ENV", "EVAL_SEED")

_WANTED_FUNCS = ("_world_size", "_global_rank", "_data_parallel_mode", "_ddp_replicated",
                 "_is_sharded_mode", "_fsdp_eval_unsupported", "_multirank_eval_supported",
                 "_all_ranks_must_run_eval", "_multirank_batch_allowed",
                 "_fsdp1_class", "_fsdp2_mixin", "_sharded_model_kind",
                 "_fsdp_full_params_context",
                 "_broadcast_object", "_multirank_eval_payload", "_unwrap_eval_model")


def _mod():
    """按 AST 编译多卡相关模块级函数（本地无法 import 该模块：缺 torchdata/transformers 版本差异）。"""
    wanted = _WANTED_FUNCS
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", None) in _WANTED_CONSTS for t in targets):
                body.append(node)
    got = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    assert got == set(wanted), f"源码里找不到这些函数: {sorted(set(wanted) - got)}"
    ast.fix_missing_locations(tree)
    ns = {"Any": object, "List": list, "Dict": dict, "Callable": object, "Iterator": object,
          "Tuple": tuple, "Optional": object, "torch": torch, "os": os,
          "contextlib": contextlib}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return SimpleNamespace(**{k: ns[k] for k in list(wanted) + list(_WANTED_CONSTS)})


def _args(mode):
    return SimpleNamespace(train=SimpleNamespace(data_parallel_mode=mode))


def _simulate_protocol(m, *, ranks, all_ranks_run, run_by_rank):
    """按 rank 顺序**忠实模拟**一次 `broadcast_object_list`（rank0 写、其余读 rank0 的）。

    真实语义：每个 rank 各自持有 payload；rank0 的那份是真相（`src=0`），
    其余 rank 的内容会被 rank0 的覆盖。这里按 rank 升序调用 ⇒ rank0 先写进 box。
    返回 ``(每个 rank 的返回值 / 抛出的异常, 每个 rank 调用 run() 的次数)``。
    """
    box = {"rank0_payload": None}
    calls = {r: 0 for r in ranks}
    out = {}
    for r in sorted(ranks):
        def _run(rr=r):
            calls[rr] += 1
            return run_by_rank[rr]()

        def _bc(payload, rr=r):
            if rr == 0:
                box["rank0_payload"] = payload[0]
            else:
                payload[0] = box["rank0_payload"]

        try:
            out[r] = m._multirank_eval_payload(run=_run, ws=len(ranks), rank=r,
                                               broadcast=_bc, all_ranks_run=all_ranks_run)
        except BaseException as exc:  # noqa: BLE001
            out[r] = exc
    return out, calls


# ---------------------------------------------------------------------------
# 单卡路径：逐字不变
# ---------------------------------------------------------------------------
def test_single_rank_runs_locally_without_broadcast():
    m = _mod()
    calls = {"run": 0, "bc": 0}
    out = m._multirank_eval_payload(run=lambda: (calls.__setitem__("run", calls["run"] + 1),
                                                 {"mse": 1.0})[1],
                                    ws=1, rank=0,
                                    broadcast=lambda p: calls.__setitem__("bc", calls["bc"] + 1))
    assert out == {"mse": 1.0} and calls == {"run": 1, "bc": 0}


# ---------------------------------------------------------------------------
# 多卡：rank0 算 + 广播；其他 rank 只用广播结果
# ---------------------------------------------------------------------------
def test_rank0_computes_and_broadcasts():
    m = _mod()
    seen = {}

    def _bc(payload):
        seen["payload"] = list(payload)

    out = m._multirank_eval_payload(run=lambda: {"mse": 0.25, "per_traj_mse": [1, 2]},
                                    ws=2, rank=0, broadcast=_bc)
    assert out == {"mse": 0.25, "per_traj_mse": [1, 2]}
    assert seen["payload"][0]["ok"] is True and seen["payload"][0]["result"] == out


def test_nonzero_rank_does_not_compute_and_uses_broadcast():
    m = _mod()
    calls = {"run": 0}

    def _bc(payload):                      # 模拟 rank0 填好结果
        payload[0] = {"ok": True, "result": {"mse": 0.5}}

    out = m._multirank_eval_payload(run=lambda: (calls.__setitem__("run", 1), {})[1],
                                    ws=2, rank=1, broadcast=_bc)
    assert out == {"mse": 0.5}
    assert calls["run"] == 0, "非 0 rank 不得自己算（否则各 rank 指标可能不同 ⇒ 决策分叉）"


def test_rank0_failure_is_broadcast_not_raised_locally():
    """rank0 失败 ⇒ 必须**广播错误**（否则对端永久等在广播上），本进程也要抛。"""
    m = _mod()
    seen = {}

    def _run():
        raise RuntimeError("CUDA out of memory")

    def _bc(payload):
        seen["payload"] = list(payload)

    with pytest.raises(RuntimeError, match="多卡评测在 rank0 失败"):
        m._multirank_eval_payload(run=_run, ws=2, rank=0, broadcast=_bc)
    assert seen["payload"][0]["ok"] is False
    assert "CUDA out of memory" in seen["payload"][0]["error"]


def test_peer_raises_when_rank0_failed():
    m = _mod()

    def _bc(payload):
        payload[0] = {"ok": False, "error": "RuntimeError: boom"}

    with pytest.raises(RuntimeError, match="boom"):
        m._multirank_eval_payload(run=lambda: {}, ws=2, rank=1, broadcast=_bc)


def test_empty_broadcast_payload_is_rejected():
    m = _mod()
    with pytest.raises(RuntimeError, match="广播失败"):
        m._multirank_eval_payload(run=lambda: {}, ws=2, rank=1, broadcast=lambda p: None)


# ---------------------------------------------------------------------------
# 并行模式判定
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode,ok", [("ddp", True), ("fsdp1", False), ("fsdp2", False),
                                     ("fsdp2-vescale", False), (None, True)])
def test_only_ddp_is_replicated(mode, ok):
    m = _mod()
    args = SimpleNamespace(train=SimpleNamespace(data_parallel_mode=mode))
    assert m._ddp_replicated(args) is ok
    assert m._data_parallel_mode(args) == (mode or "ddp")


def test_missing_args_defaults_to_ddp():
    m = _mod()
    assert m._ddp_replicated(SimpleNamespace()) is True


# ---------------------------------------------------------------------------
# 多卡评测模式白名单（2026-10-10：fsdp1/fsdp2/fsdp2-vescale 解禁）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["dp", "ddp", "fsdp1", "fsdp2", "fsdp2-vescale"])
def test_sharded_modes_are_now_supported(mode):
    """`ws>1 + fsdp2` **不再**抛错（这是本次改造的核心诉求）。"""
    m = _mod()
    assert m._multirank_eval_supported(_args(mode)) is True, mode


@pytest.mark.parametrize("mode", ["hsdp", "tp", "bogus"])
def test_unknown_modes_still_fail_fast(mode):
    """白名单之外的模式**不猜**：宁可直接报错，也不要拿分片参数算出无效指标。"""
    m = _mod()
    assert m._multirank_eval_supported(_args(mode)) is False, mode


@pytest.mark.parametrize("mode,all_ranks", [("ddp", False), ("dp", False), ("fsdp1", True),
                                            ("fsdp2", True), ("fsdp2-vescale", True)])
def test_all_ranks_run_only_for_sharded(mode, all_ranks):
    """分片并行：**每个 rank 都要跑前向**（all-gather 是集合通信）；复制并行只有 rank0 算。"""
    m = _mod()
    assert m._all_ranks_must_run_eval(_args(mode)) is all_ranks, mode


def test_sharded_modes_are_superset_of_replicated():
    m = _mod()
    for mode in ("dp", "ddp"):
        assert mode in m.MULTIRANK_EVAL_MODES
    for mode in ("fsdp1", "fsdp2", "fsdp2-vescale"):
        assert mode in m.SHARDED_EVAL_MODES and mode in m.MULTIRANK_EVAL_MODES
    assert not set(m.SHARDED_EVAL_MODES) & {"dp", "ddp"}


# ---------------------------------------------------------------------------
# 逃生开关：AL_EVAL_FSDP_UNSUPPORTED=1 ⇒ 回到旧的 fail-fast 行为
# ---------------------------------------------------------------------------
def test_escape_hatch_restores_old_behaviour(monkeypatch):
    m = _mod()
    monkeypatch.setenv("AL_EVAL_FSDP_UNSUPPORTED", "1")
    assert m._fsdp_eval_unsupported() is True
    for mode in ("fsdp1", "fsdp2", "fsdp2-vescale"):
        assert m._multirank_eval_supported(_args(mode)) is False, mode   # 旧的 fail-fast
        assert m._all_ranks_must_run_eval(_args(mode)) is False, mode
    assert m._multirank_eval_supported(_args("ddp")) is True             # ddp 仍然可用
    assert m._multirank_eval_supported(SimpleNamespace()) is True        # 缺省=ddp


def test_escape_hatch_is_off_by_default(monkeypatch):
    m = _mod()
    monkeypatch.delenv("AL_EVAL_FSDP_UNSUPPORTED", raising=False)
    assert m._fsdp_eval_unsupported() is False
    assert m._multirank_eval_supported(_args("fsdp2")) is True


def test_multirank_batch_escape_hatch(monkeypatch):
    """多卡分片默认**禁止批处理**（批内 OOM 回退是本 rank 局部决策 ⇒ all-gather 次数错配）。"""
    m = _mod()
    monkeypatch.delenv("AL_EVAL_BATCH_MULTIRANK", raising=False)
    assert m._multirank_batch_allowed() is False
    monkeypatch.setenv("AL_EVAL_BATCH_MULTIRANK", "1")
    assert m._multirank_batch_allowed() is True


# ---------------------------------------------------------------------------
# all_ranks_run=True 的协议（分片并行）：每个 rank 都算，但只认 rank0 的结果
# ---------------------------------------------------------------------------
def test_all_ranks_run_calls_run_on_every_rank():
    m = _mod()
    out, calls = _simulate_protocol(
        m, ranks=[0, 1, 2], all_ranks_run=True,
        run_by_rank={r: (lambda rr=r: {"mse": 0.1 * rr, "who": rr}) for r in (0, 1, 2)})
    assert calls == {0: 1, 1: 1, 2: 1}, "分片并行下**每个** rank 都必须真跑前向"
    assert all(v == {"mse": 0.0, "who": 0} for v in out.values()), out


def test_all_ranks_run_uses_rank0_result_everywhere():
    """各 rank 拿到的指标必须**完全相同**（决策不分叉）——即使每个 rank 都自己算过。"""
    m = _mod()
    out, _ = _simulate_protocol(
        m, ranks=[0, 1], all_ranks_run=True,
        run_by_rank={0: lambda: {"mse": 1.0}, 1: lambda: {"mse": 2.0}})
    assert out[0] == out[1] == {"mse": 1.0}


def test_replicated_mode_keeps_rank0_only_compute():
    """复制并行（ddp）行为**逐字不变**：非 0 rank 不自算。"""
    m = _mod()
    out, calls = _simulate_protocol(
        m, ranks=[0, 1], all_ranks_run=False,
        run_by_rank={0: lambda: {"mse": 3.0}, 1: lambda: {"mse": 4.0}})
    assert calls == {0: 1, 1: 0}
    assert out[0] == out[1] == {"mse": 3.0}


def test_all_ranks_run_nonzero_failure_does_not_disturb_protocol():
    """非 0 rank 自己失败 ⇒ 不抛出、不影响结果（用 rank0 的）。"""
    m = _mod()
    def _boom():
        raise RuntimeError("rank1 local OOM (non-collective)")
    out, calls = _simulate_protocol(
        m, ranks=[0, 1], all_ranks_run=True,
        run_by_rank={0: lambda: {"mse": 5.0}, 1: _boom})
    assert calls == {0: 1, 1: 1}
    assert out[0] == out[1] == {"mse": 5.0}


def test_all_ranks_run_rank0_failure_is_broadcast():
    """rank0 失败 ⇒ 所有 rank 一起抛（不能让对端死等）。"""
    m = _mod()
    def _boom():
        raise RuntimeError("CUDA out of memory (rank0)")
    out, _ = _simulate_protocol(
        m, ranks=[0, 1], all_ranks_run=True,
        run_by_rank={0: _boom, 1: lambda: {"mse": 9.9}})
    for r in (0, 1):
        assert isinstance(out[r], RuntimeError) and "rank0 失败" in str(out[r])
    assert "CUDA out of memory (rank0)" in str(out[1])


def test_single_rank_path_unchanged_with_all_ranks_flag():
    """单卡（ws=1）：无论 all_ranks_run 取值都直接本地跑、不广播。"""
    m = _mod()
    for flag in (False, True):
        calls = {"n": 0, "bc": 0}
        out = m._multirank_eval_payload(
            run=lambda: (calls.__setitem__("n", calls["n"] + 1), {"mse": 7.0})[1],
            ws=1, rank=0,
            broadcast=lambda p: calls.__setitem__("bc", calls["bc"] + 1),
            all_ranks_run=flag)
        assert out == {"mse": 7.0} and calls == {"n": 1, "bc": 0}


# ---------------------------------------------------------------------------
# 分片识别（结构性，不看 data_parallel_mode）+ 取全参数窗口
# ---------------------------------------------------------------------------
def test_plain_module_is_not_sharded():
    m = _mod()
    assert m._sharded_model_kind(torch.nn.Linear(2, 2)) == ""


def test_full_params_context_is_noop_on_plain_model():
    m = _mod()
    layer = torch.nn.Linear(2, 2)
    with m._fsdp_full_params_context(layer) as kind:
        assert kind == ""
        layer(torch.ones(1, 2))          # 窗口内可正常前向


def test_full_params_context_yields_inside_and_restores():
    """`_fsdp_full_params_context` 是 contextmanager：yield 后**一定**回到原状态。"""
    m = _mod()
    seen = []
    with m._fsdp_full_params_context(torch.nn.Linear(2, 2)) as kind:
        seen.append(("in", kind))
    seen.append(("out", None))
    assert seen == [("in", ""), ("out", None)]


def test_broadcast_requires_initialized_dist():
    """未初始化 dist ⇒ 明确报错（不静默走单卡逻辑，否则多卡下会静默不同步）。"""
    m = _mod()
    with pytest.raises(RuntimeError, match="torch.distributed"):
        m._broadcast_object([None])


# ---------------------------------------------------------------------------
# 各 rank 输入一致：噪声种子固定 + 每次评测重置；episode_ids 文件原子写
# ---------------------------------------------------------------------------
def _validator_class(methods):
    """AST 编译 `OpenLoopValidator` 的指定方法（本地无法 import 该模块）。"""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "OpenLoopValidator")
    keep = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    got = {n.name for n in keep}
    assert got == set(methods), sorted(set(methods) - got)
    body = [ast.ClassDef(name="OpenLoopValidator", bases=[], keywords=[],
                         body=keep, decorator_list=[])]
    for node in tree.body:                       # 方法依赖的模块级函数/常量
        if isinstance(node, ast.FunctionDef) and node.name == "_global_rank":
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", None) == "EVAL_SEED" for t in targets):
                body.append(node)
    ast.fix_missing_locations(tree)
    ns = {"os": os, "json": json, "torch": torch, "Any": object, "List": list, "Dict": dict,
          "Sequence": object, "Tuple": tuple, "Optional": object}
    new_mod = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(new_mod)          # 新建的 ClassDef 要补行号才能 compile
    exec(compile(new_mod, str(SRC), "exec"), ns)
    return ns["OpenLoopValidator"]


def test_noise_generator_is_seed_fixed_and_identical_across_ranks():
    """各 rank 的 noise 必须**逐位相同**：独立 generator + 固定 EVAL_SEED（不掺 rank）。"""
    V = _validator_class(("_noise_generator",))

    def _mk():
        v = V.__new__(V)
        v._noise_gen = None
        v.device = "cpu"
        return v

    shape = (1, 4, 3)
    ranks = [_mk() for _ in range(3)]                      # 三个「rank」
    draws = [torch.randn(shape, generator=v._noise_generator("cpu")) for v in ranks]
    assert all(torch.equal(draws[0], d) for d in draws[1:]), "各 rank 噪声不一致"

    # 每次评测重置（`_run` / `_evaluate_run` 都置 None）⇒ 不同 step 之间可比
    for v in ranks:
        v._noise_gen = None
    draws2 = [torch.randn(shape, generator=v._noise_generator("cpu")) for v in ranks]
    assert torch.equal(draws[0], draws2[0]), "重置后没有回到同一个噪声序列起点"
    assert all(torch.equal(draws2[0], d) for d in draws2[1:])

    # 源码级：两处入口都必须重置 generator
    src = SRC.read_text(encoding="utf-8")
    assert src.count("self._noise_gen = None") >= 3, "每次评测都要重置噪声起点"


def test_episode_ids_file_same_content_every_rank(tmp_path):
    """同一组 ids ⇒ 各 rank 写出的白名单文件**内容相同**（evaluation 输入一致的前提）。"""
    V = _validator_class(("_episode_ids_file",))
    args = SimpleNamespace(train=SimpleNamespace(output_dir=str(tmp_path)))
    v = V.__new__(V)
    v.args = args
    p0 = v._episode_ids_file([99, 1, 5], "val")
    assert json.loads(Path(p0).read_text()) == [1, 5, 99]
    p1 = v._episode_ids_file([5, 99, 1], "val")            # 另一个「rank」同样的 ids
    assert p0 == p1 and json.loads(Path(p1).read_text()) == [1, 5, 99]
    assert not list(tmp_path.rglob("*.tmp")), "临时文件必须已被 os.replace 吃掉"


def test_episode_ids_file_write_is_atomic_under_concurrency(tmp_path):
    """多卡下所有 rank 写**同一个路径** ⇒ 必须原子替换，读方绝不能读到写了一半的 JSON。"""
    import threading

    V = _validator_class(("_episode_ids_file",))
    g = V._episode_ids_file.__globals__
    tls = threading.local()
    g["_global_rank"] = lambda: getattr(tls, "rank", 0)     # 每个线程扮演一个 rank
    args = SimpleNamespace(train=SimpleNamespace(output_dir=str(tmp_path)))

    ids = list(range(200))
    stop = threading.Event()
    bad: List[str] = []

    def _writer(rank: int):
        tls.rank = rank
        v = V.__new__(V)
        v.args = args
        for _ in range(60):
            v._episode_ids_file(list(reversed(ids)), "val")

    def _reader():
        tls.rank = 99
        v = V.__new__(V)
        v.args = args
        p = v._episode_ids_file(ids, "val")
        while not stop.is_set():
            try:
                if json.loads(Path(p).read_text()) != ids:
                    bad.append("读到不完整/不一致的白名单")
            except Exception as exc:  # noqa: BLE001
                bad.append(f"{type(exc).__name__}: {exc}")

    writers = [threading.Thread(target=_writer, args=(r,)) for r in range(4)]
    reader = threading.Thread(target=_reader)
    reader.start()
    for t in writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    reader.join()
    assert not bad, f"并发写破坏了白名单文件: {bad[:3]}"
    assert json.loads(Path(tmp_path, "_open_loop_ids", "val.json").read_text()) == sorted(ids)


def test_noise_seed_constant_is_not_rank_dependent():
    """噪声/输入路径里**不得**出现按 rank 变化的分支（否则各 rank 输入不同 ⇒ 结果不可比）。"""
    src = SRC.read_text(encoding="utf-8")
    noise = src[src.index("def _noise_generator"):src.index("def _probe_identity_for")]
    assert "_global_rank" not in noise and "RANK" not in noise and "rank" not in noise.replace(
        "不能靠", ""), noise


# ---------------------------------------------------------------------------
# 接线（源码级）：三处旧限制必须消失，新协议必须接上
# ---------------------------------------------------------------------------
def test_source_wiring():
    src = SRC.read_text(encoding="utf-8")
    assert "open-loop 评测目前只在 rank0 上做" not in src, "旧的 rank0-only 限制必须移除"
    assert "eval batching unsupported on multiple FSDP ranks" not in src
    assert "目前**只支持单卡**" not in src
    assert "_multirank_eval_payload(" in src and "_evaluate_run" in src
    # 三处旧守卫（构造 / 批处理 / evaluate_ids）都改成「模式白名单 + 分片取全参数」
    assert src.count("_multirank_eval_supported(") >= 3
    assert src.count("_all_ranks_must_run_eval(") >= 4
    # 旧的 "只支持 DDP" 硬失败文案必须消失
    assert "多卡评测只支持" not in src
    # 分片模式的取全参数窗口必须存在，且用结构化识别（不看 data_parallel_mode）
    assert "_fsdp_full_params_context" in src and "summon_full_params" in src
    assert 'unshard()' in src and "FSDPModule" in src
    # 探针证据目录按 rank 分目录（多卡下不许互相覆盖）
    assert "rank{_global_rank()}" in src


# ---------------------------------------------------------------------------
# 事件流写入：多卡下只允许 rank0 写（否则事件 JSONL 会重复 N 份并互相交错）
# ---------------------------------------------------------------------------
def test_event_logger_writes_only_when_enabled(tmp_path):
    from lingbotvla.auto_learning.real.build import SchedulerLoggerAdapter

    class _Log:
        def info_rank0(self, *a, **k):
            pass

        info = warning = info_rank0

    path = tmp_path / "auto_learning_events.jsonl"
    a = SchedulerLoggerAdapter(_Log(), writer=None, event_path=str(path), write_events=False)
    a.log_event({"kind": "event", "action": "bootstrap", "signature": "x"})
    a.log_text(1, "text/current_task", "task_a")
    assert not path.exists(), "write_events=False（多卡的 rank!=0）不得写事件文件"

    b = SchedulerLoggerAdapter(_Log(), writer=None, event_path=str(path), write_events=True)
    b.log_event({"kind": "event", "action": "bootstrap", "signature": "y"})
    b.log_text(2, "text/current_task", "task_b")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and '"bootstrap"' in lines[0]


def test_event_logger_default_follows_rank(tmp_path):
    """未显式传 `write_events` ⇒ 按 dist rank 自动判定（无 dist ⇒ rank0 ⇒ 写）。"""
    from lingbotvla.auto_learning.real.build import SchedulerLoggerAdapter, _is_rank0

    assert _is_rank0() is True
    path = tmp_path / "e.jsonl"
    a = SchedulerLoggerAdapter(type("L", (), {"info_rank0": lambda *x, **k: None})(),
                               writer=None, event_path=str(path))
    a.log_event({"kind": "event", "action": "select"})
    assert path.is_file() and path.read_text(encoding="utf-8").strip()


def test_scout_cache_write_guard(tmp_path):
    """多卡：只有 rank0 写 scout 缓存（避免并发写撕裂）；读不受影响。"""
    import json
    from lingbotvla.auto_learning.scout_cache import BootstrapScoutCache

    fp = "a" * 64
    metrics = {"task": "click_bell", "episode_ids": [1, 2], "mse": 0.5}
    ro = BootstrapScoutCache(tmp_path, fingerprint=fp, write_enabled=False)
    ro.store("click_bell", [1, 2], metrics)
    assert not ro.path.exists() or not list(ro.path.glob("*.json")), "非 rank0 不得写缓存"

    rw = BootstrapScoutCache(tmp_path, fingerprint=fp, write_enabled=True)
    rw.store("click_bell", [1, 2], metrics)
    files = list(rw.path.glob("*.json"))
    assert len(files) == 1
    doc = json.loads(files[0].read_text(encoding="utf-8"))
    assert doc["task"] == "click_bell" and doc["fingerprint"] == fp


# ---------------------------------------------------------------------------
# 训练器：非 0 rank 的 writer（2026-10-09 2×4090 实测的 UnboundLocalError）
# ---------------------------------------------------------------------------
def test_trainer_writer_defined_for_nonzero_ranks():
    """`writer` 只在 rank0 分支里赋值 ⇒ rank≠0 走 `finish_auto_learning(writer=writer)` 会
    `UnboundLocalError`。必须在 rank0 分支**之前**无条件初始化。"""
    src = (REPO / "tasks/vla/train_lingbotvla.py").read_text(encoding="utf-8")
    assert "writer = None" in src, "必须在 rank0 分支前把 writer 初始化为 None"
    i_none = src.index("writer = None")
    i_ctor = src.index("writer = AsyncTBWriter(")
    i_al = src.index("writer=writer, logger=logger, use_depth_align=use_depth_align")
    assert i_none < i_ctor < i_al, "初始化顺序应为：writer=None → rank0 建 writer → AL 使用"
    # 所有 writer.add_* 必须在 rank0 守卫内（置 None 才安全）
    add_lines = [l for l in src.splitlines() if "writer.add_" in l]
    assert add_lines, "找不到 writer.add_* 调用（源码结构变了？）"


# ---------------------------------------------------------------------------
# 模型包装剥离（2026-10-09 2×4090 DDP 实测：`model.config` AttributeError）
# ---------------------------------------------------------------------------
def _unwrap():
    """AST 编译（本地 import 该模块会拉 torchdata，本机没有）。"""
    return _mod()._unwrap_eval_model


class _FakeInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = "cfg"


def test_unwrap_peels_compile_and_ddp():
    u = _unwrap()
    inner = _FakeInner()
    # torch.compile 的外层（按类名识别，不必真编译）
    opt = type("OptimizedModule", (torch.nn.Module,), {})( )
    opt._orig_mod = inner
    assert u(opt) is inner
    # DDP 类名 + .module
    ddp = type("DistributedDataParallel", (torch.nn.Module,), {})()
    ddp.module = inner
    assert u(ddp) is inner
    # 叠加：compile(DDP(inner))
    both = type("OptimizedModule", (torch.nn.Module,), {})()
    both._orig_mod = ddp
    assert u(both) is inner
    assert u(inner) is inner
    assert u(inner).config == "cfg"


def test_unwrap_does_not_strip_plain_submodule_named_module():
    """普通模型可能有名为 `module` 的子模块 —— **不能**误剥（这正是不能用 'has .module' 判定的原因）。"""
    u = _unwrap()
    outer = torch.nn.Module()
    outer.module = _FakeInner()          # 普通子模块，不是 DDP 包装
    assert u(outer) is outer, "非包装类的 .module 不得被剥掉"


def test_unwrap_keeps_fsdp1_wrapper():
    """🔴 FSDP1 **不能剥**（2026-10-10 修正）。

    `FullyShardedDataParallel.module` 里是**分片**参数 ⇒ 剥掉后前向要么报
    `RuntimeError: size mismatch ...`、要么结果无效（CPU/gloo 实测，见
    `tests/test_openloop_multirank_fsdp.py`）。评测改用 `summon_full_params` 取全参数。
    """
    u = _unwrap()
    inner = _FakeInner()
    fsdp1 = type("FullyShardedDataParallel", (torch.nn.Module,), {})()
    fsdp1.module = inner
    assert u(fsdp1) is fsdp1, "FSDP1 包装必须保留（它的 forward 才做 all-gather）"
    # 叠加 compile(FSDP1(model))：只剥 compile，保留 FSDP1
    opt = type("OptimizedModule", (torch.nn.Module,), {})()
    opt._orig_mod = fsdp1
    assert u(opt) is fsdp1


def test_validator_unwraps_before_assigning_self_model():
    """顺序红线：必须**先解包、再 `self.model = model`** ——
    先赋值后解包的话 `self.model` 仍是包装体，`model.config` 照旧 AttributeError。"""
    src = SRC.read_text(encoding="utf-8")
    # ⚠️ 锚点必须精确：注释里也出现了 "self.model = model" 字样（上一次就被它骗了）
    i_unwrap = src.index("_inner = _unwrap_eval_model(model)")
    i_assign = src.index("self.model = model\n        self._model_config")
    i_config = src.index("self._model_config = model_config if model_config is not None else model.config")
    assert i_unwrap < i_assign < i_config, "顺序应为：解包 → self.model= → 取 model.config"


# ---------------------------------------------------------------------------
# 48G 多卡上"批处理全程不生效"的根因（2026-10-09 2×4090 实测）
# ---------------------------------------------------------------------------
def test_vram_gate_is_gone():
    """**显存闸门已废除**（用户 2026-10-09 定："去掉闸门，没用"）。

    历史：组批前 `free_before < reserve + 12`（22 GiB 硬阈值）在 2×48G DDP 上
    （已占 26.5 GB、空闲 21.4 GiB）把**每一组**都判成不足 ⇒ 批处理全程不生效；
    96G 单卡又形同虚设。真正的保护是**经验性的 OOM 回退**。
    """
    src = SRC.read_text(encoding="utf-8")
    # ⚠️ 用**代码级锚点**：废弃说明的注释里会出现同名文字（这轮已踩三次），
    #    裸子串断言会误判 ⇒ 一律断言"那行代码"不存在。
    for gone in ("os.environ.get('AL_EVAL_BATCH_RESERVE_GIB'",
                 "os.environ.get('AL_EVAL_BATCH_HEADROOM_GIB'",
                 "free_before = torch.cuda.mem_get_info()",
                 "or free_before < reserve",
                 "eval batch peak VRAM headroom guard failed"):
        assert gone not in src, f"{gone} 必须随闸门一起移除（只在废弃注释里提及是可以的）"
    # （`reserve + 12` 这类字样只允许出现在"历史教训"注释里 ⇒ 不做裸子串断言）
    # 机械守卫只剩两条**正确性**约束（不是数值/显存预判）
    assert "take < 2 or not identical_tensor_shapes(" in src
    # 且必须仍会记录"闸门已废除"这件事（避免以后有人又加回来）
    assert "显存闸门已废除" in src


def test_oom_falls_back_to_single_group():
    """批内 OOM ⇒ 该组退回单条（不打断长跑）；连续 2 次 ⇒ 批大小减半（下限 2）。"""
    src = SRC.read_text(encoding="utf-8")
    assert "except torch.cuda.OutOfMemoryError as _oom:" in src
    assert "empty_cache()" in src and "_oom_groups += 1" in src
    # **一次 OOM 就降级**（用户 2026-10-09 定）：>2 ⇒ 立即减半并本 run 保持；==2 ⇒ 整轮关闭
    assert "if batch_size > 2:" in src and "self._eval_batch_oom_cap = batch_size" in src
    assert "self._eval_batch_disabled = True" in src
    assert "其中 OOM 回退" in src, "小结里必须能看到 OOM 回退组数"


def test_trainer_mem_log_is_env_gated_and_syntactically_sane():
    """扫 micro 用的峰值显存日志：**默认零开销**（`AL_MEM_LOG` 未设则不统计），且必须能编译。

    踩过：第一次把统计块插进了 f-string 链中间 ⇒ SyntaxError（整条 `logger.info_rank0(...)`
    是多行 f-string 拼接，插入点必须在**语句之前**）。
    """
    import ast
    src_path = REPO / "tasks/vla/train_lingbotvla.py"
    src = src_path.read_text(encoding="utf-8")
    ast.parse(src)                                     # 必须能编译
    assert "os.environ.get('AL_MEM_LOG')" in src
    assert "_mem_str" in src and "max_memory_reserved()" in src
    # 统计块必须出现在日志语句**之前**，而不是 f-string 链中间
    i_block = src.index("_mem_str = (f\", PeakReserved")
    i_log = src.index("logger.info_rank0(\n                f\"Step {global_step}")
    assert i_block < i_log, "统计块必须在日志语句之前"
    assert src.count("f\"{_mem_str}\"") == 1
