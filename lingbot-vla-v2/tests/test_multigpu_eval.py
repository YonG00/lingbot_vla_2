"""多卡（DDP）评测支持的 CPU 测试。

背景：AL 的 hook 在**所有 rank** 无条件驱动调度器 ⇒ 评测也必须在所有 rank 拿到**同一份**结果，
否则各 rank 决策分叉（各训各的任务）。设计：
* 所有 rank 一起进入评测（保持 DDP 对称）；
* **rank0 计算并广播结果**（广播即同步点）；
* rank0 失败时**广播错误标记**，让所有 rank 一起抛 —— 不会永久等在广播上。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "lingbotvla/utils/open_loop_validation.py"


def _mod():
    """按 AST 编译多卡相关模块级函数（本地无法 import 该模块：transformers 版本差异）。"""
    wanted = ("_world_size", "_global_rank", "_data_parallel_mode", "_ddp_replicated",
              "_broadcast_object", "_multirank_eval_payload", "_unwrap_eval_model")
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    fns = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in wanted]
    assert len(fns) == len(wanted), [f.name for f in fns]
    ast.fix_missing_locations(tree)
    ns = {"Any": object, "List": list, "Dict": dict, "Callable": object, "torch": torch}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(SRC), "exec"), ns)
    return SimpleNamespace(**{k: ns[k] for k in wanted})


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


def test_broadcast_requires_initialized_dist():
    """未初始化 dist ⇒ 明确报错（不静默走单卡逻辑，否则多卡下会静默不同步）。"""
    m = _mod()
    with pytest.raises(RuntimeError, match="torch.distributed"):
        m._broadcast_object([None])


# ---------------------------------------------------------------------------
# 接线（源码级）：三处旧限制必须消失，新协议必须接上
# ---------------------------------------------------------------------------
def test_source_wiring():
    src = SRC.read_text(encoding="utf-8")
    assert "open-loop 评测目前只在 rank0 上做" not in src, "旧的 rank0-only 限制必须移除"
    assert "eval batching unsupported on multiple FSDP ranks" not in src
    assert "目前**只支持单卡**" not in src
    assert "_multirank_eval_payload(" in src and "_evaluate_run" in src
    # 构造检查与批处理守卫都要改成"允许 DDP、拒绝分片"
    assert src.count("_ddp_replicated(") >= 3
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


def test_validator_unwraps_before_assigning_self_model():
    """顺序红线：必须**先解包、再 `self.model = model`** ——
    先赋值后解包的话 `self.model` 仍是包装体，`model.config` 照旧 AttributeError。"""
    src = SRC.read_text(encoding="utf-8")
    # ⚠️ 锚点必须精确：注释里也出现了 "self.model = model" 字样（上一次就被它骗了）
    i_unwrap = src.index("_inner = _unwrap_eval_model(model)")
    i_assign = src.index("self.model = model\n        self._model_config")
    i_config = src.index("self._model_config = model_config if model_config is not None else model.config")
    assert i_unwrap < i_assign < i_config, "顺序应为：解包 → self.model= → 取 model.config"
