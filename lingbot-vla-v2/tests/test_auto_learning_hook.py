"""Stage B1 —— 训练循环接线的**无卡**测试（测试计划 §7.9 + §7.5/§7.7 的接线部分）。

不依赖 torch / GPU：Scheduler 用 Stage A 的假世界，Sampler 用真实现。

    python -m pytest tests/test_auto_learning_hook.py -q
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lingbotvla.auto_learning.config import AutoLearningConfig        # noqa: E402
from lingbotvla.auto_learning.real.hook import (                      # noqa: E402
    AutoLearnLoopHook, StepDirective, build_hook,
)
from lingbotvla.auto_learning.real.sampler import AutoLearnSampler    # noqa: E402
from lingbotvla.auto_learning.sampling.sampler import BatchSampler    # noqa: E402
from lingbotvla.auto_learning.testing import fake_tasks as ft         # noqa: E402

_TMP_ROOT = REPO / ".pytest_tmp"


@pytest.fixture
def tmp_path():
    import shutil
    import uuid

    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    d = _TMP_ROOT / f"alh_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _cfg(**kw):
    # ⚠️ 用 5 而不是 3：默认 min_steps_before_defer=100 与 defer_retry_steps=50
    #    都必须是 eval_interval_steps 的整数倍（config 的校验）
    base = dict(batch_size=10, new_slots=7, replay_slots=3,
                eval_interval_steps=5, seed=7)
    base.update(kw)
    al = AutoLearningConfig(**base)
    return ft.make_cfg([ft._spec("easy_pass", curve=[0.2, 0.18]),
                        ft._spec("slow_learn", curve=[0.9, 0.5, 0.28])], al=al)


def _hook(cfg):
    """用假世界组装 scheduler + sampler，再包成 hook（不碰 torch）。"""
    from lingbotvla.auto_learning.testing.sim import build_backend

    backend = build_backend(cfg)
    from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
    sched = Scheduler(backend, cfg.auto_learning, seed=cfg.sim.seed)
    sampler = AutoLearnSampler(
        BatchSampler(cfg.auto_learning, backend.resolver, backend.catalog, sched.rng),
        batch_size=cfg.auto_learning.batch_size)
    return AutoLearnLoopHook(scheduler=sched, sampler=sampler, cfg=cfg.auto_learning), sched, sampler


# --------------------------------------------------------------------------- #
# §7.9  disabled 回归
# --------------------------------------------------------------------------- #
def test_build_hook_returns_none_when_disabled():
    assert build_hook(None) is None
    assert build_hook(AutoLearningConfig(enabled=False)) is None


def test_disabled_hook_does_not_touch_backend():
    """`enabled=false` ⇒ 不创建对象、不触碰 scheduler / sampler（零副作用）。"""

    class _Spy:
        def __init__(self):
            self.calls = 0

        def __getattr__(self, item):
            self.calls += 1
            raise AssertionError(f"disabled 时不该访问 {item}")

    s, p = _Spy(), _Spy()
    assert build_hook(AutoLearningConfig(enabled=False), scheduler=s, sampler=p) is None
    assert s.calls == 0 and p.calls == 0


def test_enabled_hook_requires_scheduler_and_sampler():
    with pytest.raises(ValueError) as err:
        build_hook(AutoLearningConfig(enabled=True))
    assert "scheduler" in str(err.value)


def test_enabled_hook_sets_defer_train():
    cfg = _cfg()
    hook, sched, _ = _hook(cfg)
    assert hook.enabled is True
    assert sched.defer_train is True, "hook 必须让 scheduler 进入延迟训练模式"


# --------------------------------------------------------------------------- #
# 延迟训练流程（unit 边界 → 发布 request → 回填 result）
# --------------------------------------------------------------------------- #
def test_first_step_begin_returns_rebuild_directive():
    cfg = _cfg()
    hook, sched, sampler = _hook(cfg)
    d = hook.on_step_begin(0)
    assert isinstance(d, StepDirective)
    assert d.rebuild_iterator is True, "unit 边界必须要求重建 DataLoader 迭代器（丢弃 prefetch）"
    assert d.request is not None and d.unit_steps == cfg.auto_learning.eval_interval_steps
    assert d.request.batch_size == 10
    assert sched.pending_train_request is not None


def test_steps_within_unit_do_not_ask_for_rebuild():
    cfg = _cfg()
    hook, _, sampler = _hook(cfg)
    n = cfg.auto_learning.eval_interval_steps
    d0 = hook.on_step_begin(0)
    assert d0.rebuild_iterator is True
    it = iter(sampler)
    for i in range(n):
        [next(it) for _ in range(cfg.auto_learning.batch_size)]
        hook.on_step_end(i, loss=0.1)
        d = hook.on_step_begin(i + 1)
        if i < n - 1:
            assert d.rebuild_iterator is False, f"unit 内第 {i + 1} 步不该要求重建"
        else:
            assert d.rebuild_iterator is True, "unit 跑完 ⇒ 下一边界应当要求重建"


def test_unit_completes_and_advances_scheduler():
    cfg = _cfg()
    hook, sched, sampler = _hook(cfg)
    n = cfg.auto_learning.eval_interval_steps

    hook.on_step_begin(0)
    # 模拟真实循环：每步从 sampler 取 batch_size 个 index
    it = iter(sampler)
    for step in range(n):
        [next(it) for _ in range(cfg.auto_learning.batch_size)]
        hook.on_step_end(step, loss=0.2)
    assert sched.pending_train_request is not None, "回填前应当还挂着 pending"

    # 跨到下一步 ⇒ 自动 drain 上一个 unit
    hook.on_step_begin(n)
    assert sched.state.units_run >= 1, "unit 应当被记账"
    assert sched.state.global_step >= n


def test_unit_samples_seen_matches_batch_size_times_steps():
    cfg = _cfg()
    hook, sched, sampler = _hook(cfg)
    n = cfg.auto_learning.eval_interval_steps
    bs = cfg.auto_learning.batch_size

    hook.on_step_begin(0)
    it = iter(sampler)
    for step in range(n):
        [next(it) for _ in range(bs)]
        hook.on_step_end(step, loss=0.1)
    before = sched.state.global_samples_seen
    hook.on_step_begin(n)
    assert sched.state.global_samples_seen - before == n * bs


def test_invariant_violation_fails_fast():
    """采样步数与 scheduler 要求不符 ⇒ 立刻报错（测试计划 §12 fail-fast）。"""
    cfg = _cfg()
    hook, _, sampler = _hook(cfg)
    n = cfg.auto_learning.eval_interval_steps
    hook.on_step_begin(0)
    it = iter(sampler)
    for step in range(n - 1):          # 故意少跑一步
        [next(it) for _ in range(cfg.auto_learning.batch_size)]
        hook.on_step_end(step, loss=0.1)
    with pytest.raises(RuntimeError) as err:
        hook.on_unit_end(0)            # 显式结束 ⇒ 必须发现步数不符
    assert "步" in str(err.value)


def test_driver_can_run_until_finished():
    """把整个 4-task 流程驱动完，不应死循环（§7.10 的接线版）。"""
    cfg = _cfg()
    hook, sched, sampler = _hook(cfg)
    n = cfg.auto_learning.eval_interval_steps
    it = iter(sampler)
    for _ in range(400):
        d = hook.on_step_begin(sched.state.global_step)
        if d.finished:
            break
        if d.rebuild_iterator:
            it = iter(sampler)
        for _ in range(n):
            [next(it) for _ in range(cfg.auto_learning.batch_size)]
            hook.on_step_end(0, loss=0.1)
    else:
        pytest.fail("400 个 unit 还没结束 ⇒ 疑似死循环")
    assert sched.state.finished


# --------------------------------------------------------------------------- #
# extra_state
# --------------------------------------------------------------------------- #
def test_extra_state_roundtrip(tmp_path):
    cfg = _cfg()
    hook, sched, sampler = _hook(cfg)
    hook.on_step_begin(0)
    it = iter(sampler)
    for step in range(cfg.auto_learning.eval_interval_steps):
        [next(it) for _ in range(cfg.auto_learning.batch_size)]
        hook.on_step_end(step, loss=0.1)

    raw = hook.extra_state()
    assert raw["version"] == 1
    for key in ("scheduler", "registry", "sampler", "sampler_rng",
                "step_in_unit", "unit_steps", "steps_done"):
        assert key in raw, f"extra_state 缺字段 {key}"

    # 恢复到一个全新 hook
    hook2, sched2, _ = _hook(cfg)
    hook2.load_extra_state(raw)
    assert sched2.state.global_step == sched.state.global_step
    assert sched2.state.units_run == sched.state.units_run
    assert sched2.rng.getstate() == sched.rng.getstate()
    assert sched2.defer_train is True, "恢复后必须仍在延迟训练模式"


def test_extra_state_rejects_bad_version():
    cfg = _cfg()
    hook, _, _ = _hook(cfg)
    with pytest.raises(ValueError):
        hook.load_extra_state({"version": 99})


# --------------------------------------------------------------------------- #
# 🔴 resume 落在 unit 中途（G9 Phase 2 实测踩到的坑，2026-10-07）
# --------------------------------------------------------------------------- #
def test_resume_mid_unit_republishes_request():
    """存档落在 unit 中途 ⇒ 恢复后**必须**已经重新发布 request。

    修复前：`on_step_begin()` 因 `_step_in_unit(1) < _unit_steps(5)` 直接
    early-return，既不发布 request 也不要求重建 ⇒ 循环开头的
    `iter(train_dataloader)` 撞上「AutoLearnSampler 还没收到 TrainRequest」。
    """
    cfg = _cfg()
    n = cfg.auto_learning.eval_interval_steps
    bs = cfg.auto_learning.batch_size
    hook, sched, sampler = _hook(cfg)

    # 跑到 unit 中途（1/n 步）后存档
    hook.on_step_begin(0)
    it = iter(sampler)
    [next(it) for _ in range(bs)]
    hook.on_step_end(1, loss=0.3)
    raw = hook.extra_state()
    assert raw["step_in_unit"] == 1 and raw["unit_steps"] == n

    hook2, sched2, sampler2 = _hook(cfg)
    hook2.load_extra_state(raw)

    # ① 恢复后必须已经有 pending request，否则下面的 iter 会炸
    assert sched2.pending_train_request is not None
    it2 = iter(sampler2)                       # 修复前这里抛 RuntimeError
    d = hook2.on_step_begin(0)                 # prime：unit 内 ⇒ 不要求重建
    assert d.rebuild_iterator is False
    assert d.unit_steps == n

    # ② 本 unit 从头重跑满 n 步 ⇒ 记账完整（scheduler 只看到一个自洽的 unit）
    for _ in range(n):
        [next(it2) for _ in range(bs)]
        hook2.on_step_end(0, loss=0.1)
    assert sched2.state.units_run == sched.state.units_run
    hook2.on_step_begin(n)
    assert sched2.state.units_run == sched.state.units_run + 1
    assert sched2.state.global_samples_seen == n * bs


def test_resume_at_unit_end_does_not_silently_drop_the_unit():
    """存档正好落在「unit 已跑满、还没回填」⇒ 恢复后该 unit 仍会被记账。"""
    cfg = _cfg()
    n = cfg.auto_learning.eval_interval_steps
    bs = cfg.auto_learning.batch_size
    hook, sched, sampler = _hook(cfg)

    hook.on_step_begin(0)
    it = iter(sampler)
    for _ in range(n):
        [next(it) for _ in range(bs)]
        hook.on_step_end(0, loss=0.1)
    assert sched.pending_train_request is not None      # 还没回填
    raw = hook.extra_state()
    assert raw["step_in_unit"] == n

    hook2, sched2, sampler2 = _hook(cfg)
    hook2.load_extra_state(raw)
    it2 = iter(sampler2)
    for _ in range(n):
        [next(it2) for _ in range(bs)]
        hook2.on_step_end(0, loss=0.1)
    hook2.on_step_begin(n)
    assert sched2.state.units_run == sched.state.units_run + 1, \
        "恢复后这个 unit 必须被记账，不能静默丢掉"


def test_resume_at_unit_start_republishes_request():
    """存档落在「unit 已开、0 步还没跑」⇒ 也必须重新发布 request。

    判据必须是 `_unit_steps > 0`（有 unit 在飞），而不是 `_step_in_unit > 0`：
    此例 `step_in_unit == 0` 但 request 已经丢了，只按后者判会漏掉。
    """
    cfg = _cfg()
    n = cfg.auto_learning.eval_interval_steps
    bs = cfg.auto_learning.batch_size
    hook, sched, sampler = _hook(cfg)

    hook.on_step_begin(0)
    it = iter(sampler)
    for _ in range(n):
        [next(it) for _ in range(bs)]
        hook.on_step_end(0, loss=0.1)
    hook.on_step_begin(n)                       # 跨边界 ⇒ 回填 + 开下一个 unit
    raw = hook.extra_state()
    assert raw["step_in_unit"] == 0 and raw["unit_steps"] == n

    hook2, sched2, sampler2 = _hook(cfg)
    hook2.load_extra_state(raw)
    assert sched2.pending_train_request is not None, \
        "unit 在飞（哪怕 0 步）⇒ 恢复时必须重发 request"
    iter(sampler2)                              # 修复前这里抛 RuntimeError
    assert hook2.on_step_begin(0).rebuild_iterator is False


def test_resume_with_no_unit_in_flight_is_a_noop():
    """`_unit_steps == 0`（没有 unit 在飞）⇒ 恢复时不该乱发 request。"""
    cfg = _cfg()
    hook, sched, sampler = _hook(cfg)
    raw = hook.extra_state()
    assert raw["unit_steps"] == 0 and raw["step_in_unit"] == 0

    hook2, sched2, sampler2 = _hook(cfg)
    hook2.load_extra_state(raw)
    assert sched2.pending_train_request is None, "没有在飞的 unit ⇒ 不该凭空发 request"
    d = hook2.on_step_begin(0)                  # prime 负责开第一个 unit
    assert d.rebuild_iterator is True
    iter(sampler2)


def test_extra_state_is_json_serialisable(tmp_path):
    import json

    cfg = _cfg()
    hook, _, sampler = _hook(cfg)
    hook.on_step_begin(0)
    it = iter(sampler)
    for step in range(cfg.auto_learning.eval_interval_steps):
        [next(it) for _ in range(cfg.auto_learning.batch_size)]
        hook.on_step_end(step, loss=0.1)
    blob = json.dumps(hook.extra_state(), default=str)
    assert len(blob) > 100
