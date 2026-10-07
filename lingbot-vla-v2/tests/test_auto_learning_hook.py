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
    """🔴 review v0.2 #2：unit 的**最后一步跑完就当场回填**，不等下一个 `on_step_begin`。"""
    cfg = _cfg()
    hook, sched, sampler = _hook(cfg)
    n = cfg.auto_learning.eval_interval_steps

    hook.on_step_begin(0)
    # 模拟真实循环：每步从 sampler 取 batch_size 个 index
    it = iter(sampler)
    for step in range(n):
        [next(it) for _ in range(cfg.auto_learning.batch_size)]
        hook.on_step_end(step, loss=0.2)
    # 最后一步结束时就应当已经记账（模型与 Scheduler 不再有「领先一个 unit」的窗口）
    assert sched.pending_train_request is None, "unit 跑满 ⇒ 当场回填，不该还挂着 pending"
    assert sched.state.units_run >= 1, "unit 应当被记账"
    assert sched.state.global_step >= n
    # 且此刻就是安全存档边界
    assert hook.at_safe_checkpoint_boundary is True


def test_unit_samples_seen_matches_batch_size_times_steps():
    cfg = _cfg()
    hook, sched, sampler = _hook(cfg)
    n = cfg.auto_learning.eval_interval_steps
    bs = cfg.auto_learning.batch_size

    hook.on_step_begin(0)
    before = sched.state.global_samples_seen
    it = iter(sampler)
    for step in range(n):
        [next(it) for _ in range(bs)]
        hook.on_step_end(step, loss=0.1)
    # 当场回填 ⇒ 循环结束即已计入
    assert sched.state.global_samples_seen - before == n * bs
    assert hook.at_safe_checkpoint_boundary is True


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
# 🔴 review v0.2 #2：unit 边界 / 存档 / resume 语义
# --------------------------------------------------------------------------- #
def test_resume_mid_unit_fails_fast():
    """存档落在 unit **中途**（已跑过步）⇒ 恢复时**必须 fail-fast**，不静默重跑。

    旧实现在这里「把本 unit 从头重跑」：模型已做 k 次 update，resume 后又多做完整 N 次
    ⇒ 轨迹 / sample exposure / LR 语义都与不中断运行不同。
    """
    cfg = _cfg()
    n = cfg.auto_learning.eval_interval_steps
    bs = cfg.auto_learning.batch_size
    hook, sched, sampler = _hook(cfg)

    hook.on_step_begin(0)
    it = iter(sampler)
    [next(it) for _ in range(bs)]
    hook.on_step_end(1, loss=0.3)          # 只跑了 1/n 步
    raw = hook.extra_state()
    assert raw["step_in_unit"] == 1 and raw["unit_steps"] == n
    assert raw["at_boundary"] is False

    hook2, sched2, sampler2 = _hook(cfg)
    with pytest.raises(RuntimeError) as err:
        hook2.load_extra_state(raw)
    msg = str(err.value)
    assert "unit" in msg and "中途" in msg
    assert "拒绝静默重跑" in msg


def test_resume_at_unit_end_is_already_accounted():
    """🔴 unit 跑满就当场回填 ⇒ 「unit 末尾」其实已经是干净边界（unit_steps==0）。"""
    cfg = _cfg()
    n = cfg.auto_learning.eval_interval_steps
    bs = cfg.auto_learning.batch_size
    hook, sched, sampler = _hook(cfg)

    hook.on_step_begin(0)
    it = iter(sampler)
    for _ in range(n):
        [next(it) for _ in range(bs)]
        hook.on_step_end(0, loss=0.1)
    assert sched.pending_train_request is None, "应当是当场回填"
    raw = hook.extra_state()
    assert raw["step_in_unit"] == 0 and raw["unit_steps"] == 0
    assert raw["at_boundary"] is True

    hook2, sched2, sampler2 = _hook(cfg)
    hook2.load_extra_state(raw)              # 干净边界 ⇒ 不报错
    assert sched2.pending_train_request is None
    assert sched2.state.units_run == sched.state.units_run
    assert sched2.state.global_samples_seen == sched.state.global_samples_seen


def test_resume_at_unit_start_republishes_request():
    """存档落在「unit 已开、0 步未跑」⇒ 模型未被更新 ⇒ 安全地重发 request。"""
    cfg = _cfg()
    n = cfg.auto_learning.eval_interval_steps
    bs = cfg.auto_learning.batch_size
    hook, sched, sampler = _hook(cfg)

    hook.on_step_begin(0)
    it = iter(sampler)
    for _ in range(n):
        [next(it) for _ in range(bs)]
        hook.on_step_end(0, loss=0.1)
    hook.on_step_begin(n)                       # 跨边界 ⇒ 开下一个 unit（0 步）
    raw = hook.extra_state()
    assert raw["step_in_unit"] == 0 and raw["unit_steps"] == n

    hook2, sched2, sampler2 = _hook(cfg)
    hook2.load_extra_state(raw)
    assert sched2.pending_train_request is not None, \
        "unit 已发布但 0 步未跑 ⇒ 恢复时必须重发 request"
    iter(sampler2)
    assert hook2.on_step_begin(0).rebuild_iterator is False


def test_resume_with_no_unit_in_flight_is_a_noop():
    """`_unit_steps == 0`（没有 unit 在飞）⇒ 恢复时不该乱发 request。"""
    cfg = _cfg()
    hook, sched, sampler = _hook(cfg)
    raw = hook.extra_state()
    assert raw["unit_steps"] == 0 and raw["step_in_unit"] == 0
    assert raw["at_boundary"] is True

    hook2, sched2, sampler2 = _hook(cfg)
    hook2.load_extra_state(raw)
    assert sched2.pending_train_request is None, "没有在飞的 unit ⇒ 不该凭空发 request"
    d = hook2.on_step_begin(0)                  # prime 负责开第一个 unit
    assert d.rebuild_iterator is True
    iter(sampler2)


def test_flush_partial_unit_accounts_actual_steps():
    """训练收尾：没跑满的 unit 按**实际步数**记账 ⇒ Scheduler 与模型对齐。"""
    cfg = _cfg()
    n = cfg.auto_learning.eval_interval_steps
    bs = cfg.auto_learning.batch_size
    hook, sched, sampler = _hook(cfg)

    hook.on_step_begin(0)
    it = iter(sampler)
    k = n - 2
    for _ in range(k):
        [next(it) for _ in range(bs)]
        hook.on_step_end(0, loss=0.1)
    assert sched.pending_train_request is not None
    assert hook.at_safe_checkpoint_boundary is False

    ev = hook.flush_partial_unit(global_step=k)
    assert ev is not None
    assert sched.state.global_step == k, "应当按实际步数 k 记账（不是整个 unit）"
    assert sched.state.global_samples_seen == k * bs
    assert hook.at_safe_checkpoint_boundary is True


def test_flush_partial_unit_cancels_zero_step_unit():
    """unit 刚发布、0 步未跑 ⇒ 收尾时**撤回**，不改账（模型没动）。"""
    cfg = _cfg()
    hook, sched, sampler = _hook(cfg)
    hook.on_step_begin(0)                       # 发布 unit，但一步都没跑
    assert sched.pending_train_request is not None
    before = sched.state.global_step

    assert hook.flush_partial_unit(0) is None
    assert sched.pending_train_request is None
    assert sched.state.global_step == before
    assert hook.at_safe_checkpoint_boundary is True


def test_boundary_resume_equals_continuous_run():
    """**无卡 deterministic resume**：连续跑 4 个 unit ≡ 跑 2 个 → 存 → 恢复 → 再跑 2 个。

    逐项比较 Scheduler state / registry / sample-id 序列 / global_step / samples_seen。
    （review v0.2 #2 的必加测试之一。）
    """
    cfg = _cfg()
    n = cfg.auto_learning.eval_interval_steps
    bs = cfg.auto_learning.batch_size

    def _run_units(hook, sampler, n_units, topup=0):
        """跑 n_units 个完整 unit；`topup` 用于恢复路径补跑最初那几步。"""
        ids: list = []
        it = iter(sampler)
        for _ in range(n_units):
            d = hook.on_step_begin(0)
            if d.finished:
                break
            if d.rebuild_iterator:
                it = iter(sampler)
            for _ in range(n):
                ids.extend(next(it) for _ in range(bs))
                hook.on_step_end(0, loss=0.1)
        return ids

    # A：连续 4 个 unit
    hookA, schedA, sampA = _hook(cfg)
    idsA = _run_units(hookA, sampA, 4)

    # B：2 个 unit → 存档 → 新 hook 恢复 → 再 2 个 unit
    hookB, schedB, sampB = _hook(cfg)
    idsB = _run_units(hookB, sampB, 2)
    raw = hookB.extra_state()
    assert raw["at_boundary"] is True

    hookB2, schedB2, sampB2 = _hook(cfg)
    hookB2.load_extra_state(raw)
    idsB += _run_units(hookB2, sampB2, 2)

    assert idsA == idsB, "boundary resume 的 sample-id 序列必须与连续运行完全一致"
    assert schedA.state.to_state() == schedB2.state.to_state(), "Scheduler state 必须一致"
    assert schedA.registry.to_state() == schedB2.registry.to_state(), "Registry 必须一致"
    assert schedA.rng.getstate() == schedB2.rng.getstate(), "RNG 状态必须一致"


def test_unique_batches_counts_batch_compositions_not_losses():
    """🔴 review v0.2 #9：unique_batches 必须按 (new ids, old ids) 去重。"""
    cfg = _cfg()
    n = cfg.auto_learning.eval_interval_steps
    bs = cfg.auto_learning.batch_size
    hook, sched, sampler = _hook(cfg)

    hook.on_step_begin(0)
    it = iter(sampler)
    for _ in range(n):
        [next(it) for _ in range(bs)]
        hook.on_step_end(0, loss=0.1)        # 所有 step 的 loss 完全相同
    stats = sampler.stats_upto(n)
    assert stats.unique_batches == n, "每步都是不同的 batch 组成 ⇒ unique_batches 应等于步数"
    # 反证：按 loss 去重会得到 1
    assert len({round(x, 12) for x in [0.1] * n}) == 1


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
