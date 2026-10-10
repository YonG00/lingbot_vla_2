"""CPU 层：**任务切换触发的真重扫（rescan）** —— 计数、事件流、以及「不吃 bootstrap 缓存」。

为什么单独加这一条（2026-10-10）
--------------------------------
已有的 `tests/test_gmean_pass_pipeline.py` / `tests/test_gmean_ratio_priority.py` 直接调
`sched._rescan()` 验证重扫结果；`tests/test_al_openloop_update*.py` 用 Mock 顶掉 `_rescan`
验证**节奏**（`task_switch_count` / `full_rescan_count`）。

缺的是把两半接起来的**触发路径**证明：真实任务切换 → 真的走了一遍评测 →
① 计数递增；② 评测事件流里出现 `kind="rescan"`；③ 重扫**没有**走 bootstrap scout 缓存
（`_timed_eval` 只在 `bootstrap_cache=True 且 global_step==0` 时才用缓存）。

这也是运维判据的 CPU 版：真机上看 `auto_learning_events.jsonl` 里
`{"kind":"metric","name":"task/<任务>/rescan_nmse"}`（由 `SchedulerLoggerAdapter.log_metrics`
落盘）与日志里的 `full_rescan_count`。
"""
from __future__ import annotations

from al_fixtures import make_cfg, scheduler_of
from lingbotvla.auto_learning.config import AutoLearningConfig


class _RecordingLogger:
    """记录 `log_event` / `log_metrics`，模拟 `SchedulerLoggerAdapter` 的调用面。"""

    def __init__(self) -> None:
        self.events = []
        self.metrics = []
        self.texts = []

    # Scheduler 只要求这三个
    def log_event(self, event):
        self.events.append(dict(event or {}))

    def log_metrics(self, step, name, value):
        self.metrics.append((int(step), str(name), value))

    def log_text(self, step, name, value):
        self.texts.append((int(step), str(name), str(value)))

    # 转发接口（Scheduler 可能用来打日志）
    def info_rank0(self, *a, **k):
        pass

    info = warning = info_rank0

    def metric_names(self):
        return [name for _, name, _ in self.metrics]


def _spy_evaluator(sched):
    """把 evaluate / evaluate_bootstrap_scout 都记一笔，返回 calls 列表。

    真实 `RealEvaluator.evaluate_bootstrap_scout` 会先查 scout 缓存；这里用「哪个入口被调用」
    证明重扫确实绕过了缓存入口（而不是靠读日志猜）。
    """
    calls = []
    inner_evaluate = sched.evaluator.evaluate

    def _evaluate(task, split, episode_ids):
        calls.append(('evaluate', task, split))
        return inner_evaluate(task, split, episode_ids)

    def _bootstrap_cache(task, split, episode_ids):
        calls.append(('bootstrap_cache', task, split))
        return inner_evaluate(task, split, episode_ids)

    sched.evaluator.evaluate = _evaluate
    sched.evaluator.evaluate_bootstrap_scout = _bootstrap_cache
    return calls


def _bootstrap_all(sched):
    seen = 0
    while sched.next_action() == 'bootstrap':
        sched.advance()
        seen += 1
    return seen


def test_task_switch_triggers_real_rescan_with_rescan_event_rows():
    logger = _RecordingLogger()
    cfg = make_cfg(3, al=AutoLearningConfig(seed=3, rescan_every_n_task_switches=1,
                                           scout_confirm_enabled=False))
    sched = scheduler_of(cfg, logger=logger)
    calls = _spy_evaluator(sched)

    boot = _bootstrap_all(sched)
    assert boot == 3, 'bootstrap 应该把 3 个任务都扫一遍'
    assert any(kind == 'bootstrap_cache' for kind, *_ in calls), \
        'bootstrap（global_step==0）必须走缓存入口'

    # ---- 触发一次真实的「任务切换」（与 scheduler._after_transition 的调用面一致）----
    calls.clear()
    logger.metrics.clear()
    eval_events_before = sched.state.eval_events
    sched._after_transition('t0', 'DEFER', 'cpu-test')

    assert sched.state.task_switch_count == 1
    assert sched.state.full_rescan_count == 1
    assert sched.state.eval_events > eval_events_before, '重扫必须真的产生评测事件'

    # ① 重扫**不吃** bootstrap 缓存：全部走 evaluate
    assert not any(kind == 'bootstrap_cache' for kind, *_ in calls), \
        f'重扫不该进 bootstrap 缓存入口，实际调用：{calls}'
    assert any(kind == 'evaluate' and split == 'scout' for kind, _, split in calls), \
        f'重扫必须以 scout 口径真评测，实际调用：{calls}'

    # ② 评测事件流里出现 kind="rescan"
    rescanned = [row for row in sched.heatmap_rows if row.get('kind') == 'rescan']
    assert rescanned, 'heatmap_rows（评测事件流，会随 AL 状态一起持久化）里应有 kind=rescan'
    assert {row['task'] for row in rescanned} <= {'t0', 't1', 't2'}
    for rec in sched.registry:
        rows = [row for row in rec.eval_history if row.get('kind') == 'rescan']
        if rec.task_name in {row['task'] for row in rescanned}:
            assert rows, f'{rec.task_name} 的 eval_history 里应有 kind=rescan 记录'
            assert rows[-1]['split'] == 'scout'

    # ③ 落盘形态：JSONL/TB 的 metric 名是 task/<任务>/rescan_nmse
    rescan_metric_names = [n for n in logger.metric_names() if n.endswith('/rescan_nmse')]
    assert rescan_metric_names, f'应写 task/<任务>/rescan_nmse 指标，实际：{logger.metric_names()}'


def test_rescan_cadence_can_be_widened_and_cache_stays_untouched():
    """`rescan_every_n_task_switches=3`：前两次切换不重扫，也不碰 bootstrap 缓存。"""
    logger = _RecordingLogger()
    cfg = make_cfg(3, al=AutoLearningConfig(seed=3, rescan_every_n_task_switches=3,
                                           scout_confirm_enabled=False))
    sched = scheduler_of(cfg, logger=logger)
    calls = _spy_evaluator(sched)
    _bootstrap_all(sched)
    calls.clear()

    for _ in range(2):
        sched._after_transition('t0', 'DEFER', 'cpu-test')
    assert sched.state.task_switch_count == 2
    assert sched.state.full_rescan_count == 0
    assert calls == [], f'未到节奏点不该有评测调用：{calls}'

    sched._after_transition('t1', 'DEFER', 'cpu-test')
    assert sched.state.task_switch_count == 3
    assert sched.state.full_rescan_count == 1
    assert not any(kind == 'bootstrap_cache' for kind, *_ in calls)
    assert any(row.get('kind') == 'rescan' for row in sched.heatmap_rows)
