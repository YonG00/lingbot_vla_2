"""扫描模式（`AL_SCAN_SKIP_HARDNESS=1`）的契约测试 —— 2026-10-11 用户要求。

背景
----
launcher 的启动门控**只看 scout 覆盖**（`check_coverage` 检查的是 scout 缓存文件）。
但预扫 worker 用的推进循环是「一直 advance 直到 `train_unit`」，于是 bootstrap 队列跑空后
它还会继续走进 `_select()` —— 那里会给挑中的任务算 **hardness**（单卡实测 15–20 分钟/任务，
占预扫耗时 99%，而门控根本不看它）。实测：6 片并行、50 任务 ⇒ 2–3 小时。

修法：扫描 worker 带 `AL_SCAN_SKIP_HARDNESS=1` ⇒ **bootstrap 队列一空就收工**，
hardness 回归训练循环按需扫（训练是 6 卡同进程组，`AL_HARDNESS_SHARD=1` 才真正生效）。

本文件锁定三件事：
  1. 带开关 ⇒ bootstrap 排空后**不进入 select/训练**，直接收工（返回 None + finished）；
  2. 不带开关 ⇒ 行为**不变**（继续推进到 train_unit）；
  3. launcher：开关**只注入扫描 worker**，训练环境里必须显式为 `'0'`（防父进程 AL_* 透传）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import pytest

from lingbotvla.auto_learning.real.hook import (
    SCAN_SKIP_HARDNESS_ENV,
    AutoLearnLoopHook,
)

REPO = Path(__file__).resolve().parents[1]


class _FakeState:
    def __init__(self, bootstrap_queue: List[str]) -> None:
        self.bootstrap_queue = list(bootstrap_queue)
        self.finished = False
        self.stop_reason = ''


class _FakeScheduler:
    """最小假 scheduler：复刻 `next_action()` 的判据与 `_bootstrap_one()` 的副作用。"""

    def __init__(self, tasks: List[str]) -> None:
        self.state = _FakeState(tasks)
        self.defer_train = False
        self.selected: List[str] = []      # 记录是否走到 select（即会不会算 hardness）
        self.trained: List[str] = []
        self.pending_train_request = None
        self.pending_train_steps = 0

    def next_action(self) -> str:
        if self.state.finished:
            return 'finished'
        if self.state.bootstrap_queue:
            return 'bootstrap'
        if not self.selected:
            return 'select'
        return 'train_unit'

    def advance(self) -> Dict[str, Any]:
        action = self.next_action()
        if action == 'bootstrap':
            name = self.state.bootstrap_queue.pop(0)
            return {'action': 'bootstrap', 'task': name, 'result': 'candidate'}
        if action == 'select':
            self.selected.append('picked')
            return {'action': 'select', 'task': 'picked'}
        if action == 'train_unit':
            self.trained.append('unit')
            self.pending_train_request = object()
            self.pending_train_steps = 1
            return {'action': 'train_unit'}
        return {'action': 'finish', 'stop_reason': self.state.stop_reason}

    def complete_train_unit(self, result: Any, allow_partial: bool = False) -> Dict[str, Any]:
        return {'action': 'train_result'}

    def _finish(self, reason: str) -> None:
        self.state.finished = True
        self.state.stop_reason = reason


class _FakeSampler:
    """`_open_next_unit()` 会调 `sampler.set_request(req)`（重建迭代器前必须完成）。"""

    def __init__(self) -> None:
        self.requests: List[Any] = []

    def set_request(self, req: Any) -> None:
        self.requests.append(req)


def _hook(tasks: List[str]) -> tuple[AutoLearnLoopHook, _FakeScheduler]:
    sched = _FakeScheduler(tasks)
    hook = AutoLearnLoopHook(scheduler=sched, sampler=_FakeSampler(), cfg=object())
    return hook, sched


# --------------------------------------------------------------------------- #
# 1) 带开关：bootstrap 排空即收工（不算 hardness）
# --------------------------------------------------------------------------- #
def test_scan_mode_stops_after_bootstrap(monkeypatch):
    monkeypatch.setenv(SCAN_SKIP_HARDNESS_ENV, '1')
    hook, sched = _hook(['t1', 't2', 't3'])
    assert hook._open_next_unit() is None, '扫描模式必须返回 None（不发布 train_unit）'
    assert sched.state.finished, '必须把 scheduler 标成 finished'
    assert sched.state.stop_reason == 'scan_only_scout_done'
    assert sched.state.bootstrap_queue == [], 'bootstrap 队列应已排空（scout 全做完）'
    assert sched.selected == [], '**不得进入 _select**（那一步会算 hardness）'
    assert sched.trained == [], '不得进入训练'
    # 事件流可审计
    actions = [e.get('action') for e in hook._events]
    assert actions.count('bootstrap') == 3 and actions[-1] == 'finish'


def test_scan_mode_does_not_stop_mid_bootstrap(monkeypatch):
    """队列没排空前不得提前收工（scout 必须扫完，否则门控过不了）。"""
    monkeypatch.setenv(SCAN_SKIP_HARDNESS_ENV, '1')
    hook, sched = _hook(['only_one'])
    assert hook._open_next_unit() is None
    assert sched.state.bootstrap_queue == []
    assert [e.get('action') for e in hook._events].count('bootstrap') == 1


# --------------------------------------------------------------------------- #
# 2) 不带开关：行为不变（继续走到 select / train_unit）
# --------------------------------------------------------------------------- #
def test_without_scan_mode_behaviour_unchanged(monkeypatch):
    monkeypatch.delenv(SCAN_SKIP_HARDNESS_ENV, raising=False)
    hook, sched = _hook(['t1'])
    req = hook._open_next_unit()
    assert req is not None, '训练模式必须发布 train_unit 的 request'
    assert sched.selected == ['picked'], '训练模式会走到 select（并算 hardness）'
    assert sched.trained == ['unit']
    assert not sched.state.finished


@pytest.mark.parametrize('value', ['0', '', 'no', 'false'])
def test_scan_mode_off_values(monkeypatch, value):
    monkeypatch.setenv(SCAN_SKIP_HARDNESS_ENV, value)
    hook, sched = _hook(['t1'])
    assert hook._open_next_unit() is not None
    assert sched.selected == ['picked']


# --------------------------------------------------------------------------- #
# 3) launcher：开关只给扫描 worker，训练环境显式 '0'
# --------------------------------------------------------------------------- #
def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        '_al_launch_scan_probe', REPO / 'experiment/robotwin/al_launch.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launcher_injects_flag_only_for_workers():
    al = _load_launcher()
    ckpt = '/models/x/train_al_v37/checkpoints/global_step_500/hf_ckpt'
    args = al.build_parser().parse_args([
        '--checkpoint', ckpt, '--scout-cache-file', '/tmp/scout.json', '--min-free-gb', '0'])
    common = al.base_env(args=args, repo=Path('/tmp/repo'), python=Path('/tmp/py'),
                         al_cfg=Path('/tmp/al.yaml'), train_config=Path('/tmp/c.yaml'),
                         split_dir=Path('/tmp/split'), phases=Path('/tmp/phases'),
                         checkpoint=Path(ckpt), manifest=Path('/tmp/m.json'),
                         baseline=Path('/tmp/b.json'), norm=Path('/tmp/n.json'),
                         dtype='bfloat16')
    # 训练侧：显式 '0'（即使父进程 export 了 AL_SCAN_SKIP_HARDNESS 也不受影响）
    assert common['AL_SCAN_SKIP_HARDNESS'] == '0', (
        '训练环境必须显式关闭扫描模式，否则训练会跳过 hardness')
    # worker 侧：launcher 在 `_scan_round()` 里覆盖为 '1'
    werker = al.worker_env(common, gpu='0', train_out=Path('/tmp/out'), max_steps=1,
                           master_port=62510, smoke_no_checkpoint=True)
    text = (REPO / 'experiment/robotwin/al_launch.py').read_text(encoding='utf-8')
    assert "env['AL_SCAN_SKIP_HARDNESS'] = '1'" in text, (
        '扫描 worker 必须被注入 AL_SCAN_SKIP_HARDNESS=1')


def test_parent_env_cannot_leak_scan_mode_into_training(monkeypatch):
    """父进程误带该变量时，训练环境仍必须是 '0'（base_env 白名单显式覆盖）。"""
    monkeypatch.setenv(SCAN_SKIP_HARDNESS_ENV, '1')
    al = _load_launcher()
    ckpt = '/models/x/hf_ckpt'
    args = al.build_parser().parse_args(['--checkpoint', ckpt, '--min-free-gb', '0'])
    common = al.base_env(args=args, repo=Path('/tmp/repo'), python=Path('/tmp/py'),
                         al_cfg=Path('/tmp/al.yaml'), train_config=Path('/tmp/c.yaml'),
                         split_dir=Path('/tmp/split'), phases=Path('/tmp/phases'),
                         checkpoint=Path(ckpt), manifest=Path('/tmp/m.json'),
                         baseline=Path('/tmp/b.json'), norm=Path('/tmp/n.json'),
                         dtype='bfloat16')
    assert common['AL_SCAN_SKIP_HARDNESS'] == '0'
