"""Regression checks for real training / Auto Learning TensorBoard integration.

CPU-only: no GPU, checkpoint or trained model needed.
"""
from __future__ import annotations

import json

import pytest

from lingbotvla.auto_learning.real.build import SchedulerLoggerAdapter
from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.decision.thresholds import PassThresholds
from lingbotvla.auto_learning.testing.fake_tasks import cfg_with
from al_fixtures import scheduler_of


class RepoLog:
    def info_rank0(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class Writer:
    def __init__(self):
        self.points = []

    def add_scalar(self, name, value, step):
        self.points.append((name, value, step))


def test_starting_from_step500_uses_absolute_tb_axis_and_distinct_loss(tmp_path):
    writer = Writer()
    event_path = tmp_path / "auto_learning_events.jsonl"
    logger = SchedulerLoggerAdapter(RepoLog(), writer=writer, event_path=str(event_path))
    logger.set_tb_step_offset(train_global_step=500, al_global_step=0)
    logger.log_metrics(0, "debug/click_bell/scout_mse", 0.03)
    logger.log_metrics(5, "training/loss", 0.34)
    logger.log_metrics(5, "current_skill/val_nmse", 0.21)
    logger.log_event({"action": "train_unit", "step": 5})
    assert writer.points == [
        ("debug/click_bell/scout_mse", 0.03, 500),
        ("auto_learning/unit_loss", 0.34, 505),
        ("current_skill/val_nmse", 0.21, 505),
    ]
    assert not any(x[0] == "training/loss" for x in writer.points)
    rows = [json.loads(x) for x in event_path.read_text(encoding="utf-8").splitlines()]
    assert [(x["step"], x["tb_step"]) for x in rows] == [
        (0, 500), (5, 505), (5, 505), (5, 505),
    ]


@pytest.mark.parametrize("trainer_step,al_step,expected", [
    (0, 0, 10), (500, 0, 510), (510, 10, 510), (700, 200, 510),
])
def test_tb_alignment_is_computed_from_actual_resume_state(trainer_step, al_step, expected):
    writer = Writer()
    logger = SchedulerLoggerAdapter(RepoLog(), writer=writer)
    logger.set_tb_step_offset(train_global_step=trainer_step, al_global_step=al_step)
    logger.log_metrics(10, "current_skill/val_nmse", 0.8)
    assert writer.points == [("current_skill/val_nmse", 0.8, expected)]


def test_stage_a_relative_scheduler_axis_unchanged_and_logs_pass_threshold():
    cfg = AutoLearningConfig(seed=3, eval_interval_steps=5, min_steps_before_defer=10,
                             pass_metric="mse", pass_thresholds_file="test_only")
    cfg.pass_thresholds = PassThresholds(tasks={"easy_pass": 0.10,
                                                 "unlearnable": 0.10})
    writer = Writer()
    logger = SchedulerLoggerAdapter(RepoLog(), writer=writer)
    logger.set_tb_step_offset(train_global_step=500, al_global_step=0)
    sim = cfg_with(["easy_pass", "unlearnable"], al=cfg)
    sched = scheduler_of(sim, logger=logger)
    sched.run(max_actions=40)
    metrics = {(tag, step): val for tag, val, step in writer.points}
    # The tag axes reflect the actual run's wall-clock training steps.
    assert any(tag == "current_skill/pass_threshold_mse" and step >= 505
               for tag, _val, step in writer.points)
    assert any(tag == "current_skill/val_to_pass_threshold" and step >= 505
               for tag, _val, step in writer.points)
    assert any(tag.startswith("task/") and tag.endswith("/val_to_pass_threshold")
               for tag, _val, step in writer.points)
    assert ("auto_learning/unit_loss", 505) in metrics
    assert ("training/loss", 505) not in metrics
