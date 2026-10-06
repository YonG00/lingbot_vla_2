"""测试方案 §11 Level H —— 日志与可解释性。

自动学习最大的风险之一是：系统跑了几个小时，最后不知道为什么这么跑。
所以**日志本身也要被测**。
"""

from __future__ import annotations

import json
import os

import pytest

from lingbotvla.auto_learning.obs.logger import EventLogger
from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
from al_fixtures import local_tmpdir
from lingbotvla.auto_learning.testing.fake_tasks import cfg_with
from lingbotvla.auto_learning.types import ReasonCode, TaskStatus
from al_fixtures import scheduler_of

TRANSITION_KINDS = {"PASS", "DEFER", "DEFER_OVERFIT", "EXHAUSTED", "REOPEN", "REOPEN_EXHAUSTED"}

VALID_CODES = {c.value for c in ReasonCode}


def _run(names=("easy_pass", "unlearnable", "already_known", "slow_learn"), **kw):
    cfg = cfg_with(list(names), **kw)
    sched = scheduler_of(cfg)
    sched.run(max_actions=800)
    return cfg, sched


# =========================================================================== #
# H01 每次状态迁移必须有 reason（人话）+ code（机器码）
# =========================================================================== #
def test_H01_every_transition_has_reason_and_code():
    _, sched = _run()
    assert sched.state.transitions
    for row in sched.state.transitions:
        assert row["kind"] in TRANSITION_KINDS, row
        assert row["reason"], f"缺 reason：{row}"
        assert row["reason"] != "status changed", f"reason 太笼统：{row}"
        assert row["code"], f"缺机器可读 code：{row}"
        assert row["code"] in VALID_CODES, row


def test_H01_registry_keeps_last_reason():
    _, sched = _run()
    for rec in sched.registry:
        assert rec.last_transition_reason, rec.task_name
        assert rec.last_transition_code in VALID_CODES, rec.task_name


# =========================================================================== #
# H02 Event Trace 能重建学习顺序
# =========================================================================== #
def test_H02_event_trace_rebuilds_learning_order():
    with local_tmpdir() as d:
        cfg = cfg_with(["easy_pass", "unlearnable", "slow_learn"], seed=3)
        logger = EventLogger(d)
        sched = scheduler_of(cfg, logger=logger)
        sched.run(max_actions=800)
        logger.close()

        events = [json.loads(line) for line in open(os.path.join(d, "events.jsonl"))]
        selected = [e["task"] for e in events if e.get("action") == "select"]
        assert selected, "应当有 select 事件"
        # 顺序必须与 scheduler 内部记录一致
        assert selected == [t["task"] for t in sched.state.transitions]

        # 每条 transition 都能在事件流里找到对应的 unit / review
        unit_steps = {e["step"] for e in events if e.get("action") == "train_unit"}
        assert unit_steps, "应当有 train_unit 事件"
        for row in sched.state.transitions:
            assert row["step"] in unit_steps or row["kind"].startswith("REOPEN")

        # 从 trace 能回答「为什么选它」
        first = next(e for e in events if e.get("action") == "select")
        assert first["scout_nmse"] is not None
        assert "hardness_version" in first


def test_H02_event_jsonl_is_valid_and_ordered():
    with local_tmpdir() as d:
        logger = EventLogger(d)
        sched = scheduler_of(cfg_with(["easy_pass", "unlearnable"]), logger=logger)
        sched.run(max_actions=400)
        logger.close()
        rows = [json.loads(line) for line in open(os.path.join(d, "events.jsonl"))]
        assert rows[-1]["action"] == "finish"
        assert all(isinstance(r, dict) and "action" in r for r in rows)


# =========================================================================== #
# H03 Skill Overview 一致性
# =========================================================================== #
def test_H03_counts_sum_to_total_and_coverage_matches_registry():
    _, sched = _run(names=("easy_pass", "unlearnable", "already_known", "slow_learn"))
    counts = sched.registry.counts()
    assert sum(counts.values()) == len(sched.registry)

    summary = sched.registry.summary()
    assert summary["coverage"] == len(sched.registry.by_status(TaskStatus.PASS)) / len(sched.registry)
    assert summary["pass"] == counts[TaskStatus.PASS.value]

    # 每个 unit 行里的 pass_count / coverage 必须自洽
    n = len(sched.registry)
    for row in sched.metrics_rows:
        assert abs(row["coverage"] - row["pass_count"] / n) < 1e-9, row
        assert 0.0 <= row["coverage"] <= 1.0


def test_H03_tb_scalars_use_the_documented_tag_names():
    with local_tmpdir() as d:
        logger = EventLogger(d)
        sched = scheduler_of(cfg_with(["easy_pass", "unlearnable"]), logger=logger)
        sched.run(max_actions=400)
        logger.close()

        tags = {json.loads(l)["tag"] for l in open(os.path.join(d, "tb_scalars.jsonl"))}
        for prefix in (
            "training/",
            "current_skill/",
            "skill_overview/",
            "memory/",
            "system/",
            "task/",
            "debug/",
        ):
            assert any(t.startswith(prefix) for t in tags), prefix
        assert "skill_overview/coverage" in tags
        assert "current_skill/val_nmse" in tags
        assert "memory/pass_pool_size" in tags


# =========================================================================== #
# H04 Heatmap 数据完整
# =========================================================================== #
def test_H04_heatmap_rows_are_complete_and_monotone():
    _, sched = _run()
    rows = sched.heatmap_rows
    assert rows, "应当有 heatmap 数据"

    events = [r["event"] for r in rows]
    assert events == sorted(events), "event 序号必须单调递增"
    assert events[0] == 1 and events[-1] == sched.state.eval_events

    for r in rows:
        assert {"event", "step", "task", "kind", "nmse"} <= set(r)
        assert r["task"] in sched.registry.names()
        assert r["nmse"] is None or isinstance(r["nmse"], float)

    # 每个任务至少被评估过一次
    seen = {r["task"] for r in rows}
    assert seen == set(sched.registry.names())


def test_H04_heatmap_needs_no_tensorboard():
    """方案要求 heatmap 不必从 TB 人工拼 —— 直接从 heatmap.csv 就能画。"""
    from lingbotvla.auto_learning.obs import report

    _, sched = _run()
    tasks, labels, matrix = report.build_heatmap_matrix(
        sched.heatmap_rows, sched.registry.names()
    )
    assert len(tasks) == len(sched.registry)
    assert len(labels) == len(matrix)
    assert all(len(row) == len(tasks) for row in matrix)
    assert any(v is not None for row in matrix for v in row)


def test_H02_resume_appends_instead_of_truncating_event_log():
    """resume 时**不能**把上一次 run 的 events.jsonl 截断（`compare_resume_run` 抓出来的）。"""
    from lingbotvla.auto_learning.config import AutoLearningConfig
    from lingbotvla.auto_learning.testing.fake_tasks import cfg_with as make_fake
    from lingbotvla.auto_learning.state import persistence

    with local_tmpdir() as d:
        cfg = make_fake(["easy_pass", "unlearnable"],
                        al=AutoLearningConfig(seed=3, max_transitions=1))
        logger = EventLogger(d)
        sched = scheduler_of(cfg, logger=logger)
        sched.run(max_actions=200)
        logger.close()
        persistence.save_state(sched, os.path.join(d, "state.json"))
        first = [json.loads(l) for l in open(os.path.join(d, "events.jsonl"))]
        assert first and first[-1]["action"] == "finish"

        # 恢复：换一个 logger（append=True），同一个目录
        cfg2 = make_fake(["easy_pass", "unlearnable"],
                         al=AutoLearningConfig(seed=3, max_transitions=1))
        logger2 = EventLogger(d, append=True)
        sched2 = scheduler_of(cfg2, logger=logger2)
        persistence.load_state(sched2, os.path.join(d, "state.json"))
        sched2.resume()
        sched2.al.max_transitions = None
        sched2.run(max_actions=200)
        logger2.close()

        second = [json.loads(l) for l in open(os.path.join(d, "events.jsonl"))]
        assert len(second) > len(first), "resume 不能把历史事件截断"
        assert second[: len(first)] == first, "历史事件必须原样保留在文件前面"
        # 内存里的事件流也要能跨 resume 保留（否则 trace 只能重建后半段）
        assert len(sched2.events) >= len(first)


def test_H02_in_memory_events_survive_state_roundtrip():
    from lingbotvla.auto_learning.config import AutoLearningConfig
    from lingbotvla.auto_learning.testing.fake_tasks import cfg_with as make_fake
    from lingbotvla.auto_learning.state import persistence

    with local_tmpdir() as d:
        cfg = make_fake(["easy_pass", "unlearnable"],
                        al=AutoLearningConfig(seed=3))
        sched = scheduler_of(cfg)
        sched.run(max_actions=200)
        path = os.path.join(d, "state.json")
        persistence.save_state(sched, path)

        fresh = scheduler_of(cfg)
        persistence.load_state(fresh, path)
        assert [e.get("action") for e in fresh.events] == [e.get("action") for e in sched.events]
        assert [e.get("task") for e in fresh.events if e.get("action") == "select"] == [
            e.get("task") for e in sched.events if e.get("action") == "select"
        ]



def test_tensorboard_event_file_is_written_and_readable():
    """run 应当直接产出**真的** TensorBoard event 文件（不是只有 JSONL）。"""
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from lingbotvla.auto_learning.config import AutoLearningConfig
    from lingbotvla.auto_learning.testing.fake_tasks import cfg_with as make_fake

    with local_tmpdir() as d:
        cfg = make_fake(["easy_pass", "unlearnable"], al=AutoLearningConfig(seed=3))
        logger = EventLogger(d)
        sched = scheduler_of(cfg, logger=logger)
        sched.run(max_actions=200)
        logger.close()

        assert logger.tb.ok, f"没写出 event 文件（backend={logger.tb.backend}）"
        ea = EventAccumulator(logger.tb_dir)
        ea.Reload()
        tags = set(ea.Tags()["scalars"])
        for want in ("training/loss", "current_skill/val_nmse", "skill_overview/coverage"):
            assert want in tags, f"TB 里缺 tag {want}"
        # 与 JSONL 里的点数一致
        n_jsonl = sum(1 for _ in open(os.path.join(d, "tb_scalars.jsonl")))
        n_tb = sum(len(ea.Scalars(t)) for t in tags)
        assert n_tb == n_jsonl


def test_tb_from_jsonl_converter_roundtrip():
    """早先的 run 只有 tb_scalars.jsonl，转换脚本要能补齐 event 文件。"""
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from lingbotvla.auto_learning.config import AutoLearningConfig
    from lingbotvla.auto_learning.testing.fake_tasks import cfg_with as make_fake
    from lingbotvla.auto_learning.tools.tb_from_jsonl import convert

    with local_tmpdir() as d:
        cfg = make_fake(["easy_pass"], al=AutoLearningConfig(seed=3))
        logger = EventLogger(d, tensorboard=False)  # 模拟「没有 tensorboard」的旧 run
        sched = scheduler_of(cfg, logger=logger)
        sched.run(max_actions=100)
        logger.close()
        assert not os.path.exists(os.path.join(d, "tb"))

        out, n, per_tag = convert(d)
        assert n > 0 and out == os.path.join(d, "tb")
        ea = EventAccumulator(out)
        ea.Reload()
        assert "training/loss" in set(ea.Tags()["scalars"])
