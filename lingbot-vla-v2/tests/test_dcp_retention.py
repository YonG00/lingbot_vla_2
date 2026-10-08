"""`dcp_retention` 的 CPU 专项测试（全部用临时目录，不需要 torch / GPU）。

覆盖用户批准的规则：默认保留最近 2 份完整 DCP、先验证新 DCP 再删旧、保存失败/磁盘不足
时一个字节都不删、未完成目录既不删也不计数、HF 里程碑树永不受影响、最终保存后同样清理、
清理日志含保留/删除的 step、Smoke 无存档模式不启用。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lingbotvla.utils.dcp_retention import (
    is_complete_dcp,
    list_dcps,
    plan_retention,
    prune_dcps,
    root_is_forbidden,
    should_prune,
)


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
def make_dcp(root: Path, step: int, *, complete: bool = True, files: int = 2) -> Path:
    """造一个（完整或残缺的）DCP 目录。"""
    d = root / f"global_step_{step}"
    (d / "model").mkdir(parents=True, exist_ok=True)
    (d / "optimizer").mkdir(parents=True, exist_ok=True)
    (d / "extra_state").mkdir(parents=True, exist_ok=True)
    for i in range(files):
        (d / "model" / f"__0_{i}.distcp").write_bytes(b"m" * 32)
        (d / "optimizer" / f"__0_{i}.distcp").write_bytes(b"o" * 32)
    (d / "extra_state" / "extra_state_rank_0.pt").write_bytes(b"e")
    if complete:
        (d / "model" / ".metadata").write_bytes(b"{}")
        (d / "optimizer" / ".metadata").write_bytes(b"{}")
    return d


def steps_of(pairs) -> list:
    return [s for s, _ in pairs]


class RecLogger:
    def __init__(self):
        self.lines = []

    def info_rank0(self, msg):
        self.lines.append(str(msg))

    def warning(self, msg):
        self.lines.append("WARN " + str(msg))


# --------------------------------------------------------------------------- #
# 完整性判断
# --------------------------------------------------------------------------- #
def test_complete_and_incomplete_detection(tmp_path):
    good = make_dcp(tmp_path, 1000)
    bad_meta = make_dcp(tmp_path, 2000, complete=False)
    (tmp_path / "not_a_dcp").mkdir()
    assert is_complete_dcp(str(good)) is True
    assert is_complete_dcp(str(bad_meta)) is False, "缺 .metadata ⇒ 视为未完成"
    assert is_complete_dcp(str(tmp_path / "not_a_dcp")) is False
    assert list_dcps(str(tmp_path)) == [(1000, str(good)), (2000, str(bad_meta))]


def test_empty_distcp_dir_is_incomplete(tmp_path):
    d = make_dcp(tmp_path, 500)
    for f in (d / "model").glob("*.distcp"):
        f.unlink()
    assert is_complete_dcp(str(d)) is False


# --------------------------------------------------------------------------- #
# 保留决策
# --------------------------------------------------------------------------- #
def test_keep_latest_two_of_five(tmp_path):
    for s in (1000, 2000, 3000, 4000, 5000):
        make_dcp(tmp_path, s)
    plan = plan_retention(str(tmp_path), 2, newest_verified_step=5000)
    assert steps_of(plan["keep"]) == [4000, 5000]
    assert steps_of(plan["delete"]) == [1000, 2000, 3000]
    assert plan["guarded"] is False


def test_keep_last_one_and_zero(tmp_path):
    for s in (1000, 2000, 3000):
        make_dcp(tmp_path, s)
    p1 = plan_retention(str(tmp_path), 1, newest_verified_step=3000)
    assert steps_of(p1["keep"]) == [3000] and steps_of(p1["delete"]) == [1000, 2000]
    p0 = plan_retention(str(tmp_path), 0, newest_verified_step=3000)
    assert p0["guarded"] is True and p0["delete"] == [], "keep_last=0 ⇒ 关闭清理"


def test_no_deletion_when_newest_is_incomplete(tmp_path):
    """保存失败/崩溃残留：更新的未完成目录存在 ⇒ 一份都不删。"""
    for s in (1000, 2000, 3000):
        make_dcp(tmp_path, s)
    make_dcp(tmp_path, 3001, complete=False)  # 崩溃残留
    plan = plan_retention(str(tmp_path), 2, newest_verified_step=3000)
    assert plan["guarded"] is True
    assert plan["delete"] == [], "有更新的未完成目录 ⇒ 保守不清理"
    assert "未完成" in plan["reason"]
    assert steps_of(plan["incomplete"]) == [3001]


def test_no_deletion_when_save_failed_or_unverified(tmp_path):
    """磁盘不足/保存失败 ⇒ 调用方不传 verified step ⇒ 绝不删旧恢复点。"""
    for s in (1000, 2000, 3000):
        make_dcp(tmp_path, s)
    plan = plan_retention(str(tmp_path), 2, newest_verified_step=None)
    assert plan["guarded"] is True and plan["delete"] == []
    assert "保存失败" in plan["reason"] or "未校验" in plan["reason"]


def test_no_deletion_when_verified_step_is_not_newest(tmp_path):
    """已验证 step 与最新完整 DCP 不一致（例如校验的是旧的一份）⇒ 不清理。"""
    for s in (1000, 2000, 3000):
        make_dcp(tmp_path, s)
    plan = plan_retention(str(tmp_path), 2, newest_verified_step=2000)
    assert plan["guarded"] is True and plan["delete"] == []


def test_incomplete_never_counts_toward_keep(tmp_path):
    for s in (1000, 2000, 3000):
        make_dcp(tmp_path, s)
    make_dcp(tmp_path, 1500, complete=False)  # 夹在中间的残缺目录
    plan = plan_retention(str(tmp_path), 2, newest_verified_step=3000)
    assert steps_of(plan["keep"]) == [2000, 3000], "残缺目录不占保留名额"
    assert steps_of(plan["delete"]) == [1000]
    assert steps_of(plan["incomplete"]) == [1500]


# --------------------------------------------------------------------------- #
# 执行与保护
# --------------------------------------------------------------------------- #
def test_prune_executes_and_logs_steps(tmp_path):
    for s in (1000, 2000, 3000, 4000):
        make_dcp(tmp_path, s)
    log = RecLogger()
    res = prune_dcps(str(tmp_path), 2, newest_verified_step=4000, logger=log)
    assert res["deleted"] == [1000, 2000]
    assert not (tmp_path / "global_step_1000").exists()
    assert not (tmp_path / "global_step_2000").exists()
    assert (tmp_path / "global_step_3000").exists() and (tmp_path / "global_step_4000").exists()
    line = [l for l in log.lines if "ckpt-retention" in l][0]
    assert "保留 step=[3000, 4000]" in line and "删除 step=[1000, 2000]" in line, line


def test_resume_target_survives_and_keep_count_is_exact(tmp_path):
    """Resume 目标（最新完整 DCP）必须永远存在，且完整份数收敛到 keep_last。"""
    for s in (1000, 2000, 3000, 4000, 5000):
        make_dcp(tmp_path, s)
    prune_dcps(str(tmp_path), 2, newest_verified_step=5000)
    remaining = [s for s, p in list_dcps(str(tmp_path)) if is_complete_dcp(p)]
    assert remaining == [4000, 5000], "Resume 目标 5000 必须保留"
    assert (tmp_path / "global_step_5000" / "model" / ".metadata").is_file()


def test_final_save_then_prune(tmp_path):
    """收尾保存后同样执行清理（最终 DCP 保留，最旧被删）。"""
    for s in (1000, 2000, 3000):
        make_dcp(tmp_path, s)
    prune_dcps(str(tmp_path), 2, newest_verified_step=3000)  # 这份就是"最终 DCP"
    remaining = [s for s, p in list_dcps(str(tmp_path)) if is_complete_dcp(p)]
    assert remaining == [2000, 3000]
    assert (tmp_path / "global_step_3000").is_dir(), "最终 DCP 必须保留"


def test_hf_milestones_tree_untouched(tmp_path):
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    for s in (1000, 2000, 3000):
        make_dcp(ckpt, s)
    milestone = tmp_path / "hf_milestones" / "global_step_1500" / "hf_ckpt"
    milestone.mkdir(parents=True)
    (milestone / "model-00001-of-00006.safetensors").write_bytes(b"hf")
    prune_dcps(str(ckpt), 1, newest_verified_step=3000)
    assert (milestone / "model-00001-of-00006.safetensors").is_file(), "HF 里程碑不得被清理"
    assert (tmp_path / "hf_milestones").is_dir()


def test_forbidden_root_is_refused(tmp_path):
    hf = tmp_path / "hf_milestones"
    make_dcp(hf, 1000)
    assert root_is_forbidden(str(hf)) is True
    plan = plan_retention(str(hf), 1, newest_verified_step=1000)
    assert plan["guarded"] is True and plan["delete"] == []
    assert "HF 里程碑" in plan["reason"]


def test_unknown_entries_are_ignored(tmp_path):
    for s in (1000, 2000):
        make_dcp(tmp_path, s)
    (tmp_path / "some_other_dir").mkdir()
    (tmp_path / "global_step_abc").mkdir()
    res = prune_dcps(str(tmp_path), 1, newest_verified_step=2000)
    assert res["deleted"] == [1000]
    assert (tmp_path / "some_other_dir").is_dir() and (tmp_path / "global_step_abc").is_dir()


def test_prune_is_idempotent(tmp_path):
    for s in (1000, 2000, 3000):
        make_dcp(tmp_path, s)
    prune_dcps(str(tmp_path), 2, newest_verified_step=3000)
    again = prune_dcps(str(tmp_path), 2, newest_verified_step=3000)
    assert again["deleted"] == [], "已收敛后再跑不应再删任何东西"


@pytest.mark.parametrize(
    "smoke,keep,rank0,expected",
    [(False, 2, True, True), (True, 2, True, False), (False, 0, True, False),
     (False, None, True, False), (False, 2, False, False)],
)
def test_should_prune_switch(smoke, keep, rank0, expected):
    """Smoke 无存档 / keep_last=0 / 非 rank0 ⇒ 都不启用清理。"""
    assert should_prune(smoke_no_checkpoint=smoke, keep_last=keep, is_rank0=rank0) is expected


def test_dry_run_does_not_delete(tmp_path):
    for s in (1000, 2000, 3000):
        make_dcp(tmp_path, s)
    res = prune_dcps(str(tmp_path), 1, newest_verified_step=3000, dry_run=True)
    assert res["deleted"] == [1000, 2000]
    assert all((tmp_path / f"global_step_{s}").is_dir() for s in (1000, 2000, 3000))
