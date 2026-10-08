"""审查最小修复的回归测试（2026-10-08 独立审查 D1/D4/D5/D6/D7）。

只读源码/配置 + 纯状态构造，**不需要 torch / GPU**。
被审补丁本身见 `tests/test_al_smoke_visibility_total_pass.py`；本文件只守「审查方追加的修复」，
避免以后有人把这几处改回去。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# D1：启动器里 $VAR 紧跟非 ASCII 字符（bash 3.2 会把多字节字节并入变量名）
# --------------------------------------------------------------------------- #
def test_d1_launcher_has_no_var_glued_to_non_ascii_byte():
    raw = (ROOT / "experiment/robotwin/al_50task_bf16.sh").read_bytes()
    bad = re.findall(rb"\$[A-Za-z_][A-Za-z0-9_]*[\x80-\xff]", raw)
    assert not bad, (
        "以下写法在 bash 3.2（macOS 自带）下会被解析成含多字节的变量名，"
        f"配合 set -u 直接报 unbound variable，请写成 ${{VAR}}：{bad[:3]}"
    )


def test_d1_launcher_still_thinks_it_is_a_dry_run_safe_script():
    """最小修复只动引号/花括号，不应改变任何参数拼装。"""
    src = (ROOT / "experiment/robotwin/al_50task_bf16.sh").read_text(encoding="utf-8")
    for frag in ("--train.smoke_no_checkpoint", "--train.save_steps", "--train.disk_guard"):
        assert frag in src


# --------------------------------------------------------------------------- #
# D4：类默认 0.33 → 0.10 后，历史配置必须显式固定，避免静默漂移
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["g10_click_bell", "smoke_2task", "smoke_click_bell"])
def test_d4_legacy_configs_pin_hardness_fraction_explicitly(name):
    txt = (ROOT / f"configs/auto_learning/{name}.yaml").read_text(encoding="utf-8")
    assert re.search(r"^hardness_probe_fraction:\s*0\.33\s*(#.*)?$", txt, re.M), (
        f"{name}.yaml 必须显式写 0.33（历史可比），否则会吃新类默认 0.10"
    )


@pytest.mark.parametrize("name", ["formal_50task_4pass", "smoke_gbs4_4task_tb5", "smoke_gbs4_4task_warn200"])
def test_d4_current_configs_use_ten_percent(name):
    txt = (ROOT / f"configs/auto_learning/{name}.yaml").read_text(encoding="utf-8")
    assert re.search(r"^hardness_probe_fraction:\s*0\.10\s*(#.*)?$", txt, re.M), name


def test_d4_class_default_and_config_overrides_agree():
    import yaml

    from lingbotvla.auto_learning.config import AutoLearningConfig

    assert AutoLearningConfig().hardness_probe_fraction == 0.10
    legacy = AutoLearningConfig.from_dict(
        yaml.safe_load((ROOT / "configs/auto_learning/g10_click_bell.yaml").read_text(encoding="utf-8"))
    )
    assert legacy.hardness_probe_fraction == 0.33
    assert legacy.target_total_passed_tasks is None, "历史配置不得被塞进新的收工目标"


# --------------------------------------------------------------------------- #
# D5：smoke YAML 的注释必须与新收工语义一致
# --------------------------------------------------------------------------- #
def test_d5_smoke_tb5_comment_matches_total_pass_semantics():
    txt = (ROOT / "configs/auto_learning/smoke_gbs4_4task_tb5.yaml").read_text(encoding="utf-8")
    assert "不设 early-stop" not in txt, "注释已过时：现在目标=4，4 个全 PASS 会提前收工"
    assert re.search(r"^target_total_passed_tasks:\s*4\s*$", txt, re.M)
    assert ("全 PASS" in txt) or ("提前收工" in txt), "注释里应说明提前收工属预期"


# --------------------------------------------------------------------------- #
# D6：测试不得依赖 CWD
# --------------------------------------------------------------------------- #
def test_d6_no_relative_config_open_in_tests():
    offenders = []
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        txt = p.read_text(encoding="utf-8")
        if re.search(r"""open\(\s*['"]configs/""", txt):
            offenders.append(p.name)
    assert not offenders, f"这些测试用相对路径读配置 ⇒ 任意 CWD 会挂：{offenders}"


def test_d6_target_test_uses_root_anchor():
    txt = (ROOT / "tests/test_al_openloop_update.py").read_text(encoding="utf-8")
    assert "ROOT = Path(__file__).resolve().parents[1]" in txt
    assert "ROOT / 'configs/auto_learning/formal_50task_4pass.yaml'" in txt


# --------------------------------------------------------------------------- #
# D7（用户 18:5x 修正口径）：旧 DCP **不得**把 `auto_passed` 当成 Bootstrap PASS
#   * 新存档：正常使用准确记录的 `bootstrap_passed`
#   * 旧存档：不推断、不伪造；日志显式标 unavailable，且**不输出数字**（不给假 0）
#   * 收工判定仍以 registry 当前状态为准，不受影响
# --------------------------------------------------------------------------- #
class _RecLogger:
    """记录 log_metrics / log_text 调用，用于断言"到底写了什么"。"""

    def __init__(self):
        self.metrics = []
        self.texts = []

    def log_metrics(self, step, name, value):
        self.metrics.append((step, name, value))

    def log_text(self, step, name, value):
        self.texts.append((step, name, value))

    def log_event(self, event):
        pass


def _row(task="t0", step=5):
    return {
        "step": step, "loss": 0.2, "task": task, "attempt": 1, "attempt_step": 5,
        "train_nmse": 0.3, "val_nmse": 0.4, "lp50": 0.1, "gap": 1.1, "overfit": False,
        "n_old": 1, "old_tasks": "t1",
    }


def _sched(*, logger=None, pass_tasks=("t0",), auto_passed=("t0",), bootstrap=("t0",),
           target_total=None):
    """t0 / t1 两个任务；可指定谁已 PASS、谁在 auto_passed、谁被记进 bootstrap_passed。"""
    from al_fixtures import make_cfg, scheduler_of

    from lingbotvla.auto_learning.config import AutoLearningConfig
    from lingbotvla.auto_learning.types import TaskStatus

    kw = {"target_total_passed_tasks": target_total} if target_total else {}
    al = AutoLearningConfig(**kw)
    sched = scheduler_of(make_cfg(curves={"t0": [0.4, 0.2], "t1": [0.5, 0.3]}, al=al), logger=logger)
    for name in pass_tasks:
        sched.registry.get(name).status = TaskStatus.PASS.value
    sched.state.auto_passed = list(auto_passed)
    sched.state.bootstrap_passed = list(bootstrap)
    return sched


def _old_dcp(sched):
    """构造"补丁前旧存档"：scheduler 段里没有 bootstrap_passed / _available 两个键。"""
    from lingbotvla.auto_learning.state import persistence as P

    raw = P.collect_state(sched)
    raw["scheduler"].pop("bootstrap_passed", None)
    raw["scheduler"].pop("bootstrap_passed_available", None)
    return P, raw


def _metric(rec, name):
    return [v for _, n, v in rec.metrics if n == name]


def test_d7_old_dcp_with_rescan_pass_does_not_infer_bootstrap():
    """核心场景：旧存档里只有 auto_passed（含 Rescan 免费 PASS）⇒ 不许推断出 Bootstrap 数。"""
    sched = _sched(pass_tasks=("t0", "t1"), auto_passed=("t0", "t1"), bootstrap=())
    P, raw = _old_dcp(sched)
    P.restore_state(sched, raw)
    assert sched.state.bootstrap_passed == [], "auto_passed 可能含 Rescan PASS，不能当 Bootstrap 来源"
    assert sched.state.bootstrap_passed_available is False


def test_d7_unavailable_logs_status_marker_and_no_number():
    rec = _RecLogger()
    sched = _sched(logger=rec, pass_tasks=("t0", "t1"), auto_passed=("t0", "t1"), bootstrap=())
    P, raw = _old_dcp(sched)
    P.restore_state(sched, raw)
    sched.logger = rec
    sched._log_unit_metrics(_row())

    assert _metric(rec, "curriculum/bootstrap_pass_count") == [], "不可用时不得输出数字（假 0 也算错）"
    assert ("curriculum/bootstrap_pass_count_status", "unavailable_old_checkpoint") in [
        (n, v) for _, n, v in rec.texts
    ], "必须显式标记 unavailable"


def test_d7_new_checkpoint_uses_recorded_bootstrap_source():
    rec = _RecLogger()
    # t0=Bootstrap PASS，t1=Rescan/训练 PASS ⇒ 只应计入 t0
    sched = _sched(logger=rec, pass_tasks=("t0", "t1"), auto_passed=("t0", "t1"), bootstrap=("t0",))
    assert sched.state.bootstrap_passed_available is True
    sched._log_unit_metrics(_row())
    assert _metric(rec, "curriculum/bootstrap_pass_count") == [1]
    assert [n for _, n, _ in rec.texts if n.endswith("bootstrap_pass_count_status")] == []


def test_d7_recorded_empty_zero_is_still_logged_as_zero():
    rec = _RecLogger()
    sched = _sched(logger=rec, pass_tasks=("t1",), auto_passed=("t1",), bootstrap=())
    sched._log_unit_metrics(_row())
    assert _metric(rec, "curriculum/bootstrap_pass_count") == [0], "键在但为空 ⇒ 0 是准确的，可以显示"


def test_d7_roundtrip_preserves_availability_flag():
    from lingbotvla.auto_learning.state import persistence as P

    sched = _sched(pass_tasks=("t0",), auto_passed=("t0",), bootstrap=("t0",))
    P.restore_state(sched, P.collect_state(sched))  # 新存档带两个键
    assert sched.state.bootstrap_passed_available is True
    assert sched.state.bootstrap_passed == ["t0"]


def test_d7_stop_condition_still_uses_registry_state():
    """不可用标志只影响统计口径：收工判定仍按 registry 当前 PASS 数。"""
    sched = _sched(target_total=2, pass_tasks=("t0", "t1"), auto_passed=("t0", "t1"), bootstrap=())
    P, raw = _old_dcp(sched)
    P.restore_state(sched, raw)
    assert sched.state.bootstrap_passed_available is False
    # 未跑完 bootstrap 时不许收工（这是既有护栏，与本修正无关）
    assert sched._total_pass_target_reached() is False
    sched.state.bootstrap_queue = []          # 模拟 bootstrap 已跑完
    assert sched._total_pass_target_reached() is True, "收工判定只看 registry 当前 PASS 数"
    assert sched._current_total_passed() == 2


def test_d7_final_report_reports_none_and_status_when_unavailable():
    sched = _sched(pass_tasks=("t0", "t1"), auto_passed=("t0", "t1"), bootstrap=())
    P, raw = _old_dcp(sched)
    P.restore_state(sched, raw)
    fr = sched.final_report()
    assert fr["bootstrap_pass_count"] is None, "unknown 必须是 None，不能是 0"
    assert fr["bootstrap_pass_count_status"] == "unavailable_old_checkpoint"
    assert fr["current_total_pass_count"] == 2, "当前总 PASS 仍必须准确"
