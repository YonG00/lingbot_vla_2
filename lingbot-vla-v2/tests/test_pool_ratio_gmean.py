"""候选池「GMean ratio 一把尺子」筛选的**无卡**契约测试。

    python -m pytest tests/test_pool_ratio_gmean.py -q

覆盖（用户口径：统一到"一把尺子"，默认关闭、向后兼容）：

* ``pool_filter_by_gmean_ratio=false``（默认）⇒ 与改造前**逐字一致**：
  缺 baseline 仍 fail-fast、指纹不一致仍 RuntimeError、候选池判据仍是 NMSE 有限。
* 开关打开 ⇒
  * ratio = ``实测 gmean_mse / 该任务参考线`` 分三档：
    ``< pool_ratio_pass`` 已过 / ``[pass, skip]`` 可练 / ``> pool_ratio_skip`` 太难；
    边界值 0.19 / 0.2 / 5.0 / 5.01 逐个钉死。
  * 缺 baseline **不再拒绝启动**（降级 warning）。
  * ``config_fingerprint`` 不一致**只 warning**（不 RuntimeError）。
  * bootstrap 时 NMSE=None 不再把任务排除出候选池；太难/已过的任务由 ratio 档位挡在池外。
* ratio 分档**只筛候选池**：PASS 判定（``check_pass``）逻辑与数值一字未动。
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from al_fixtures import default_al, make_cfg, scheduler_of          # noqa: E402
from lingbotvla.auto_learning.baseline import (                     # noqa: E402
    BaselineStore, runtime_config_fingerprint,
)
from lingbotvla.auto_learning.config import AutoLearningConfig      # noqa: E402
from lingbotvla.auto_learning.decision import thresholds as TH      # noqa: E402
from lingbotvla.auto_learning.decision.thresholds import (          # noqa: E402
    PassCheck, PassThresholds, check_pass,
)
from lingbotvla.auto_learning.orchestration.scheduler import (      # noqa: E402
    Scheduler, _nmse_is_required,
)
from lingbotvla.auto_learning.real import build as al_build         # noqa: E402
from lingbotvla.auto_learning.types import TrajectoryMetrics        # noqa: E402


class _Log:
    def __init__(self):
        self.msgs = []

    def info_rank0(self, msg, *a, **k):
        self.msgs.append(str(msg))

    def info(self, msg, *a, **k):
        self.msgs.append(str(msg))

    def warning(self, msg, *a, **k):
        self.msgs.append(str(msg))


# --------------------------------------------------------------------------- #
# 配置层
# --------------------------------------------------------------------------- #
def _al(*, line: float = 1.0, enabled: bool = False, **kw) -> AutoLearningConfig:
    """gmean 口径 + 一张最小阈值表（参考线 = line）。"""
    base = dict(
        pass_metric="gmean_mse", pass_thresholds_file="th.json",
        pool_filter_by_gmean_ratio=enabled,
    )
    base.update(kw)
    al = default_al(**base)
    al.pass_thresholds = PassThresholds(
        metric="mse", stat="geomean", tasks={"t0": line}, config_fingerprint="fp")
    return al


def test_defaults_are_off_and_legacy_thresholds():
    al = AutoLearningConfig()
    assert al.pool_filter_by_gmean_ratio is False
    assert al.pool_ratio_pass == 0.2
    assert al.pool_ratio_skip == 5.0
    assert TH.pool_filter_enabled(al) is False
    assert _nmse_is_required(al) is True


@pytest.mark.parametrize("pass_,skip", [
    (0.0, 5.0), (-1.0, 5.0), (5.0, 5.0), (6.0, 5.0), (0.2, 0.0), (0.2, -1.0),
])
def test_validation_rejects_bad_ratio_bounds(pass_, skip):
    """必须 0 < pool_ratio_pass < pool_ratio_skip（含相等也拒绝）。"""
    with pytest.raises(ValueError) as err:
        AutoLearningConfig(pool_ratio_pass=pass_, pool_ratio_skip=skip)
    assert "pool_ratio" in str(err.value)


def test_validation_rejects_non_numeric_bounds():
    with pytest.raises(ValueError):
        AutoLearningConfig(pool_ratio_pass="0.2")
    with pytest.raises(ValueError):
        AutoLearningConfig(pool_ratio_skip=True)


def test_validation_requires_gmean_and_thresholds_file():
    with pytest.raises(ValueError) as err:
        AutoLearningConfig(pass_metric="nmse", pool_filter_by_gmean_ratio=True)
    assert "gmean_mse" in str(err.value)

    with pytest.raises(ValueError) as err:
        AutoLearningConfig(pass_metric="gmean_mse", pool_filter_by_gmean_ratio=True)
    assert "pass_thresholds_file" in str(err.value)


# --------------------------------------------------------------------------- #
# ratio 分档（纯函数）—— 边界逐个钉死
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ratio,expected", [
    (0.19, TH.POOL_RATIO_PASSED),      # ratio < 0.2      ⇒ 已过
    (0.20, TH.POOL_RATIO_TRAINABLE),   # 边界含在可练段
    (5.00, TH.POOL_RATIO_TRAINABLE),   # 边界含在可练段
    (5.01, TH.POOL_RATIO_TOO_HARD),    # ratio > 5.0      ⇒ 太难
])
def test_ratio_buckets_at_boundaries(ratio, expected):
    al = _al(line=1.0, enabled=True)
    assert TH.gmean_pool_ratio(al, "t0", gmean_mse=ratio) == pytest.approx(ratio)
    assert TH.gmean_pool_bucket(al, "t0", gmean_mse=ratio) == expected


def test_ratio_bucket_uses_configurable_bounds():
    al = _al(line=1.0, enabled=True, pool_ratio_pass=0.5, pool_ratio_skip=2.0)
    assert TH.gmean_pool_bucket(al, "t0", gmean_mse=0.49) == TH.POOL_RATIO_PASSED
    assert TH.gmean_pool_bucket(al, "t0", gmean_mse=0.5) == TH.POOL_RATIO_TRAINABLE
    assert TH.gmean_pool_bucket(al, "t0", gmean_mse=2.0) == TH.POOL_RATIO_TRAINABLE
    assert TH.gmean_pool_bucket(al, "t0", gmean_mse=2.01) == TH.POOL_RATIO_TOO_HARD


def test_switch_off_never_uses_ratio():
    """开关关闭 ⇒ 恒为 disabled（不自行降级成别的尺子）。"""
    al = _al(line=1.0, enabled=False)
    for value in (0.001, 0.2, 1.0, 5.0, 1e9, None, float("nan")):
        assert TH.gmean_pool_bucket(al, "t0", gmean_mse=value) == TH.POOL_RATIO_DISABLED
    assert _nmse_is_required(al) is True
    assert TH.pool_filter_enabled(al) is False


def test_ratio_needs_line_and_finite_gmean():
    al = _al(line=1.0, enabled=True)
    # 表里没有这个任务 ⇒ 没有参考线
    assert TH.gmean_pool_bucket(al, "unknown", gmean_mse=1.0) == TH.POOL_RATIO_NO_LINE
    # 实测值缺失 / 非有限
    assert TH.gmean_pool_bucket(al, "t0", gmean_mse=None) == TH.POOL_RATIO_INVALID
    assert TH.gmean_pool_bucket(al, "t0", gmean_mse=float("inf")) == TH.POOL_RATIO_INVALID


# --------------------------------------------------------------------------- #
# 候选池筛选（scheduler）：开关关闭 = 旧行为；打开 = ratio 档位
# --------------------------------------------------------------------------- #
def _metric(task: str, *, gmean: float, nmse=None, split: str = "scout") -> TrajectoryMetrics:
    """构造一条 gmean 精确等于 `gmean` 的评测结果（两条轨迹的几何均值）。"""
    mse = gmean
    return TrajectoryMetrics(
        task=task, split=split, episode_ids=[0, 1], per_traj_mse={0: mse, 1: mse},
        mse=mse, baseline_mse=0.25, nmse=nmse, r2=None, metric_valid=True, n_trajs=2,
    )


class _StubEvaluator:
    """固定返回构造好的 gmean（不看 ids）—— 用来精确驱动分档边界。"""

    def __init__(self, by_task):
        self.by_task = by_task

    def evaluate(self, task, split, episode_ids=None):
        value = self.by_task[task]
        return _metric(task, gmean=value, nmse=self.nmse_of(task), split=str(split))

    def nmse_of(self, task):
        return getattr(self, "nmse_map", {}).get(task)


def _sim_scheduler(*, enabled: bool, lines, scout_gmean, nmse_map):
    cfg = make_cfg(len(lines), al=_al(line=1.0, enabled=enabled,
                                      pool_ratio_pass=0.2, pool_ratio_skip=5.0))
    cfg.auto_learning.pass_thresholds = PassThresholds(
        metric="mse", stat="geomean", tasks=dict(lines), config_fingerprint="fp")
    sched = scheduler_of(cfg)
    stub = _StubEvaluator(dict(scout_gmean))
    stub.nmse_map = dict(nmse_map)
    sched.evaluator = stub
    return sched


def test_switch_off_candidate_pool_still_requires_nmse():
    """开关关闭：缺 baseline（nmse=None）⇒ 旧行为，任务进不了候选池。"""
    sched = _sim_scheduler(enabled=False, lines={"t0": 1.0},
                           scout_gmean={"t0": 0.1}, nmse_map={"t0": None})
    sched.advance()                       # bootstrap（2-val scout）
    rec = sched.registry.get("t0")
    assert rec.status == "CANDIDATE"
    assert rec.scout_nmse is None
    assert [r.task_name for r in sched._pool_candidates()] == []
    assert [r.task_name for r in sched.registry.candidate_records()] == []


def test_switch_on_keeps_task_without_baseline_and_classifies_ratio():
    """开关打开：nmse=None（缺 baseline）不再排除；ratio=2.0 ⇒ 进可练段。"""
    sched = _sim_scheduler(enabled=True, lines={"t0": 1.0},
                           scout_gmean={"t0": 2.0}, nmse_map={"t0": None})
    event = sched.advance()
    assert event["result"] == "candidate"
    assert event["pool_ratio"] == pytest.approx(2.0)
    rec = sched.registry.get("t0")
    assert rec.scout_nmse is None and rec.metric_valid is True
    assert [r.task_name for r in sched._pool_candidates()] == ["t0"]


def test_bootstrap_pass_without_baseline_does_not_crash_on_nmse_format():
    """回归：缺 baseline 时 bootstrap PASS 的原因串里 nmse=None。

    旧代码 `f"scout={scout.nmse:.4f}"` 会直接 TypeError（CPU 契约测试抓出来的）。
    """
    sched = _sim_scheduler(enabled=True, lines={"t0": 1.0},
                           scout_gmean={"t0": 0.5}, nmse_map={"t0": None})
    event = sched.advance()
    assert event["result"] == "pass"
    rec = sched.registry.get("t0")
    assert rec.status == "PASS"
    assert rec.current_val_gmean_mse == pytest.approx(0.5)
    assert "n/a" in rec.last_transition_reason, rec.last_transition_reason


def test_bootstrap_too_hard_is_kept_out_of_the_pool():
    """ratio > pool_ratio_skip ⇒ 太难：留 CANDIDATE、不消耗 attempt、不进池。"""
    sched = _sim_scheduler(enabled=True, lines={"t0": 1.0},
                           scout_gmean={"t0": 5.01}, nmse_map={"t0": None})
    event = sched.advance()
    assert event["result"] == "pool_too_hard"
    assert event["pool_ratio"] == pytest.approx(5.01)
    rec = sched.registry.get("t0")
    assert rec.status == "CANDIDATE" and rec.attempt_count == 0
    assert sched._pool_candidates() == []


def test_bootstrap_boundary_ratios_stay_in_the_pool():
    """边界值 5.0 仍算可练（`ratio > skip` 才是太难）。"""
    sched = _sim_scheduler(enabled=True, lines={"t0": 1.0},
                           scout_gmean={"t0": 5.0}, nmse_map={"t0": None})
    event = sched.advance()
    assert event["result"] == "candidate"
    assert [r.task_name for r in sched._pool_candidates()] == ["t0"]


def test_pool_passed_bucket_is_excluded_without_faking_a_pass():
    """``ratio < pool_ratio_pass`` ⇒ 视为已过、不进候选池，但**不**伪造 PASS。

    默认 0.2 < 1 时这类任务本来就会被真正的 PASS 判定接走（见 PASS 线不变）；
    这里把下界配到 2.0 才能观察到该分支：ratio=1.5 未达标（1.5 > 1.0），
    但低于下界 ⇒ 留 CANDIDATE、不进池、状态**不是** PASS。
    """
    cfg = make_cfg(1, al=_al(line=1.0, enabled=True,
                             pool_ratio_pass=2.0, pool_ratio_skip=5.0))
    cfg.auto_learning.pass_thresholds = PassThresholds(
        metric="mse", stat="geomean", tasks={"t0": 1.0}, config_fingerprint="fp")
    sched = scheduler_of(cfg)
    stub = _StubEvaluator({"t0": 1.5})
    stub.nmse_map = {"t0": None}
    sched.evaluator = stub

    event = sched.advance()
    assert event["result"] == "pool_passed"
    assert event["pool_ratio"] == pytest.approx(1.5)
    rec = sched.registry.get("t0")
    assert rec.status == "CANDIDATE", rec.status
    assert sched._pool_candidates() == []


def test_pool_ratio_only_filters_pool_not_pass_logic():
    """ratio 分档绝不改 PASS 判定：同一 gmean 下 check_pass 结果与开关无关。"""
    al_off = _al(line=1.0, enabled=False)
    al_on = _al(line=1.0, enabled=True)
    for gmean in (0.5, 1.0, 2.0, 10.0):
        kw = dict(nmse=None, mse=gmean, gmean_mse=gmean)
        assert check_pass(al_off, "t0", **kw) == check_pass(al_on, "t0", **kw)
    # 达标 ⇒ PASS；未达标 ⇒ BELOW（与 ratio 档位无关）
    assert check_pass(al_on, "t0", nmse=None, mse=0.5, gmean_mse=0.5) == PassCheck.PASS
    assert check_pass(al_on, "t0", nmse=None, mse=2.0, gmean_mse=2.0) == PassCheck.BELOW
    # ratio=0.1 属"已过"档，但它同时**本来就**是 PASS（线内）⇒ 判定一致
    assert check_pass(al_on, "t0", nmse=None, mse=0.1, gmean_mse=0.1) == PassCheck.PASS


def test_ratio_mode_never_hangs_on_missing_nmse():
    """缺 baseline 时 LP 必须用 GMean 算 —— 否则 attempt 永不收尾（无限训练）。"""
    from lingbotvla.auto_learning.decision.state_machine import decide_after_unit
    from lingbotvla.auto_learning.state.registry import TaskRecord

    al = _al(line=1.0, enabled=True)
    rec = TaskRecord(task_name="t0")
    rec.current_val_nmse = None                 # 缺 baseline
    rec.current_val_gmean_mse = 2.0
    rec.lp50 = -0.05                            # 由 GMean 算出来的 LP
    verdict = decide_after_unit(rec, al, attempt_step=200, defer_after_steps=100)
    assert verdict.decision.value == "DEFER", verdict

    # 对照组：GMean 缺失 ⇒ 仍然 DEFER(metric_invalid)，不静默继续
    rec.current_val_gmean_mse = None
    verdict2 = decide_after_unit(rec, al, attempt_step=200, defer_after_steps=100)
    assert verdict2.decision.value == "DEFER"
    assert "gmean_mse" in verdict2.detail


# --------------------------------------------------------------------------- #
# real/build.py：baseline 降级（只在 GMean ratio 模式下）
# --------------------------------------------------------------------------- #
_AL_YAML = """
auto_learning:
  enabled: true
  batch_size: 10
  new_slots: 7
  replay_slots: 3
  eval_interval_steps: 5
  min_steps_before_defer: 10
  defer_retry_steps: 5
  pass_metric: gmean_mse
  pass_thresholds_file: "{thresholds}"
  pool_filter_by_gmean_ratio: {pool}
"""


def _thresholds_file(tmp_path: Path, *, tasks=("t0",)) -> str:
    path = tmp_path / "th.json"
    path.write_text(json.dumps({
        "version": 1, "metric": "mse", "stat": "geomean",
        "config_fingerprint": "fp",
        "tasks": {t: 1.0 for t in tasks},
    }), encoding="utf-8")
    return str(path)


def _write_al_yaml(tmp_path: Path, *, thresholds: str, pool: bool) -> str:
    p = tmp_path / "al.yaml"
    p.write_text(_AL_YAML.format(thresholds=thresholds, pool=str(bool(pool)).lower()),
                 encoding="utf-8")
    return str(p)


class _FakeEntry:
    name = "t0"
    sha256_train = "sha"
    train_sample_ids = [0, 1]
    train_traj_ids = [0]
    val_traj_ids = [1]
    n_train_trajs = 1

    def ids_for(self, split):
        return [0]


class _FakeCatalog:
    meta = {}
    episodes_per_task = 1
    task_order = ["t0"]

    def __contains__(self, n):
        return n == "t0"

    def __len__(self):
        return 1

    def task_names(self):
        return ["t0"]

    def entry(self, n):
        return _FakeEntry()

    def verify(self):
        return []

    def attach_samples(self, resolver):
        return self

    def task_of_episode(self, ep):
        return "t0"


class _FakeResolver:
    def __len__(self):
        return 1


def _patch_catalog(monkeypatch):
    monkeypatch.setattr(al_build, "catalog_from_task_split",
                        lambda *a, **k: _FakeCatalog())
    monkeypatch.setattr(al_build, "SampleResolver", types.SimpleNamespace(
        from_dataset=lambda ds, **kw: _FakeResolver()))


def test_missing_baseline_switch_off_still_fails_fast(tmp_path, monkeypatch):
    """开关关闭 ⇒ 报错文案与改造前**逐字**一致（向后兼容的硬要求）。"""
    _patch_catalog(monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    cfg_path = _write_al_yaml(tmp_path, thresholds=_thresholds_file(tmp_path), pool=False)

    with pytest.raises(ValueError) as err:
        al_build.build_auto_learning_parts(
            config_path=cfg_path, manifest_path=str(manifest),
            train_dataset=object(), baseline_path=None, logger=_Log())
    msg = str(err.value)
    assert "没有可用的 baseline store" in msg
    assert "缺 baseline 时 NMSE 恒为 None，所有任务会被排除出候选池" in msg
    assert "compute_task_baseline" in msg


def test_missing_baseline_switch_on_starts_with_warning(tmp_path, monkeypatch):
    """开关打开 ⇒ 缺 baseline 只 warning，不再拒绝启动。"""
    _patch_catalog(monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    cfg_path = _write_al_yaml(tmp_path, thresholds=_thresholds_file(tmp_path), pool=True)

    log = _Log()
    parts = al_build.build_auto_learning_parts(
        config_path=cfg_path, manifest_path=str(manifest),
        train_dataset=object(), baseline_path=None, logger=log)
    assert parts is not None and parts.baseline_store is None
    assert any("不拒绝启动" in m for m in log.msgs), log.msgs


def _runtime_args(chunk=50):
    ns = types.SimpleNamespace()
    ns.data = types.SimpleNamespace(
        train_path="/data/train/x.txt", norm_stats_file=None,
        cameras=["camera_top"], joints=["arm.position"], img_size=256, chunk_size=None)
    ns.train = types.SimpleNamespace(chunk_size=chunk)
    return ns


def _gmean_cfg(*, pool: bool) -> AutoLearningConfig:
    return AutoLearningConfig(
        eval_interval_steps=5, min_steps_before_defer=10,
        pass_metric="gmean_mse", pass_thresholds_file="th.json",
        pool_filter_by_gmean_ratio=pool)


def test_baseline_fingerprint_mismatch_fails_fast_when_switch_off():
    store = BaselineStore(path="/tmp/x.json", config_fingerprint="deadbeef")
    parts = types.SimpleNamespace(baseline_store=store)
    with pytest.raises(RuntimeError) as err:
        al_build._verify_baseline_fingerprint(
            parts, _gmean_cfg(pool=False), _runtime_args(50), None, _Log())
    assert "指纹" in str(err.value)


def test_baseline_fingerprint_mismatch_only_warns_when_switch_on():
    store = BaselineStore(path="/tmp/x.json", config_fingerprint="deadbeef")
    parts = types.SimpleNamespace(baseline_store=store)
    log = _Log()
    al_build._verify_baseline_fingerprint(
        parts, _gmean_cfg(pool=True), _runtime_args(50), None, log)
    assert any("指纹" in m and "只 warning" in m for m in log.msgs), log.msgs
    # 仍然把两边的指纹逐字记下来（可审计）
    assert any("deadbeef" in m for m in log.msgs)
    assert any(runtime_config_fingerprint(_runtime_args(50), None) in m for m in log.msgs)


def test_baseline_missing_fingerprint_only_warns_when_switch_on():
    store = BaselineStore(path="/tmp/x.json", config_fingerprint="")
    parts = types.SimpleNamespace(baseline_store=store)
    log = _Log()
    al_build._verify_baseline_fingerprint(
        parts, _gmean_cfg(pool=True), _runtime_args(50), None, log)
    assert any("没有 config_fingerprint" in m for m in log.msgs)
    # 开关关闭时同一份 store 仍 fail-fast
    with pytest.raises(RuntimeError):
        al_build._verify_baseline_fingerprint(
            parts, _gmean_cfg(pool=False), _runtime_args(50), None, _Log())


# --------------------------------------------------------------------------- #
# resume 指纹：开关关闭时不得凭空多出键（旧 DCP 必须还能干净 resume）
# --------------------------------------------------------------------------- #
def test_resume_fingerprint_unchanged_when_switch_off():
    from lingbotvla.auto_learning.state.persistence import config_fingerprint

    al_off = _al(line=1.0, enabled=False)
    fp = config_fingerprint(al_off, ["t0"])
    assert "pool_filter_by_gmean_ratio" not in fp
    assert "pool_ratio_pass" not in fp and "pool_ratio_skip" not in fp

    al_on = _al(line=1.0, enabled=True)
    fp_on = config_fingerprint(al_on, ["t0"])
    assert fp_on["pool_filter_by_gmean_ratio"] is True
    assert fp_on["pool_ratio_pass"] == 0.2 and fp_on["pool_ratio_skip"] == 5.0
    assert set(fp_on) - set(fp) == {
        "pool_filter_by_gmean_ratio", "pool_ratio_pass", "pool_ratio_skip"}


# --------------------------------------------------------------------------- #
# tools/compute_pass_thresholds.py：缺 baseline 时守卫 2/3 跳过 + warning
# --------------------------------------------------------------------------- #
def test_compute_thresholds_skips_guards_when_baseline_mse_missing():
    from lingbotvla.auto_learning.tools.compute_pass_thresholds import compute_thresholds

    store = BaselineStore(path="/tmp/x.json", config_fingerprint="fp")
    store.tasks = {"t0": {"mse": None, "fingerprint": "sha"}}
    with pytest.warns(RuntimeWarning, match="跳过"):
        table = compute_thresholds(
            {"t0": [{"task": "t0", "mse": 0.5}]}, store,
            stat="max", margin=0.1, reference="ref")
    assert table.tasks == {"t0": pytest.approx(0.55)}


def test_compute_thresholds_keeps_guard_when_baseline_present():
    """守卫**没有**被删：baseline 可用且阈值 ≥ baseline_mse ⇒ 照旧拒绝产出。"""
    from lingbotvla.auto_learning.decision.thresholds import ThresholdsError
    from lingbotvla.auto_learning.tools.compute_pass_thresholds import compute_thresholds

    store = BaselineStore(path="/tmp/x.json", config_fingerprint="fp")
    store.tasks = {"t0": {"mse": 0.4, "fingerprint": "sha"}}
    with pytest.raises(ThresholdsError) as err:
        compute_thresholds({"t0": [{"task": "t0", "mse": 0.5}]}, store,
                           stat="max", margin=0.1, reference="ref")
    assert "全猜均值" in str(err.value)


def test_compute_thresholds_keeps_nmse_selfconsistency_guard():
    from lingbotvla.auto_learning.decision.thresholds import ThresholdsError
    from lingbotvla.auto_learning.tools.compute_pass_thresholds import compute_thresholds

    store = BaselineStore(path="/tmp/x.json", config_fingerprint="fp")
    store.tasks = {"t0": {"mse": 1.0, "fingerprint": "sha"}}
    with pytest.raises(ThresholdsError) as err:
        compute_thresholds({"t0": [{"task": "t0", "mse": 0.5, "nmse": 0.99}]}, store,
                           stat="max", margin=0.1, reference="ref")
    assert "偏差 >1%" in str(err.value)
