"""review v0.2（`stage_b1_repo_review_v0_2.md`）逐条修复的**无卡**回归测试。

    python -m pytest tests/test_b1_v0_2_fixes.py -q

覆盖：
  #1  _al_on 用真实 parsed config + rmpad fail-fast
  #2  unit 边界 / 存档边界 / resume（另见 test_auto_learning_hook.py）
  #3  AL logical batch_size == dataloader_batch_size fail-fast
  #4  evaluator dataset cache 有上界
  #5  baseline runtime 指纹对拍 + 必填
  #6  关键错误 fail-fast（断言层已由 sanity 提供，这里查「默认必填」）
  #8  hardness 的 transform 随机性
  #9  unique_batches 语义（另见 test_auto_learning_hook.py）
"""

from __future__ import annotations

import random
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lingbotvla.auto_learning.baseline import (                       # noqa: E402
    BaselineStore, FixedBaseline, config_fingerprint_from_data_config,
    runtime_config_fingerprint,
)
from lingbotvla.auto_learning.config import AutoLearningConfig        # noqa: E402
from lingbotvla.auto_learning.evaluator import EvaluatorAdapter       # noqa: E402
from lingbotvla.auto_learning.real import build as al_build           # noqa: E402
from lingbotvla.auto_learning.real.backend import RealHardnessScorer  # noqa: E402
from lingbotvla.auto_learning.real.determinism import (               # noqa: E402
    collect_feature_transforms, deterministic_sampling, restore_rng, snapshot_rng,
)

_AL_YAML = """
auto_learning:
  enabled: {enabled}
  batch_size: 10
  new_slots: 7
  replay_slots: 3
  eval_interval_steps: {interval}
  min_steps_before_defer: {defer}
  defer_retry_steps: {retry}
"""


def _write_al_yaml(tmp_path: Path, *, enabled: bool, interval: int = 5,
                   name: str = "al.yaml", extra: str = "") -> str:
    p = tmp_path / name
    text = _AL_YAML.format(enabled=str(bool(enabled)).lower(), interval=interval,
                           defer=interval * 2, retry=interval)
    p.write_text(text + extra, encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
# #1  _al_on 真开关 + rmpad fail-fast
# --------------------------------------------------------------------------- #
def test_auto_learning_enabled_parses_config_file(tmp_path):
    """🔴 关键：#1 —— `args.train.auto_learning` 是**路径字符串**。

    旧代码 `getattr(path_str, "enabled", False)` 恒为 False（AL 保护全失效）；
    必须真解析配置文件。
    """
    on_path = _write_al_yaml(tmp_path, enabled=True, name="on.yaml")
    off_path = _write_al_yaml(tmp_path, enabled=False, name="off.yaml")

    assert al_build.auto_learning_enabled(on_path) is True
    assert al_build.auto_learning_enabled(off_path) is False
    assert al_build.auto_learning_enabled(None) is False
    assert al_build.auto_learning_enabled("") is False

    # 反证：把「路径」当成 config 对象去取 enabled ⇒ 永远是 False（这就是原 bug）
    assert getattr(on_path, "enabled", False) is False


def test_rmpad_fails_fast_when_auto_learning_enabled():
    """enabled=true + rmpad ⇒ 启动前 fail-fast（不静默强改）。"""
    with pytest.raises(ValueError) as err:
        al_build.check_rmpad_for_auto_learning(True, True, False)
    assert "rmpad" in str(err.value)

    with pytest.raises(ValueError):
        al_build.check_rmpad_for_auto_learning(True, False, True)

    # disabled ⇒ 不管 rmpad 怎么设都不该管
    al_build.check_rmpad_for_auto_learning(False, True, True)
    # enabled 且 rmpad 关 ⇒ 通过
    al_build.check_rmpad_for_auto_learning(True, False, False)


# --------------------------------------------------------------------------- #
# #3  batch_size 对齐
# --------------------------------------------------------------------------- #
class _Log:
    def __init__(self):
        self.msgs = []

    def info_rank0(self, msg, *a, **k):
        self.msgs.append(str(msg))

    def info(self, msg, *a, **k):
        self.msgs.append(str(msg))

    def warning(self, msg, *a, **k):
        self.msgs.append(str(msg))


def _args_with_batch(*, micro=1, gbs=10, dls=10):
    ns = types.SimpleNamespace()
    ns.train = types.SimpleNamespace(
        micro_batch_size=micro, global_batch_size=gbs, dataloader_batch_size=dls)
    return ns


def test_batch_alignment_mismatch_fails_fast():
    cfg = AutoLearningConfig(batch_size=10, new_slots=7, replay_slots=3,
                             eval_interval_steps=5, min_steps_before_defer=10)
    log = _Log()
    with pytest.raises(ValueError) as err:
        al_build.validate_batch_alignment(cfg, _args_with_batch(dls=8), log)
    assert "dataloader_batch_size" in str(err.value)


def test_batch_alignment_ok_logs_summary():
    cfg = AutoLearningConfig(batch_size=10, new_slots=7, replay_slots=3,
                             eval_interval_steps=5, min_steps_before_defer=10)
    log = _Log()
    al_build.validate_batch_alignment(cfg, _args_with_batch(), log)
    blob = "\n".join(log.msgs)
    for key in ("AL logical batch_size", "micro_batch_size", "global_batch_size",
                "dataloader_batch_size", "num_micro_batch"):
        assert key in blob, f"日志里缺 {key}"


# --------------------------------------------------------------------------- #
# #4  evaluator dataset cache 清理
# --------------------------------------------------------------------------- #
class _Entry:
    def __init__(self, name, train_ids, val_ids):
        self.name = name
        self.sha256_train = "sha_" + name
        self._ids = {"train": list(train_ids), "val": list(val_ids)}

    def ids_for(self, split):
        return list(self._ids[split])


class _Catalog:
    def __init__(self, entries):
        self._e = {e.name: e for e in entries}

    def entry(self, name):
        return self._e[name]


class _Validator:
    def __init__(self):
        self.calls = 0
        self.cleared = 0

    def evaluate_ids(self, ids, tag):
        self.calls += 1
        return {"mse": 0.5, "n": len(ids), "n_chunks": len(ids), "frames": 3, "dims": 2}

    def clear_dataset_cache(self):
        self.cleared += 1
        return self.calls       # 假装释放了这么多份


def test_evaluator_clears_dataset_cache_every_eval():
    cat = _Catalog([_Entry("t0", [0, 1, 2, 3], [10, 11, 12, 13])])
    v = _Validator()
    adapter = EvaluatorAdapter(v, cat, None, logger=_Log())
    for ids in ([10], [10, 11], [10, 11, 12], [10, 11, 12, 13]):
        adapter.evaluate_task("t0", "val", ids)
    assert v.calls == 4
    assert v.cleared == 4, "每次评测后都必须清一次数据集缓存（否则长跑内存只增不减）"


def test_evaluator_clear_survives_validator_without_cache_api():
    cat = _Catalog([_Entry("t0", [0, 1], [10, 11])])

    class _NoClear:
        def evaluate_ids(self, ids, tag):
            return {"mse": 1.0, "n": len(ids)}

    adapter = EvaluatorAdapter(_NoClear(), cat, None, logger=_Log())
    assert adapter.evaluate_task("t0", "val", [10]).mse == 1.0


# --------------------------------------------------------------------------- #
# #5  baseline runtime 指纹对拍 + 必填
# --------------------------------------------------------------------------- #
def _data_cfg():
    return {
        "train_path": "/data/train/x.txt",
        "norm_stats_file": None,
        "cameras": ["camera_top"],
        "joints": ["arm.position"],
        "img_size": 256,
    }


def _runtime_args(chunk=50):
    ns = types.SimpleNamespace()
    ns.data = types.SimpleNamespace(
        train_path=_data_cfg()["train_path"], norm_stats_file=None,
        cameras=["camera_top"], joints=["arm.position"], img_size=256, chunk_size=None)
    ns.train = types.SimpleNamespace(chunk_size=chunk)
    return ns


def test_runtime_fingerprint_matches_cli_side_fingerprint():
    """训练侧与 CLI 侧对**同一份配置**必须算出同一个指纹。"""
    cli_fp = config_fingerprint_from_data_config(_data_cfg(), chunk_size=50)
    run_fp = runtime_config_fingerprint(_runtime_args(50), None)
    assert cli_fp == run_fp


def test_runtime_fingerprint_changes_when_chunk_changes():
    a = runtime_config_fingerprint(_runtime_args(50), None)
    b = runtime_config_fingerprint(_runtime_args(25), None)
    assert a != b, "chunk_size 变了指纹必须变（否则旧分母会被错误复用）"


def test_baseline_fingerprint_mismatch_fails_fast():
    store = BaselineStore(path="/tmp/x.json", config_fingerprint="deadbeef")
    parts = types.SimpleNamespace(baseline_store=store)
    cfg = AutoLearningConfig(eval_interval_steps=5, min_steps_before_defer=10)
    with pytest.raises(RuntimeError) as err:
        al_build._verify_baseline_fingerprint(parts, cfg, _runtime_args(50), None, _Log())
    assert "指纹" in str(err.value)


def test_baseline_fingerprint_match_passes():
    fp = runtime_config_fingerprint(_runtime_args(50), None)
    store = BaselineStore(path="/tmp/x.json", config_fingerprint=fp)
    parts = types.SimpleNamespace(baseline_store=store)
    cfg = AutoLearningConfig(eval_interval_steps=5, min_steps_before_defer=10)
    log = _Log()
    al_build._verify_baseline_fingerprint(parts, cfg, _runtime_args(50), None, log)
    assert any("对拍通过" in m for m in log.msgs)


def test_baseline_missing_is_fatal_by_default(tmp_path, monkeypatch):
    """enabled=true 且没有 baseline ⇒ 默认 fail-fast；allow_missing_baseline 才放行。"""
    monkeypatch.setattr(al_build, "catalog_from_task_split",
                        lambda *a, **k: _FakeCatalog(["t0"]))
    monkeypatch.setattr(al_build, "SampleResolver",
                        types.SimpleNamespace(from_dataset=lambda ds, **kw: _FakeResolver()))

    manifest = str(tmp_path / "manifest.json")
    Path(manifest).write_text("{}", encoding="utf-8")

    cfg_path = _write_al_yaml(tmp_path, enabled=True, name="strict.yaml")
    with pytest.raises(ValueError) as err:
        al_build.build_auto_learning_parts(
            config_path=cfg_path, manifest_path=manifest,
            train_dataset=object(), baseline_path=None, logger=_Log())
    assert "baseline" in str(err.value).lower()

    # 显式允许（smoke / 单测）⇒ 能建出来
    cfg_path2 = _write_al_yaml(tmp_path, enabled=True, name="lenient.yaml",
                               extra="  allow_missing_baseline: true\n")
    parts = al_build.build_auto_learning_parts(
        config_path=cfg_path2, manifest_path=manifest,
        train_dataset=object(), baseline_path=None, logger=_Log())
    assert parts is not None and parts.baseline_store is None


class _FakeEntry:
    def __init__(self, name):
        self.name = name
        self.sha256_train = "sha"
        self.train_sample_ids = [0, 1]
        self.train_traj_ids = [0]
        self.val_traj_ids = [1]
        self.n_train_trajs = 1

    def ids_for(self, split):
        return [0]


class _FakeCatalog:
    def __init__(self, names):
        self._names = list(names)
        self.meta = {}
        self.episodes_per_task = 1
        self.task_order = list(names)

    def __contains__(self, n):
        return n in self._names

    def __len__(self):
        return len(self._names)

    def task_names(self):
        return list(self._names)

    def entry(self, n):
        return _FakeEntry(n)

    def verify(self):
        return []

    def attach_samples(self, resolver):
        return self

    def task_of_episode(self, ep):
        return self._names[0]


class _FakeResolver:
    def __len__(self):
        return 1


# --------------------------------------------------------------------------- #
# #6 / #8  image_augment 约束 + hardness 确定性
# --------------------------------------------------------------------------- #
def test_image_augment_is_fatal_by_default():
    cfg = AutoLearningConfig(eval_interval_steps=5, min_steps_before_defer=10)
    args = types.SimpleNamespace(data=types.SimpleNamespace(image_augment=True))
    with pytest.raises(ValueError) as err:
        al_build._guard_image_augment(cfg, args, _Log())
    assert "image_augment" in str(err.value)

    # 显式放行 ⇒ 只警告
    cfg2 = AutoLearningConfig(eval_interval_steps=5, min_steps_before_defer=10,
                              allow_image_augment=True)
    log = _Log()
    al_build._guard_image_augment(cfg2, args, log)
    assert any("image_augment" in m for m in log.msgs)

    # 没开增强 ⇒ 什么都不做
    args_ok = types.SimpleNamespace(data=types.SimpleNamespace(image_augment=False))
    al_build._guard_image_augment(cfg, args_ok, _Log())


class _FT:
    def __init__(self, aug=True):
        self.image_augment = aug


class _DS:
    def __init__(self, ft):
        self.feature_transform = ft


def test_deterministic_sampling_disables_augment_and_restores_rng():
    ds = _DS(_FT(True))
    assert len(collect_feature_transforms(ds)) == 1

    before = random.getstate()
    with deterministic_sampling(ds) as rep:
        assert rep == {"n_ft": 1, "n_disabled": 1}
        assert ds.feature_transform.image_augment is False, "上下文内增强必须被关掉"
        random.random()          # 模拟「取 item 消耗了全局 RNG」
    assert ds.feature_transform.image_augment is True, "退出时必须还原"
    assert random.getstate() == before, "全局 RNG 必须逐位还原"


def test_deterministic_sampling_finds_multi_vla_sub_datasets():
    ds = types.SimpleNamespace(
        feature_transforms={"a": _FT(True)},
        _datasets=[types.SimpleNamespace(feature_transform=_FT(False))],
    )
    with deterministic_sampling(ds) as rep:
        assert rep["n_ft"] == 2 and rep["n_disabled"] == 1
        assert ds.feature_transforms["a"].image_augment is False
    assert ds.feature_transforms["a"].image_augment is True


class _ItemDS:
    """带 `__getitem__` 的数据集替身（特殊方法必须定义在**类**上，实例赋值无效）。"""

    def __init__(self, ft, items):
        self.feature_transform = ft
        self.items = items
        self.seen: list = []

    def __getitem__(self, i):
        self.seen.append(self.feature_transform.image_augment)
        return self.items[i]


def test_real_hardness_scorer_samples_without_augment():
    """`RealHardnessScorer` 取 item 时增强必须是关的，取完必须还原。"""
    ds = _ItemDS(_FT(True), {0: {"v": 0}, 1: {"v": 1}})

    class _Scorer:
        def score(self, items):
            return [float(it["v"]) for it in items]

    s = RealHardnessScorer(_Scorer(), ds)
    out = s.score("t0", [0, 1])
    assert out == {0: 0.0, 1: 1.0}
    assert ds.seen == [False, False], "取 item 期间增强必须是关的"
    assert ds.feature_transform.image_augment is True, "取完必须还原"


def test_snapshot_restore_rng_roundtrip():
    st = snapshot_rng()
    random.random()
    restore_rng(st)
    assert random.getstate() == st["random"]
