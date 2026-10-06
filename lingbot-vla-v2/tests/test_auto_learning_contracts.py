"""Stage B0 —— **no-model contract tests**。

设计原则
--------
* **不需要 torch / lerobot / GPU**：只测「纯逻辑 + 接线」。
  重依赖模块（`open_loop_validation` / `evaluator.evaluate_task` 的真实路径 /
  `hardness`）不在这里测 —— 见 `test_auto_learning_real_model.py`。
* 覆盖：划分正确 / train-val 无泄漏 / sample index 映射稳定 / baseline 可复现且口径正确 /
  adapter 走的是 `evaluate_ids`（而不是裸调 `_evaluate_ids`）。

    python -m pytest tests/test_auto_learning_contracts.py -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ⚠️ 受限沙箱里 pytest 的 `tmp_path` 可能 mkdir 失败（/private/var/... 不可写）
#    ⇒ 统一用**仓库内**的临时目录（与 auto_learning Demo 的踩坑结论一致）。
_TMP_ROOT = REPO / ".pytest_tmp"


@pytest.fixture
def tmp_path():
    import shutil
    import uuid

    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    d = _TMP_ROOT / f"al_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)

from lingbotvla.auto_learning.baseline import (          # noqa: E402
    BaselineStore, MU_GLOBAL, MU_TRAJECTORY, build_baseline, compute_mu,
    compute_fixed_baseline, baseline_fingerprint, trajectory_balanced_baseline,
)
from lingbotvla.auto_learning.catalog import TaskCatalog  # noqa: E402
from lingbotvla.auto_learning.evaluator import EvaluatorAdapter  # noqa: E402
from lingbotvla.auto_learning.ports import Backend, EvalResult, SampleRef, TaskEntry  # noqa: E402
from lingbotvla.auto_learning.resolver import SampleResolver  # noqa: E402
from lingbotvla.auto_learning.trainer import TrainerAdapter  # noqa: E402

EPISODES_PER_TASK = 50


# --------------------------------------------------------------------------- #
# 测试素材
# --------------------------------------------------------------------------- #
def _fake_manifest(root: Path, tasks=("click_bell", "turn_switch"), n_ep=EPISODES_PER_TASK,
                   val_ratio=0.2, inline_ids=True) -> Path:
    """造一份与 `tools/task_split.py` 同构的 manifest（50 回合/任务，train 40 / val 10）。"""
    n_val = max(1, int(round(n_ep * val_ratio)))
    records = {}
    for b, name in enumerate(tasks):
        lo = b * n_ep
        ids = list(range(lo, lo + n_ep))
        val = ids[:: (n_ep // n_val)][:n_val]           # 等间隔取 val，确定性
        train = [i for i in ids if i not in set(val)]
        rec = {
            "n_total": n_ep, "n_train": len(train), "n_val": len(val),
            "train_frames": 0, "val_frames": 0,
            "val_ratio": val_ratio, "strategy": "quantile",
            "sha256_train": f"tr-{name}", "sha256_val": f"va-{name}",
        }
        if inline_ids:
            rec["train_ids"] = train
            rec["val_ids"] = val
        records[name] = rec
        if not inline_ids:                               # 走 <task>.*_ids.json 分支
            (root / f"{name}.train_ids.json").write_text(json.dumps(train), encoding="utf-8")
            (root / f"{name}.val_ids.json").write_text(json.dumps(val), encoding="utf-8")
    man = {"episodes_per_task": n_ep, "val_ratio": val_ratio, "tasks": records}
    p = root / "manifest.json"
    p.write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
    return p


class _FakeHF:
    """最小 hf_dataset：只支持按列名取列。"""

    def __init__(self, cols):
        self._cols = cols

    def __getitem__(self, key):
        return self._cols[key]


class _FakeDS:
    """最小数据集：暴露 hf_dataset + __len__（+ 可选 _datasets 下钻层）。"""

    def __init__(self, episode_index, frame_index, *, wrap=False):
        self.hf_dataset = _FakeHF({"episode_index": list(episode_index),
                                   "index": list(frame_index)})
        if wrap:                                          # 模拟 MultiVLADataset 单条目
            inner = type("_Inner", (), {"dataset": self})()
            self._datasets = [inner]

    def __len__(self):
        return len(self.hf_dataset._cols["index"])


def _two_episode_dataset(wrap=False):
    """两个回合：ep10 三帧(绝对 100,101,102)、ep11 两帧(绝对 200,201)。"""
    return _FakeDS([10, 10, 10, 11, 11], [100, 101, 102, 200, 201], wrap=wrap)


def _task_of_episode(ep):
    return {10: "click_bell", 11: "click_bell"}.get(int(ep), "unknown")


class _StubValidator:
    """桩：记录调用，返回预设的 `_evaluate_ids` 风格 dict。"""

    def __init__(self, payload=None):
        self.payload = payload or {
            "mse": 0.02, "mae": 0.1, "n": 2, "n_chunks": 4, "frames": 200, "dims": 14,
            "per_traj_mse": [0.01, 0.03], "per_traj_ids": [10, 11],
            "per_traj_frames": [100, 100],
        }
        self.evaluate_ids_calls = []
        self.private_calls = 0

    def evaluate_ids(self, ids, tag):
        self.evaluate_ids_calls.append((list(ids), tag))
        return dict(self.payload)

    def _evaluate_ids(self, ids, tag):        # 不该被 adapter 直接调用
        self.private_calls += 1
        raise AssertionError("EvaluatorAdapter 不得裸调 _evaluate_ids()")


# --------------------------------------------------------------------------- #
# 1) task / episode 划分正确
# --------------------------------------------------------------------------- #
def test_catalog_loads_and_splits_are_disjoint(tmp_path):
    man = _fake_manifest(tmp_path)
    cat = TaskCatalog.from_manifest(str(man))
    assert set(cat.names()) == {"click_bell", "turn_switch"}
    all_ids = []
    for b, e in enumerate(cat):
        assert e.n_train == 40 and e.n_val == 10
        assert set(e.train_ids) & set(e.val_ids) == set()
        # 回合号是**全局**的：第 b 个任务占 [b*50, (b+1)*50)
        assert sorted(list(e.train_ids) + list(e.val_ids)) == list(range(b * 50, (b + 1) * 50))
        all_ids += list(e.train_ids) + list(e.val_ids)
    assert len(all_ids) == len(set(all_ids))              # 跨任务不重叠
    assert cat.verify() == []


def test_catalog_raises_on_leakage(tmp_path):
    man = _fake_manifest(tmp_path)
    raw = json.loads(man.read_text(encoding="utf-8"))
    raw["tasks"]["click_bell"]["val_ids"].append(raw["tasks"]["click_bell"]["train_ids"][0])
    man.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError) as err:
        TaskCatalog.from_manifest(str(man))
    assert "泄漏" in str(err.value) or "train/val" in str(err.value)


def test_catalog_raises_on_empty_split(tmp_path):
    man = _fake_manifest(tmp_path)
    raw = json.loads(man.read_text(encoding="utf-8"))
    raw["tasks"]["click_bell"]["val_ids"] = []
    man.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError) as err:
        TaskCatalog.from_manifest(str(man))
    assert "空 val" in str(err.value)


def test_catalog_block_structure_checked_against_task_order(tmp_path):
    man = _fake_manifest(tmp_path, tasks=("click_bell", "turn_switch"))
    order = ["click_bell", "turn_switch"]
    cat = TaskCatalog.from_manifest(str(man), task_order=order, episodes_per_task=50)
    assert cat.verify() == []
    assert cat.task_of_episode(0) == "click_bell"
    assert cat.task_of_episode(50) == "turn_switch"
    # 顺序给反了 ⇒ 分块结构必须报错（而不是静默算错）
    bad = TaskCatalog.from_manifest(str(man), task_order=list(reversed(order)),
                                    episodes_per_task=50, strict=False)
    assert any(p.kind == "分块结构不符" for p in bad.verify())


def test_catalog_reads_sibling_id_files(tmp_path):
    man = _fake_manifest(tmp_path, inline_ids=False)
    cat = TaskCatalog.from_manifest(str(man))
    assert cat.entry("click_bell").n_train == 40


def test_catalog_unknown_task_raises(tmp_path):
    cat = TaskCatalog.from_manifest(str(_fake_manifest(tmp_path)))
    with pytest.raises(KeyError):
        cat.entry("nope")
    with pytest.raises(ValueError):
        cat.entry("click_bell").ids_for("test")


# --------------------------------------------------------------------------- #
# 2) train / val 无泄漏（数据集侧）
# --------------------------------------------------------------------------- #
def test_no_val_leakage_into_training_whitelist(tmp_path):
    cat = TaskCatalog.from_manifest(str(_fake_manifest(tmp_path)))
    for e in cat:
        train, val = set(e.train_ids), set(e.val_ids)
        assert not (train & val)
        # 白名单是「数据集侧」的唯一入口 ⇒ 只要不重叠就不会泄漏
        assert len(train) + len(val) == EPISODES_PER_TASK


# --------------------------------------------------------------------------- #
# 3) sample index 映射稳定
# --------------------------------------------------------------------------- #
def test_resolver_roundtrip_and_columns():
    ds = _two_episode_dataset()
    res = SampleResolver.from_dataset(ds, task_of_episode=_task_of_episode)
    assert len(res) == 5
    np.testing.assert_array_equal(res.episode_map(), [10, 10, 10, 11, 11])
    np.testing.assert_array_equal(res.frame_map(), [100, 101, 102, 200, 201])
    for i in range(5):
        ref = res.local_to_ref(i)
        assert res.ref_to_local(ref) == i
    assert res.local_to_ref(3) == SampleRef(task="click_bell", episode=11, frame=200)
    assert res.local_indices_of_episode(10) == [0, 1, 2]
    assert res.verify() == []


def test_resolver_descends_into_multi_dataset_wrapper():
    res = SampleResolver.from_dataset(_two_episode_dataset(wrap=True),
                                      task_of_episode=_task_of_episode)
    assert len(res) == 5 and res.local_to_ref(0).frame == 100


def test_resolver_is_stable_across_rebuilds():
    a = SampleResolver.from_dataset(_two_episode_dataset(), task_of_episode=_task_of_episode)
    b = SampleResolver.from_dataset(_two_episode_dataset(), task_of_episode=_task_of_episode)
    assert [a.local_to_ref(i) for i in range(len(a))] == [b.local_to_ref(i) for i in range(len(b))]


def test_resolver_rejects_bad_columns():
    # 列长不一致 ⇒ 映射不可信，必须拒绝
    with pytest.raises(ValueError):
        SampleResolver(episode_index=np.array([1, 2]), frame_index=np.array([1, 2, 3]),
                       task_of_episode=_task_of_episode)
    with pytest.raises(ValueError):
        SampleResolver(episode_index=np.array([], dtype=np.int64),
                       frame_index=np.array([], dtype=np.int64),
                       task_of_episode=_task_of_episode)
    # 下钻不到 hf_dataset ⇒ RuntimeError
    class _NoHF:
        def __len__(self):
            return 3

    with pytest.raises(RuntimeError):
        SampleResolver.from_dataset(_NoHF(), task_of_episode=_task_of_episode)


def test_resolver_verify_catches_non_monotonic_frames():
    ds = _FakeDS([10, 10, 11], [100, 99, 200])           # 帧号回退
    res = SampleResolver.from_dataset(ds, task_of_episode=_task_of_episode)
    assert any("严格递增" in p for p in res.verify())


def test_resolver_rejects_missing_coordinate():
    res = SampleResolver.from_dataset(_two_episode_dataset(), task_of_episode=_task_of_episode)
    with pytest.raises(KeyError):
        res.ref_to_local(SampleRef(task="click_bell", episode=99, frame=0))


# --------------------------------------------------------------------------- #
# 4) fixed baseline：口径正确 + 可复现
# --------------------------------------------------------------------------- #
def _chunks():
    """两条轨迹：ep10 两块、ep11 一块（长度不同，用来暴露加权差异）。"""
    rng = np.random.default_rng(0)
    return [
        (10, rng.normal(0.0, 1.0, size=(4, 3))),
        (10, rng.normal(0.0, 1.0, size=(4, 3))),
        (11, rng.normal(5.0, 1.0, size=(2, 3))),
    ]


def test_mu_global_is_pooled_mean():
    ch = _chunks()
    mu = compute_mu(ch, weighting=MU_GLOBAL)
    expected = np.concatenate([g for _, g in ch], axis=0).mean(axis=0)
    np.testing.assert_allclose(mu, expected)


def test_mu_trajectory_weights_trajectories_equally():
    ch = _chunks()
    mu = compute_mu(ch, weighting=MU_TRAJECTORY)
    per = {}
    for k, g in ch:
        per.setdefault(k, []).append(g)
    means = [np.concatenate(v, axis=0).mean(axis=0) for v in per.values()]
    np.testing.assert_allclose(mu, np.mean(np.stack(means, axis=0), axis=0))
    # 与 global 不同（轨迹长度不等）
    assert not np.allclose(mu, compute_mu(ch, weighting=MU_GLOBAL))


def test_mu_rejects_unknown_weighting():
    with pytest.raises(ValueError):
        compute_mu(_chunks(), weighting="nope")


def test_baseline_is_trajectory_balanced_not_frame_weighted():
    """核心口径：先按轨迹算 MSE、再对轨迹等权 —— 而不是把所有帧 pool 在一起。"""
    ch = _chunks()
    mu = compute_mu(ch, weighting=MU_GLOBAL)
    seen = {}

    def stub_aggregate(pairs):
        """复刻 aggregate_chunks 的轨迹口径（不依赖 torch）。"""
        groups, order = {}, []
        for k, gt, pr in pairs:
            if k not in groups:
                groups[k] = []
                order.append(k)
            groups[k].append((gt, pr))
        per = []
        for k in order:
            gts = np.concatenate([g for g, _ in groups[k]], axis=0)
            prs = np.concatenate([p for _, p in groups[k]], axis=0)
            per.append(float(np.mean((prs - gts) ** 2)))
        return {"mse": float(np.mean(per)), "per_traj_mse": per}

    got = trajectory_balanced_baseline(ch, mu, aggregate=stub_aggregate)

    # 手算：每条轨迹 (gt_i - μ)² 的均值，再对轨迹等权
    per = {}
    for k, g in ch:
        per.setdefault(k, []).append(g)
    manual = []
    for k, gs in per.items():
        gts = np.concatenate(gs, axis=0)
        manual.append(float(np.mean((gts - mu) ** 2)))
    assert got == pytest.approx(float(np.mean(manual)), rel=0, abs=1e-12)

    # 反例：把帧 pool 在一起（忽略轨迹等权）会得到不同的值
    allgt = np.concatenate([g for _, g in ch], axis=0)
    frame_weighted = float(np.mean((allgt - mu) ** 2))
    assert frame_weighted != pytest.approx(got)


def test_build_baseline_is_reproducible():
    ch = _chunks()
    a = build_baseline("t", ch, fingerprint="fp", action_keys=["action.a"],
                       n_train_episodes=40, aggregate=lambda p: {"mse": 1.0})
    b = build_baseline("t", ch, fingerprint="fp", action_keys=["action.a"],
                       n_train_episodes=40, aggregate=lambda p: {"mse": 1.0})
    assert a.to_dict() == b.to_dict()
    assert a.n_chunks == 3 and a.n_frames == 10 and a.dims == 3
    assert a.n_train_episodes == 40


def test_baseline_dimension_mismatch_raises():
    ch = _chunks()
    with pytest.raises(ValueError) as err:
        trajectory_balanced_baseline(ch, np.zeros(7), aggregate=lambda p: {"mse": 0.0})
    assert "维度" in str(err.value)


# --------------------------------------------------------------------------- #
# 5) 指纹与缓存
# --------------------------------------------------------------------------- #
def test_fingerprint_changes_with_any_field(tmp_path):
    ns = tmp_path / "norm.json"
    ns.write_text('{"a": 1}', encoding="utf-8")
    base = dict(dataset_root="/d", sha256_train="abc", norm_stats_file=str(ns),
                cameras=["head"], joints=["arm"], chunk_size=50, img_size=256)
    fp0 = baseline_fingerprint(**base)
    assert baseline_fingerprint(**base) == fp0
    assert baseline_fingerprint(**{**base, "sha256_train": "xyz"}) != fp0
    assert baseline_fingerprint(**{**base, "chunk_size": 25}) != fp0
    assert baseline_fingerprint(**{**base, "cameras": ["head", "wrist"]}) != fp0
    assert baseline_fingerprint(**{**base, "mu_weighting": MU_TRAJECTORY}) != fp0
    ns.write_text('{"a": 2}', encoding="utf-8")            # 归一化统计变了
    assert baseline_fingerprint(**base) != fp0


def test_store_rejects_fingerprint_mismatch(tmp_path):
    p = tmp_path / "task_baseline.json"
    st = BaselineStore.load(str(p), fingerprint="fp1")
    st.put(build_baseline("t", _chunks(), fingerprint="fp1",
                          aggregate=lambda x: {"mse": 0.5}))
    st.save()
    assert BaselineStore.load(str(p), fingerprint="fp1").get("t") is not None
    other = BaselineStore.load(str(p), fingerprint="fp2")
    assert other.get("t") is None
    assert any("指纹不符" in w for w in other.warnings)


def test_store_roundtrip_preserves_values(tmp_path):
    p = tmp_path / "b.json"
    st = BaselineStore.load(str(p), fingerprint="fp")
    st.put(build_baseline("t", _chunks(), fingerprint="fp", action_keys=["action.x"],
                          aggregate=lambda x: {"mse": 0.25}))
    st.save()
    got = BaselineStore.load(str(p), fingerprint="fp").get("t")
    assert got is not None and got.mse == pytest.approx(0.25)
    assert got.action_keys == ("action.x",)


def test_store_ignores_version_mismatch(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"version": 999, "tasks": {"t": {}}}), encoding="utf-8")
    st = BaselineStore.load(str(p), fingerprint="fp")
    assert st.get("t") is None and any("版本" in w for w in st.warnings)


def test_compute_fixed_baseline_uses_validator_collector():
    """baseline 的 GT 必须来自 `collect_gt_chunks`（与 evaluator 同一条口径路径）。"""

    class _V:
        def __init__(self):
            self.calls = []

        def collect_gt_chunks(self, ids, tag):
            self.calls.append((list(ids), tag))
            return _chunks(), ["action.a", "action.b"]

    v = _V()
    b = compute_fixed_baseline(v, "click_bell", [0, 1, 2], fingerprint="fp",
                               aggregate=lambda x: {"mse": 0.75})
    assert v.calls == [([0, 1, 2], "baseline_click_bell")]
    assert b.mse == pytest.approx(0.75)
    assert b.action_keys == ("action.a", "action.b")
    assert b.n_train_episodes == 3


# --------------------------------------------------------------------------- #
# 6) EvaluatorAdapter：接线正确 + 防泄漏（用桩，无需 torch）
# --------------------------------------------------------------------------- #
def _adapter(tmp_path, payload=None, store=None):
    cat = TaskCatalog.from_manifest(str(_fake_manifest(tmp_path)))
    v = _StubValidator(payload)
    return EvaluatorAdapter(v, cat, store), v, cat


def test_evaluator_calls_evaluate_ids_not_private(tmp_path):
    ad, v, _ = _adapter(tmp_path)
    ad.evaluate_task("click_bell", "val")
    assert len(v.evaluate_ids_calls) == 1
    assert v.private_calls == 0                      # 绝不裸调 _evaluate_ids


def test_evaluator_default_ids_come_from_manifest_split(tmp_path):
    ad, v, cat = _adapter(tmp_path)
    ad.evaluate_task("click_bell", "val")
    ids, tag = v.evaluate_ids_calls[0]
    assert ids == cat.entry("click_bell").ids_for("val")
    assert tag == "al_click_bell_val"


def test_evaluator_rejects_ids_outside_split(tmp_path):
    ad, _, cat = _adapter(tmp_path)
    train_id = cat.entry("click_bell").ids_for("train")[0]
    with pytest.raises(ValueError) as err:
        ad.evaluate_task("click_bell", "val", episode_ids=[train_id])
    assert "不属于" in str(err.value)


def test_evaluator_nmse_uses_fixed_baseline(tmp_path):
    store = BaselineStore.load(str(tmp_path / "b.json"), fingerprint="fp")
    store.put(build_baseline("click_bell", _chunks(), fingerprint="fp",
                             aggregate=lambda x: {"mse": 0.04}))
    ad, _, _ = _adapter(tmp_path, payload={"mse": 0.02, "mae": 0.1, "n": 2, "n_chunks": 4,
                                           "frames": 200, "dims": 14,
                                           "per_traj_mse": [0.01, 0.03],
                                           "per_traj_ids": [10, 11],
                                           "per_traj_frames": [100, 100]}, store=store)
    r = ad.evaluate_task("click_bell", "val")
    assert isinstance(r, EvalResult)
    assert r.baseline_mse == pytest.approx(0.04)
    assert r.nmse == pytest.approx(0.02 / 0.04)
    assert r.n_traj == 2 and r.dims == 14 and r.per_traj_ids == [10, 11]
    assert r.eval_seconds >= 0.0


def test_evaluator_nmse_is_none_without_baseline(tmp_path):
    ad, _, _ = _adapter(tmp_path)
    r = ad.evaluate_task("click_bell", "val")
    assert r.nmse is None and r.baseline_mse is None


def test_evaluator_rejects_empty_ids(tmp_path):
    ad, _, _ = _adapter(tmp_path)
    with pytest.raises(ValueError):
        ad.evaluate_task("click_bell", "val", episode_ids=[])


# --------------------------------------------------------------------------- #
# 7) ports / Backend / TrainerAdapter
# --------------------------------------------------------------------------- #
def test_ports_are_runtime_checkable():
    from lingbotvla.auto_learning import ports

    class _C:
        def names(self): return []
        def entry(self, t): raise KeyError(t)

    class _R:
        def local_to_ref(self, i): ...
        def ref_to_local(self, r): ...
        def episode_map(self): ...

    class _E:
        def evaluate_task(self, task_id, split, episode_ids=None): ...

    class _H:
        def score(self, items): ...

    class _T:
        def train_steps(self, req, num_steps): ...

    assert isinstance(_C(), ports.TaskCatalog)
    assert isinstance(_R(), ports.SampleResolver)
    assert isinstance(_E(), ports.Evaluator)
    assert isinstance(_H(), ports.HardnessScorer)
    assert isinstance(_T(), ports.Trainer)


def test_backend_missing_reports_absent_components():
    b = Backend(catalog=object(), resolver=object())
    assert set(b.missing()) == {"evaluator", "scorer", "trainer"}
    assert Backend(catalog=1, resolver=1, evaluator=1, scorer=1, trainer=1).missing() == []


def test_task_entry_ids_for():
    e = TaskEntry(name="t", train_ids=(1, 2), val_ids=(3,), n_total=3)
    assert e.ids_for("train") == [1, 2] and e.ids_for("val") == [3]
    assert e.n_train == 2 and e.n_val == 1


def test_trainer_adapter_is_explicitly_not_implemented():
    with pytest.raises(NotImplementedError) as err:
        TrainerAdapter()
    assert "B0" in str(err.value) or "未实现" in str(err.value)
    assert "AutoLearnSampler" in TrainerAdapter.minimal_change_plan()


# --------------------------------------------------------------------------- #
# 8) auto_learning 关闭时零副作用（模块级）
# --------------------------------------------------------------------------- #
def test_importing_package_does_not_import_torch():
    """`import lingbotvla.auto_learning` 必须廉价（不拖 torch/lerobot 进来）。"""
    import subprocess

    code = (
        "import sys;"
        "import lingbotvla.auto_learning as al;"
        "heavy=[m for m in ('torch','lerobot','datasets') if m in sys.modules];"
        "print(','.join(heavy))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip() == "", f"包导入不该拉起重依赖，实际拉起了: {out.stdout.strip()}"
