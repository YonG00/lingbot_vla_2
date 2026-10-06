"""Stage B1 —— 集成验证规范 v0.1 里**尚未覆盖**的那几项（无卡部分）。

对应规范章节：
  §14  HardnessScorer **无卡** mock 语义（fixed noise/time → 同分；dtype 策略；不改输入）
  §18  PASS sampling snapshot（保存 / 覆盖 / fallback）
  §23  旧 checkpoint 兼容性（缺 auto_learning key 不能破坏 legacy resume）
  §39  训练关键字段运行时审计（schema audit helper）
  §40  Loss / Label sanity fail-fast

（§7–§10 的 Legacy Dataset 对拍需要真实 torch/lerobot + 参考 checkout ⇒ 见
  `tools/al_legacy_parity.py`，不在这里。）

    python -m pytest tests/test_auto_learning_integration.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lingbotvla.auto_learning.config import AutoLearningConfig        # noqa: E402
from lingbotvla.auto_learning.hardness import (                       # noqa: E402
    HardnessScorer, find_loss_dict,
)
from lingbotvla.auto_learning.state.registry import TaskRegistry      # noqa: E402
from lingbotvla.auto_learning.testing import fake_tasks as ft         # noqa: E402
from lingbotvla.auto_learning.types import TaskStatus                 # noqa: E402

_TMP_ROOT = REPO / ".pytest_tmp"


@pytest.fixture
def tmp_path():
    import shutil
    import uuid

    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    d = _TMP_ROOT / f"ali_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# §14  HardnessScorer 无卡 mock 语义
# --------------------------------------------------------------------------- #
class _FakeParam:
    """最小「参数」替身（只用 dtype/device）。"""

    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device


class _FakeModel:
    """假模型：不 import torch 也能验证接口语义。

    记录 `noise`/`time` 的 dtype/device，返回 `batch_mean_losses`。
    """

    def __init__(self, *, param_dtype="bf16", device="cpu", action_fp32=False):
        self.config = type("C", (), {"action_fp32": action_fp32, "loss_type": "fm",
                                     "n_action_steps": 50, "max_action_dim": 55})()
        self._p = _FakeParam(param_dtype, device)
        self.training = True
        self.calls = []

    def parameters(self):
        yield self._p

    def modules(self):
        yield self

    def eval(self):
        self.training = False
        return self

    def __call__(self, **kw):
        self.calls.append(kw)
        n = int(kw["actions"].shape[0])
        return (0.0,) * 6 + ({"batch_mean_losses": _FakeArr([0.1] * n)},)


class _FakeArr:
    """极小的「类 ndarray」：只支持 .detach().float().cpu().numpy() 链。"""

    def __init__(self, vals):
        self.vals = list(vals)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return _NP(self.vals)

    def reshape(self, *a):
        return self


class _NP:
    def __init__(self, vals):
        self.vals = vals

    def __array__(self, dtype=None):
        import numpy as np
        return np.asarray(self.vals, dtype=dtype)


def test_find_loss_dict_scans_instead_of_hardcoding_index():
    """`forward` 返回 11 元组，按下标硬编码很脆 ⇒ 实现是「扫描第一个含目标键的 dict」。"""
    out = (1, 2, 3, 4, 5, 6, {"batch_mean_losses": "X"}, 7, 8, 9, 10)
    assert find_loss_dict(out) == {"batch_mean_losses": "X"}
    assert find_loss_dict({"batch_mean_losses": 1}) == {"batch_mean_losses": 1}
    assert find_loss_dict((1, 2, 3)) is None


def test_hardness_requires_joint_mask():
    """缺 `joint_mask` ⇒ 会落到未掩码分支（口径不同）⇒ 默认 fail-fast。"""
    pytest.importorskip("torch", reason="`score()` 走 default_collate，需要 torch")
    sc = HardnessScorer(_FakeModel(), require_joint_mask=True)
    with pytest.raises(ValueError) as err:
        sc.score([{"actions": _FakeArr([0.0])}])
    assert "joint_mask" in str(err.value)


def test_hardness_rejects_empty_items():
    pytest.importorskip("torch", reason="`score()` 走 default_collate，需要 torch")
    with pytest.raises(ValueError):
        HardnessScorer(_FakeModel()).score([])


def test_hardness_forward_dtype_follows_action_fp32_flag():
    """`action_fp32=True` ⇒ float32；否则 ⇒ 模型参数 dtype（单卡训练常见 bf16）。"""
    torch = pytest.importorskip("torch", reason="dtype 断言需要 torch")
    fp32 = HardnessScorer(_FakeModel(param_dtype=torch.bfloat16, action_fp32=True))
    assert fp32.forward_dtype() == torch.float32
    bf16 = HardnessScorer(_FakeModel(param_dtype=torch.bfloat16, action_fp32=False))
    assert bf16.forward_dtype() == torch.bfloat16


def test_hardness_device_inferred_from_model():
    """device 从模型推断，不写死 cuda（规范 §30 要求显式确认 device）。"""
    m = _FakeModel(device="cpu")
    assert HardnessScorer(m).device == "cpu"


def test_hardness_clears_and_restores_align_and_grid_cache():
    pytest.importorskip("torch", reason="`score()` 走 default_collate，需要 torch")
    """§39/§30：hardness 期间对 `align_params` / 视觉网格缓存的临时改动**必须还原**。

    （GPU 实测：不还原会让后续 micro=1 训练炸 `split_with_sizes`。）
    """
    m = _FakeModel()
    m.config.align_params = {"depth": {"depth_loss_weight": 1.0}}
    before_align = dict(m.config.align_params)
    before_training = m.training

    class _Arr:
        shape = (1, 50, 55)
        dtype = None

        def to(self, *a, **k):
            return self

    try:
        HardnessScorer(m).score([{"actions": _Arr(), "joint_mask": _Arr()}])
    except Exception:  # noqa: BLE001 —— 假模型只验证「finally 有没有还原」
        pass
    assert m.config.align_params == before_align, "align_params 没还原"
    assert m.training is before_training, "模块 training 标志没还原"


# --------------------------------------------------------------------------- #
# §18  PASS sampling snapshot
# --------------------------------------------------------------------------- #
def _registry():
    al = AutoLearningConfig(batch_size=10, new_slots=7, replay_slots=3)
    cfg = ft.make_cfg([ft._spec("A", curve=[0.5, 0.2]),
                       ft._spec("B", curve=[0.5, 0.2])], al=al)
    from lingbotvla.auto_learning.testing.sim import build_backend
    cat = build_backend(cfg).catalog
    return TaskRegistry.from_catalog(cat, al), al, cat


def test_pass_snapshot_roundtrip():
    """§18：PASS 时保存采样 snapshot；存进 state 后能读回来。"""
    reg, al, cat = _registry()
    rec = reg.get("A")
    rec.sample_probs = {0: 0.6, 1: 0.4}
    rec.set_status(TaskStatus.PASS, reason="test")

    st = reg.to_state()
    reg2 = TaskRegistry.from_catalog(cat, al)
    reg2.load_state(st)
    assert reg2.get("A").sample_probs == {0: 0.6, 1: 0.4}, "PASS snapshot 必须随状态持久化"


def test_initial_auto_pass_has_no_snapshot():
    """§18：初始 auto-PASS 的 task 没有 snapshot ⇒ replay 侧要有明确 fallback（uniform）。"""
    reg, _, _ = _registry()
    rec = reg.get("B")
    assert not getattr(rec, "sample_probs", None), "没训过的 task 不该凭空有 snapshot"


def test_repass_overwrites_snapshot():
    """§18：reopen → 重训 → re-PASS 后 snapshot 被**新版本覆盖**。"""
    reg, al2, cat2 = _registry()
    rec = reg.get("A")
    rec.sample_probs = {0: 1.0}
    rec.set_status(TaskStatus.PASS, reason="first")
    first = reg.to_state()

    # reopen → 重训 → re-PASS ⇒ 新分布必须**覆盖**旧快照（不是并存）
    rec.set_status(TaskStatus.CANDIDATE, reason="forgotten")
    rec.sample_probs = {1: 1.0}
    rec.set_status(TaskStatus.PASS, reason="repass")
    second = reg.to_state()

    reg2 = TaskRegistry.from_catalog(cat2, al2)
    reg2.load_state(second)
    assert reg2.get("A").sample_probs == {1: 1.0}, "re-PASS 后快照应被新版本覆盖"
    assert json.dumps(first) != json.dumps(second)


# --------------------------------------------------------------------------- #
# §23  旧 checkpoint 兼容性
# --------------------------------------------------------------------------- #
def test_legacy_extra_state_without_auto_learning_key_is_fine():
    """§23：旧 checkpoint 的 `extra_state` 没有 `auto_learning` key ⇒
    `auto_learning=false` 时**必须能正常加载**，不能报错。"""
    legacy = {"global_step": 500, "lr_scheduler": {}, "train_dataloader": {}}
    assert "auto_learning" not in legacy
    # 关闭时我们根本不读这个 key（`build_hook` 返回 None）⇒ 天然兼容
    from lingbotvla.auto_learning.real.hook import build_hook
    assert build_hook(AutoLearningConfig(enabled=False)) is None


def test_enabled_without_state_raises_rather_than_guessing():
    """§23：`auto_learning=true` 但 checkpoint 里没有状态 ⇒ **显式报错**，
    不允许静默「从零开始」。"""
    from lingbotvla.auto_learning.real.hook import build_hook
    from lingbotvla.auto_learning.testing.sim import build_backend
    from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
    from lingbotvla.auto_learning.real.sampler import AutoLearnSampler
    from lingbotvla.auto_learning.sampling.sampler import BatchSampler
    import random

    cfg = ft.make_cfg([ft._spec("A", curve=[0.5, 0.2])],
                      al=AutoLearningConfig(batch_size=10, new_slots=7, replay_slots=3))
    be = build_backend(cfg)
    sched = Scheduler(be, cfg.auto_learning, seed=1)
    sampler = AutoLearnSampler(BatchSampler(cfg.auto_learning, be.resolver, be.catalog,
                                            random.Random(1)), batch_size=10)
    hook = build_hook(cfg.auto_learning, scheduler=sched, sampler=sampler)
    with pytest.raises(ValueError):
        hook.load_extra_state({})                       # 空状态
    with pytest.raises(ValueError):
        hook.load_extra_state({"version": 999})          # 版本不符


# --------------------------------------------------------------------------- #
# §39 / §40  运行时审计 + sanity fail-fast
# --------------------------------------------------------------------------- #
def test_schema_audit_reports_shapes_and_flags():
    """§39：前几个 step 输出一次 batch schema 审计（shape/dtype/valid count）。"""
    from lingbotvla.auto_learning.real.sanity import audit_batch_schema

    class _T:
        def __init__(self, shape, dtype="float32", n_valid=None):
            self.shape = tuple(shape)
            self.dtype = dtype
            self._n = n_valid

        def numel(self):
            p = 1
            for d in self.shape:
                p *= d
            return p

        def __array__(self, dtype=None):
            import numpy as np
            a = np.ones(self.shape, dtype="float32")
            if self._n is not None:
                a.reshape(-1)[self._n:] = 0
            return a

    batch = {"actions": _T((10, 50, 55)), "state": _T((10, 55)),
             "joint_mask": _T((10, 50, 55), n_valid=10 * 50 * 20),
             "action_is_pad": _T((10, 50), n_valid=10 * 40)}
    rep = audit_batch_schema(batch)
    assert rep["actions"]["shape"] == (10, 50, 55)
    assert rep["joint_mask"]["valid"] == 10 * 50 * 20
    assert rep["action_is_pad"]["valid"] == 10 * 40


def test_sanity_flags_all_padding_and_empty_mask():
    """§40：全 padding / mask 全 0 / baseline≈0 / 长度不一致 ⇒ fail-fast。"""
    from lingbotvla.auto_learning.real.sanity import (
        assert_batch_sane, assert_loss_sane, assert_metrics_sane,
    )

    class _Z:
        def __init__(self, shape):
            self.shape = tuple(shape)

        def numel(self):
            p = 1
            for d in self.shape:
                p *= d
            return p

        def __array__(self, dtype=None):
            import numpy as np
            return np.zeros(self.shape, dtype="float32")

    with pytest.raises(RuntimeError) as e1:
        assert_batch_sane({"joint_mask": _Z((10, 50, 55))})
    assert "mask" in str(e1.value) or "joint_mask" in str(e1.value)

    with pytest.raises(RuntimeError):
        assert_loss_sane(float("nan"), {"batch_mean_losses": _Z((10,))})
    with pytest.raises(RuntimeError):
        assert_metrics_sane(mse=0.5, baseline_mse=0.0)      # 分母≈0
    with pytest.raises(RuntimeError):
        assert_metrics_sane(mse=0.5, baseline_mse=0.2, gt_frames=50, pred_frames=16)


def test_sanity_passes_on_healthy_inputs():
    from lingbotvla.auto_learning.real.sanity import assert_loss_sane, assert_metrics_sane

    class _One:
        shape = (4,)

        def numel(self):
            return 4

        def __array__(self, dtype=None):
            import numpy as np
            return np.ones((4,), dtype="float32")

    assert_loss_sane(0.3, {"batch_mean_losses": _One()})
    assert_metrics_sane(mse=0.5, baseline_mse=0.2, gt_frames=50, pred_frames=50)


# --------------------------------------------------------------------------- #
# §3 / §25  报告结构
# --------------------------------------------------------------------------- #
def test_cpu_report_shape(tmp_path):
    """§3/§25：报告必须记录 commit / config / 环境 / 结果。"""
    from lingbotvla.auto_learning.check import CheckReport, collect_env

    env = collect_env()
    for k in ("python", "platform", "cwd"):
        assert k in env
    rep = CheckReport(suite="cpu", env=env)
    rep.add("split_no_leakage", True, detail="ok")
    rep.add("sample_resolver_identity", False, detail="mismatch", fatal=True)
    d = rep.to_dict()
    assert d["suite"] == "cpu" and d["passed"] is False
    assert len(d["checks"]) == 2
    p = rep.write(tmp_path / "cpu_test_report.json")
    assert json.loads(Path(p).read_text(encoding="utf-8"))["suite"] == "cpu"
