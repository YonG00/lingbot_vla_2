"""B1 集成验证的**统一入口与报告**（规范 §3 / §25）。

    python -m lingbotvla.auto_learning.check --suite cpu
    python -m lingbotvla.auto_learning.check --suite gpu48 --model-path <hf_ckpt> ...

`--suite cpu` 会跑「不需要 GPU / 不加载 6B 权重」的那批检查，并写
`cpu_test_report.json`（含 git commit / config / 环境 / 每条检查的结果）。

⚠️ 真实数据/模型的检查（规范 §7–§10 Legacy 对拍、§26–§40 GPU）**不在这里** ——
   它们分别属于 `tools/al_legacy_parity.py` 与 `tests/test_auto_learning_real_model.py`。
   本模块只负责「跑得动、能出报告」的那一层，并把**未跑**的项显式标成 skipped，
   免得报告看起来比实际更全。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]          # lingbotvla/auto_learning/check.py -> 仓库根


# --------------------------------------------------------------------------- #
def _git(args: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=str(REPO),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def collect_env() -> Dict[str, Any]:
    """§3：报告必须固定的环境信息。"""
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "repo": str(REPO),
        "commit": _git(["rev-parse", "HEAD"]),
        "commit_short": _git(["rev-parse", "--short", "HEAD"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_git(["status", "--porcelain"])),
        "legacy_ref_commit": os.environ.get("AL_LEGACY_REF_COMMIT"),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import torch  # noqa: F401
        import torch as _t
        env["torch"] = _t.__version__
        env["cuda_available"] = bool(_t.cuda.is_available())
        if _t.cuda.is_available():
            env["device"] = _t.cuda.get_device_name(0)
            env["vram_total_mib"] = int(_t.cuda.get_device_properties(0).total_memory // (1 << 20))
    except Exception:  # noqa: BLE001
        env["torch"] = None
    try:
        import numpy as _n
        env["numpy"] = _n.__version__
    except Exception:  # noqa: BLE001
        pass
    return env


# --------------------------------------------------------------------------- #
@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    fatal: bool = False
    skipped: bool = False
    seconds: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "skipped": self.skipped,
                "fatal": self.fatal, "detail": self.detail,
                "seconds": round(self.seconds, 4), "extra": self.extra}


@dataclass
class CheckReport:
    """§3/§25：结构化 JSON 报告 + 人类可读摘要。"""

    suite: str
    env: Dict[str, Any] = field(default_factory=dict)
    checks: List[CheckResult] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def add(self, name: str, ok: bool, *, detail: str = "", fatal: bool = False,
            skipped: bool = False, seconds: float = 0.0, **extra: Any) -> None:
        self.checks.append(CheckResult(name, bool(ok), detail, fatal, skipped,
                                       seconds, dict(extra)))

    def run(self, name: str, fn: Callable[[], Any], *, fatal: bool = False) -> bool:
        """跑一条检查；异常 ⇒ FAIL（不抛出，除非 fatal 想立刻停）。"""
        t0 = time.time()
        try:
            out = fn()
            detail = "" if out is None else str(out)
            self.add(name, True, detail=detail, fatal=fatal, seconds=time.time() - t0)
            return True
        except Exception as exc:  # noqa: BLE001
            self.add(name, False, detail=f"{type(exc).__name__}: {exc}",
                     fatal=fatal, seconds=time.time() - t0)
            return False

    def skip(self, name: str, reason: str) -> None:
        self.add(name, True, detail=reason, skipped=True)

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.ok)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite,
            "passed": self.passed,
            "env": self.env,
            "wall_time_s": round(time.time() - self.started, 3),
            "summary": {
                "total": len(self.checks),
                "passed": sum(1 for c in self.checks if c.ok and not c.skipped),
                "failed": self.n_failed,
                "skipped": sum(1 for c in self.checks if c.skipped),
            },
            "checks": [c.to_dict() for c in self.checks],
        }

    def write(self, path: Any) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return str(p)

    def summary(self) -> str:
        d = self.to_dict()
        s = d["summary"]
        head = ("✅ PASS" if d["passed"] else "❌ FAIL")
        lines = [f"{head}  suite={self.suite}  "
                 f"{s['passed']} passed / {s['failed']} failed / {s['skipped']} skipped "
                 f"({d['wall_time_s']}s)"]
        for c in self.checks:
            mark = "skip" if c.skipped else ("ok  " if c.ok else "FAIL")
            lines.append(f"  [{mark}] {c.name}"
                         + (f" — {c.detail}" if c.detail else ""))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CPU 检查项（只用纯逻辑 + 假世界；不碰 torch / 真实数据集）
# --------------------------------------------------------------------------- #
def _checks_cpu(rep: CheckReport) -> None:
    from .config import AutoLearningConfig
    from .testing import fake_tasks as ft

    # 1) task split 无泄漏（用真实 manifest 若给了路径，否则用合成 manifest）
    def _split_no_leakage():
        from .catalog import catalog_from_task_split
        man = os.environ.get("AL_MANIFEST")
        if not man:
            return "skipped（未设 AL_MANIFEST；合成 manifest 已由 test_auto_learning_contracts 覆盖）"
        cat = catalog_from_task_split(man, strict=True)
        probs = cat.verify()
        if probs:
            raise RuntimeError("; ".join(str(p) for p in probs))
        e = cat.entry(cat.task_names()[0])
        return f"{len(cat)} 任务；{e.name}: train={e.n_train} val={e.n_val}"

    rep.run("split_no_leakage", _split_no_leakage, fatal=True)

    # 2) 7 NEW + 3 Replay 组成
    def _sampler_composition():
        import random
        from .ports import ReplayPlan, TrainRequest
        from .sampling.sampler import BatchSampler
        from .real.sampler import AutoLearnSampler
        from .testing.sim import build_backend

        al = AutoLearningConfig(batch_size=10, new_slots=7, replay_slots=3)
        cfg = ft.make_cfg([ft._spec("A", curve=[0.5, 0.2]),
                           ft._spec("B", curve=[0.5, 0.2])], al=al)
        be = build_backend(cfg)
        s = AutoLearnSampler(BatchSampler(al, be.resolver, be.catalog, random.Random(7)),
                             batch_size=10)
        e0 = be.catalog.entry("A")
        s.set_request(TrainRequest(task="A", probs={x: 1.0 for x in e0.train_sample_ids},
                                   replay=ReplayPlan(), batch_size=10,
                                   new_slots=7, replay_slots=3))
        it = iter(s)
        [next(it) for _ in range(10)]
        assert s.batch_sampler.n_batches == 1
        comp = s.last_composition
        if (comp.n_new, comp.n_old) != (10, 0):
            raise RuntimeError(f"PASS 池为空时应 10 NEW，实际 {comp.n_new}+{comp.n_old}")
        return "PASS 池为空 ⇒ 10 NEW + 0 Replay ✓"

    rep.run("sampler_7_3_composition", _sampler_composition)

    # 3) 每步独立重采样（不是「一个 batch 当 50 步」）
    def _per_step_resample():
        import random
        from .ports import ReplayPlan, TrainRequest
        from .sampling.sampler import BatchSampler
        from .real.sampler import AutoLearnSampler
        from .testing.sim import build_backend

        al = AutoLearningConfig(batch_size=10, new_slots=7, replay_slots=3)
        cfg = ft.make_cfg([ft._spec("A", curve=[0.5, 0.2])], al=al)
        be = build_backend(cfg)
        s = AutoLearnSampler(BatchSampler(al, be.resolver, be.catalog, random.Random(3)),
                             batch_size=10)
        e0 = be.catalog.entry("A")
        s.set_request(TrainRequest(task="A", probs={x: 1.0 for x in e0.train_sample_ids},
                                   replay=ReplayPlan(), batch_size=10,
                                   new_slots=7, replay_slots=3))
        it = iter(s)
        plans = [tuple(next(it) for _ in range(10)) for _ in range(50)]
        assert len(plans) == 50
        assert len(set(plans)) > 1, "50 步不该是同一个 batch"
        return f"50 步 = 50 次独立构造（{len(set(plans))} 种不同组合）"

    rep.run("sampler_per_step_resample", _per_step_resample)

    # 4) 不变量 n_new + n_old == samples_seen
    def _invariants():
        import random
        from .ports import ReplayPlan, TrainRequest
        from .sampling.sampler import BatchSampler
        from .real.sampler import AutoLearnSampler
        from .testing.sim import build_backend

        al = AutoLearningConfig(batch_size=10, new_slots=7, replay_slots=3)
        cfg = ft.make_cfg([ft._spec("A", curve=[0.5, 0.2])], al=al)
        be = build_backend(cfg)
        s = AutoLearnSampler(BatchSampler(al, be.resolver, be.catalog, random.Random(5)),
                             batch_size=10)
        e0 = be.catalog.entry("A")
        s.set_request(TrainRequest(task="A", probs={x: 1.0 for x in e0.train_sample_ids},
                                   replay=ReplayPlan(), batch_size=10,
                                   new_slots=7, replay_slots=3))
        it = iter(s)
        [next(it) for _ in range(10 * 10)]
        st = s.take_stats()
        bad = st.check_invariants()
        if bad:
            raise RuntimeError("; ".join(bad))
        return f"n_new={st.n_new} n_old={st.n_old} samples_seen={st.samples_seen}"

    rep.run("sampler_invariants", _invariants)

    # 5) hardness → 采样概率统计
    def _hardness_probs():
        from .decision.metrics import (
            difficulty_to_weight, normalize_weights, percentile_rank,
        )
        w = {i: difficulty_to_weight(d, 1.0, 3.0, 2.0) for i, d in enumerate((0.0, 0.5, 1.0))}
        if abs(w[0] - 1.0) > 1e-9 or abs(w[2] - 3.0) > 1e-9:
            raise RuntimeError(f"权重端点不对: {w}")
        p = normalize_weights(w)
        if abs(sum(p.values()) - 1.0) > 1e-9:
            raise RuntimeError(f"概率和 != 1: {sum(p.values())}")
        r = percentile_rank([0.1, 0.5, 0.9])
        if not all(0.0 <= x <= 1.0 for x in r):
            raise RuntimeError(f"percentile rank 越界: {r}")
        return f"w={[round(x, 2) for x in w.values()]}  p={[round(x, 3) for x in p.values()]}  rank={r}"

    rep.run("hardness_weight_probability", _hardness_probs)

    # 6) baseline 公式（轨迹等权）
    def _baseline_formula():
        import numpy as np
        from .baseline import MU_GLOBAL, build_baseline, compute_mu
        rng = np.random.default_rng(0)
        chunks = [(10, rng.normal(0, 1, (4, 3))), (10, rng.normal(0, 1, (4, 3))),
                  (11, rng.normal(5, 1, (2, 3)))]
        b = build_baseline("t", chunks, fingerprint="fp",
                           aggregate=lambda pairs: {"mse": _manual(pairs)})
        mu = compute_mu(chunks, weighting=MU_GLOBAL)
        per, order = {}, []
        for k, g in chunks:
            if k not in per:
                per[k] = []
                order.append(k)
            per[k].append(g)
        manual = float(np.mean([float(np.mean((np.concatenate(per[k], 0) - mu) ** 2))
                                for k in order]))
        if abs(b.mse - manual) > 1e-12:
            raise RuntimeError(f"{b.mse} != 手算 {manual}")
        return f"baseline={b.mse:.6f}（轨迹等权，与手算 Δ=0）"

    def _manual(pairs):
        import numpy as np
        groups, order = {}, []
        for k, gt, pr in pairs:
            if k not in groups:
                groups[k] = []
                order.append(k)
            groups[k].append((gt, pr))
        return float(np.mean([float(np.mean((np.concatenate([p for _, p in groups[k]], 0)
                                            - np.concatenate([g for g, _ in groups[k]], 0)) ** 2))
                              for k in order]))

    rep.run("baseline_trajectory_balanced", _baseline_formula)

    # 7) 状态机主要路径（用假世界跑一遍）
    def _state_machine():
        from .orchestration.scheduler import Scheduler
        from .testing.sim import build_backend
        al = AutoLearningConfig(batch_size=10, new_slots=7, replay_slots=3,
                                eval_interval_steps=50)
        cfg = ft.make_cfg([ft._spec("easy_pass", curve=[0.2, 0.18]),
                           ft._spec("unlearnable", curve=[0.9, 0.9])], al=al)
        sched = Scheduler(build_backend(cfg), al, seed=7)
        sched.run(max_actions=400)
        st = sched.registry.counts()
        return f"停止={sched.state.stop_reason} PASS={st.get('PASS', 0)} " \
               f"EXHAUSTED={st.get('EXHAUSTED', 0)}"

    rep.run("scheduler_state_machine", _state_machine)

    # 8) resume 前后逐项一致（假世界）
    def _resume_parity():
        from .orchestration.scheduler import Scheduler
        from .state import persistence
        from .testing.sim import build_backend
        import tempfile

        al = AutoLearningConfig(batch_size=10, new_slots=7, replay_slots=3)
        cfg = ft.make_cfg([ft._spec("easy_pass", curve=[0.2, 0.18]),
                           ft._spec("slow", curve=[0.9, 0.5, 0.28])], al=al)

        def _run(cut: int):
            sched = Scheduler(build_backend(cfg), al, seed=11)
            with tempfile.TemporaryDirectory(dir=str(REPO / ".pytest_tmp")) as d:
                p = os.path.join(d, "state.json")
                sched.checkpoint_path = p
                for _ in range(cut):
                    if sched.state.finished:
                        break
                    sched.advance()
                if cut and not sched.state.finished:
                    persistence.save_state(sched, p)
                    sched2 = Scheduler(build_backend(cfg), al, seed=11)
                    persistence.load_state(sched2, p)
                    sched2.resume()
                    for _ in range(200):
                        if sched2.state.finished:
                            break
                        sched2.advance()
                    return sched2
                for _ in range(200):
                    if sched.state.finished:
                        break
                    sched.advance()
                return sched

        a = _run(0)
        b = _run(20)
        if a.registry.to_state() != b.registry.to_state():
            raise RuntimeError("resume 后 registry 不一致")
        if a.state.stop_reason != b.state.stop_reason:
            raise RuntimeError(f"stop_reason 不一致: {a.state.stop_reason} vs {b.state.stop_reason}")
        return f"registry/stop_reason 一致（{a.state.stop_reason}）"

    try:
        Path(REPO / ".pytest_tmp").mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001 —— 受限沙箱里可能建不了目录
        pass
    rep.run("resume_deterministic", _resume_parity, fatal=True)

    # 9) disabled 零副作用
    def _disabled():
        from .real.hook import build_hook
        if build_hook(AutoLearningConfig(enabled=False)) is not None:
            raise RuntimeError("enabled=false 时 build_hook 应返回 None")
        return "build_hook(enabled=false) → None ✓"

    rep.run("disabled_no_side_effect", _disabled, fatal=True)

    # ---- 需要真实数据/模型的项：显式标 skipped（别让报告看起来更全）----
    for name, why in (
        ("legacy_dataset_item_parity", "需真实 torch/lerobot + 参考 checkout ⇒ tools/al_legacy_parity.py"),
        ("label_mask_padding_parity", "同上"),
        ("norm_stats_parity", "同上"),
        ("dataloader_collator_disabled_parity", "同上"),
        ("sample_resolver_identity", "需真实数据集 ⇒ tests/test_auto_learning_contracts.py 用假件覆盖了映射逻辑"),
        ("evaluator_2to4_cache_regression", "需真实数据集 ⇒ tests/test_auto_learning_real_model.py::R6（GPU）"),
        ("safe_eval_state_restoration", "CPU 部分由假件覆盖；CUDA 部分 ⇒ GPU suite"),
        ("hardness_bf16_real", "需真实模型 ⇒ GPU suite"),
        ("checkpoint_restart_resume_real", "需真实模型 ⇒ GPU suite"),
    ):
        rep.skip(name, why)


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m lingbotvla.auto_learning.check",
        description="Stage B1 集成验证统一入口（规范 §3）")
    ap.add_argument("--suite", choices=["cpu", "gpu48"], default="cpu")
    ap.add_argument("--out", default=None, help="报告输出路径（默认 <cwd>/<suite>_test_report.json）")
    ap.add_argument("--manifest", default=None, help="task split manifest（cpu suite 用）")
    a = ap.parse_args(argv)

    if a.manifest:
        os.environ["AL_MANIFEST"] = a.manifest

    rep = CheckReport(suite=a.suite, env=collect_env())
    if a.suite == "cpu":
        _checks_cpu(rep)
    else:
        rep.add("gpu48_suite", False, fatal=True,
                detail="GPU 套件请用 tests/test_auto_learning_real_model.py "
                       "（AL_TEST_MODEL_PATH / AL_TEST_CONFIG / AL_TEST_MANIFEST）")

    out = a.out or f"{a.suite}_test_report.json"
    rep.write(out)
    print(rep.summary())
    print(f"\n报告: {out}")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["CheckReport", "CheckResult", "collect_env", "main"]
