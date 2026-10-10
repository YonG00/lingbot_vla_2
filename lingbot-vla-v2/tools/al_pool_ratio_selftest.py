#!/usr/bin/env python
"""CPU 自测：候选池「GMean ratio 一把尺子」筛选（``pool_filter_by_gmean_ratio``）。

    /opt/anaconda3/bin/python tools/al_pool_ratio_selftest.py

**不需要 GPU / 不需要数据集 / 不需要 baseline 权重**。四段：

  A. 用**真工具** ``build_gmean_thresholds.build_thresholds`` 从参考模型逐轨迹 MSE
     造一张 ``stat="geomean"`` 阈值表（= ratio 的分母）。
  B. ratio 三档 + 四个边界值（0.19 / 0.20 / 5.00 / 5.01）。
  C. **开关关闭** ⇒ 缺 baseline 仍然拒绝启动（报错文案与改造前逐字一致）。
  D. **开关打开** ⇒ 缺 baseline 只 warning；整链（Stage A 模拟世界，nmse 全部抹成
     None = 缺 baseline 的真实表现）跑到收工，打印每个任务的 ratio 分档与路线。

退出码 0 = 全部断言通过。
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lingbotvla.auto_learning.baseline import BaselineStore          # noqa: E402
from lingbotvla.auto_learning.config import (                        # noqa: E402
    AutoLearningConfig, DemoConfig, SimConfig, TaskSpecConfig,
)
from lingbotvla.auto_learning.decision import thresholds as TH       # noqa: E402
from lingbotvla.auto_learning.decision.thresholds import (           # noqa: E402
    PassThresholds, active_metric,
)
from lingbotvla.auto_learning.real import build as al_build          # noqa: E402
from lingbotvla.auto_learning.tools.build_gmean_thresholds import (  # noqa: E402
    build_thresholds,
)

OK = 0
FAILED: list = []


def check(cond, label: str, detail: str = "") -> None:
    global OK
    if cond:
        print(f"  ✅ {label}")
    else:
        OK = 1
        FAILED.append(label)
        print(f"  ❌ {label} {detail}")


class _Log:
    def __init__(self):
        self.msgs = []

    def _add(self, msg, *a, **k):
        self.msgs.append(str(msg))

    info_rank0 = info = warning = _add


# --------------------------------------------------------------------------- #
print("\n[A] 用真工具造一张 geomean 阈值表（ratio 的分母）")
store = BaselineStore(path="/tmp/_selftest_baseline.json", config_fingerprint="fp-selftest")
TASKS = ("t0", "t1", "t2")
store.tasks = {t: {"mse": 0.25, "fingerprint": f"sha-{t}"} for t in TASKS}
per_task = {t: [0.001] * 10 for t in TASKS}          # 参考模型逐轨迹 MSE（等值 ⇒ CV=0）
table, diag = build_thresholds(
    per_task, store, multiplier=200.0, min_trajectories=10,
    max_cv=1.5, baseline_cap=0.99, high_cv_policy="null", reference="selftest")
LINE = table.tasks["t0"]
print(f"  阈值表：metric={table.metric!r} stat={table.stat!r} "
      f"tasks={table.tasks} status={diag['t0']['status']}")
check(table.metric == "mse" and table.stat == "geomean", "阈值表是 mse/metric + stat=geomean")
check(abs(LINE - 0.2) < 1e-9, f"参考线 = gmean×200 = {LINE}", f"实际 {LINE}")


def _cfg(*, enabled: bool, **kw) -> AutoLearningConfig:
    al = AutoLearningConfig(
        seed=3, eval_interval_steps=50, min_steps_before_defer=100, pass_nmse=0.30,
        min_lp50=0.05, max_attempts_per_task=2, batch_size=10, new_slots=7, replay_slots=3,
        pass_metric="gmean_mse", pass_thresholds_file="th.json",
        pool_filter_by_gmean_ratio=enabled, **kw)
    al.pass_thresholds = PassThresholds(
        config_fingerprint="fp-selftest", tasks=dict(table.tasks),
        metric="mse", stat="geomean", reference="selftest")
    return al


# --------------------------------------------------------------------------- #
print("\n[B] ratio 分档：0.19 / 0.20 / 5.00 / 5.01（边界含在可练段）")
al_on = _cfg(enabled=True)
al_off = _cfg(enabled=False)
check(active_metric(al_on) == "gmean_mse", "口径 = gmean_mse")
check(TH.pool_filter_enabled(al_on) and not TH.pool_filter_enabled(al_off),
      "开关：ON=True / OFF=False")
for ratio, expected in ((0.19, TH.POOL_RATIO_PASSED),
                        (0.20, TH.POOL_RATIO_TRAINABLE),
                        (5.00, TH.POOL_RATIO_TRAINABLE),
                        (5.01, TH.POOL_RATIO_TOO_HARD)):
    got = TH.gmean_pool_bucket(al_on, "t0", gmean_mse=ratio * LINE)
    check(got == expected, f"ratio={ratio:>4} ⇒ {got}", f"期望 {expected}")
check(TH.gmean_pool_bucket(al_off, "t0", gmean_mse=1e9) == TH.POOL_RATIO_DISABLED,
      "开关关闭 ⇒ disabled（不自行换尺子）")


# --------------------------------------------------------------------------- #
print("\n[C] 开关关闭：缺 baseline ⇒ 仍然拒绝启动（文案不变）")
root = tempfile.mkdtemp(prefix="al_ratio_selftest_", dir=os.getcwd())
try:
    manifest = Path(root) / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    th_path = Path(root) / "th.json"
    th_path.write_text(json.dumps({
        "version": 1, "metric": "mse", "stat": "geomean", "config_fingerprint": "fp",
        "tasks": {t: LINE for t in TASKS}}), encoding="utf-8")


    def _write_yaml(name: str, *, enabled: bool) -> str:
        p = Path(root) / name
        p.write_text(f"""
auto_learning:
  enabled: true
  batch_size: 10
  new_slots: 7
  replay_slots: 3
  eval_interval_steps: 5
  min_steps_before_defer: 10
  defer_retry_steps: 5
  pass_metric: gmean_mse
  pass_thresholds_file: "{th_path}"
  pool_filter_by_gmean_ratio: {str(enabled).lower()}
""", encoding="utf-8")
        return str(p)


    class _Entry:
        name = "t0"
        sha256_train = "sha"
        train_sample_ids = [0, 1]
        train_traj_ids = [0]
        val_traj_ids = [1]
        n_train_trajs = 1

        def ids_for(self, split):
            return [0]


    class _Catalog:
        meta, episodes_per_task, task_order = {}, 1, ["t0"]

        def __contains__(self, n):
            return n == "t0"

        def __len__(self):
            return 1

        def task_names(self):
            return ["t0"]

        def entry(self, n):
            return _Entry()

        def verify(self):
            return []

        def attach_samples(self, resolver):
            return self

        def task_of_episode(self, ep):
            return "t0"


    class _Resolver:
        def __len__(self):
            return 1


    _orig_cat, _orig_res = al_build.catalog_from_task_split, al_build.SampleResolver
    al_build.catalog_from_task_split = lambda *a, **k: _Catalog()
    al_build.SampleResolver = SimpleNamespace(from_dataset=lambda ds, **kw: _Resolver())
    try:
        try:
            al_build.build_auto_learning_parts(
                config_path=_write_yaml("off.yaml", enabled=False),
                manifest_path=str(manifest), train_dataset=object(),
                baseline_path=None, logger=_Log())
            check(False, "开关关闭 + 缺 baseline ⇒ 应当 ValueError")
        except ValueError as exc:
            msg = str(exc)
            check("没有可用的 baseline store" in msg, "开关关闭 + 缺 baseline ⇒ ValueError（旧行为）")
            check("所有任务会被排除出候选池" in msg, "报错文案与改造前一致")

        log = _Log()
        parts = al_build.build_auto_learning_parts(
            config_path=_write_yaml("on.yaml", enabled=True),
            manifest_path=str(manifest), train_dataset=object(),
            baseline_path=None, logger=log)
        check(parts is not None and parts.baseline_store is None,
              "开关打开 + 缺 baseline ⇒ 正常启动（不再拒绝）")
        check(any("不拒绝启动" in m for m in log.msgs), "并且打了一行 warning")

        # 指纹守卫：关闭 fail-fast / 打开只 warning
        ns = SimpleNamespace()
        ns.data = SimpleNamespace(train_path="/data/x.txt", norm_stats_file=None,
                                  cameras=["c"], joints=["j"], img_size=256, chunk_size=None)
        ns.train = SimpleNamespace(chunk_size=50)
        bad_store = BaselineStore(path="/tmp/x.json", config_fingerprint="deadbeef")
        fake_parts = SimpleNamespace(baseline_store=bad_store)
        try:
            al_build._verify_baseline_fingerprint(
                fake_parts, _cfg(enabled=False), ns, None, _Log())
            check(False, "开关关闭 + 指纹不一致 ⇒ 应当 RuntimeError")
        except RuntimeError as exc:
            check("指纹" in str(exc), "开关关闭 + 指纹不一致 ⇒ RuntimeError（旧行为）")
        log2 = _Log()
        al_build._verify_baseline_fingerprint(fake_parts, _cfg(enabled=True), ns, None, log2)
        check(any("只 warning" in m for m in log2.msgs), "开关打开 + 指纹不一致 ⇒ 只 warning")
    finally:
        al_build.catalog_from_task_split, al_build.SampleResolver = _orig_cat, _orig_res
finally:
    shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- #
print("\n[D] 开关打开 + 全程 nmse=None（缺 baseline）⇒ 整链跑通、按 ratio 分档选任务")
from lingbotvla.auto_learning.testing.sim import build_scheduler       # noqa: E402


class _NoBaselineEvaluator:
    """外层 evaluator 的 nmse 全抹成 None —— 等价于"没有 task_baseline.json"。"""

    def __init__(self, inner):
        self.inner = inner

    def evaluate(self, task, split, episode_ids=None):
        m = self.inner.evaluate(task, split, episode_ids)
        return dataclasses.replace(m, nmse=None, baseline_mse=0.0)


def _sim_cfg(lines) -> DemoConfig:
    tasks = [TaskSpecConfig(name=f"t{i}", curve=[0.90, 0.50, 0.25], nmse0=0.90,
                            nmse_floor=0.20, learn_k=1.0, forget_rate=0.05,
                            degrade_slope=0.10, eval_noise=0.03, baseline_mse=0.25,
                            n_train_trajs=40, n_val_trajs=10, samples_per_traj=77)
             for i in range(len(lines))]
    al = _cfg(enabled=True)
    al.pass_thresholds = PassThresholds(
        config_fingerprint="fp-selftest", tasks=dict(lines),
        metric="mse", stat="geomean", reference="selftest")
    return DemoConfig(auto_learning=al, sim=SimConfig(seed=3, steps_per_unit=50, tasks=tasks))


# 参考线刻意配成三档：t0 可练（1 < ratio < 5）/ t1 太难（ratio ≫ 5）/
# t2 已过（远低于线 ⇒ bootstrap 直接 PASS）
lines = {"t0": 0.08, "t1": 0.0005, "t2": 50.0}
cfg = _sim_cfg(lines)
sched = build_scheduler(cfg)
sched.evaluator = _NoBaselineEvaluator(sched.evaluator)
sched.run(max_actions=2000)

buckets = {}
for rec in sched.registry:
    bucket, ratio = sched._pool_bucket_and_ratio(rec.task_name, rec.scout_gmean_mse)
    buckets[rec.task_name] = (bucket, None if ratio is None else round(ratio, 4),
                              rec.status, rec.attempt_count)
print(f"  finished={sched.state.finished} stop_reason={sched.state.stop_reason!r}")
for name, (bucket, ratio, status, attempts) in sorted(buckets.items()):
    print(f"    {name}: ratio={ratio} bucket={bucket:<10s} status={status:<9s} attempts={attempts}")

check(sched.state.finished, "整链能跑到收工（缺 baseline 不会卡死）")
check(buckets["t0"][0] == TH.POOL_RATIO_TRAINABLE, "t0 落在可练段")
check(buckets["t1"][0] == TH.POOL_RATIO_TOO_HARD and buckets["t1"][3] == 0,
      "t1 太难 ⇒ 一次都没被训练（不消耗 attempt）")
check(buckets["t2"][2] == "PASS", "t2 远低于参考线 ⇒ 直接 PASS（已过，不进候选池）")
check(sched.registry.get("t0").attempt_count >= 1, "t0 真的被训练了（可练段生效）")
check(any(e.get("task") == "t0" and e.get("action") == "train_unit"
          for e in sched.events), "事件流里有 t0 的 train_unit（评测/判定都在跑）")
# 缺 baseline 时 LP 必须用 GMean 算出来，否则 attempt 永不收尾
check(sched.registry.get("t0").lp50 is not None,
      "缺 baseline 时 lp50 仍由 GMean 算出（不会无限 CONTINUE）")
check(not any(e.get("task") == "t1" and e.get("action") == "select"
              for e in sched.events), "t1 从未被 select（太难档位真的挡住了）")

print("\n" + ("=" * 70))
if OK == 0:
    print("✅ 全部通过：候选池 ratio 筛选 + baseline 降级（开关关闭时行为不变）")
else:
    print(f"❌ 有 {len(FAILED)} 项未通过：{FAILED}")
sys.exit(OK)
