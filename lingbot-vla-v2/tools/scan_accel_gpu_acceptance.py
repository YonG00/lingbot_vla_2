#!/usr/bin/env python3
"""扫描加速 GPU 验收入口（默认 PLAN ONLY；判定子命令为纯 CPU、fail-closed）。

设计（用户 2026-10-09 要求）：
  * `plan`（默认）只打印**开卡后要跑的完整命令**、输出目录、预计耗时与安全停止条件，
    **不碰 GPU、不建目录、不设任何 `*_APPROVED` 环境变量**。
  * `verify-probe`  读取真实模型 probe 报告（`AL_EVAL_BATCH_PROBE_OUT`），判定
    Batch1/2/4 数值一致性 / 显存余量 / 吞吐，并给出 **PASS / BLOCKED / FAIL**。
  * `verify-hardness` 读取两次（或三次）hardness 逐样本报告（`AL_HARDNESS_REPORT_OUT`），
    判定逐样本 Loss 一致性、排序一致性、随机数可复现性与显存守卫。

判定原则：**没有真实 GPU 证据一律 BLOCKED，绝不因"策略已合入"就判 PASS**；
未通过前不得启用 `auto`，也不得声称已获得加速。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROBE_ATOL = 1e-5
PROBE_RTOL = 1e-3
HARD_ATOL = 1e-6
HARD_RTOL = 1e-4
HARD_RESERVE_GIB = 10.0
REQUIRED_PROBE_BATCHES = (2, 4)


# --------------------------------------------------------------------------- #
# 报告读取
# --------------------------------------------------------------------------- #
def load_records(path) -> list:
    """读取 `append_json_record` 写出的报告；格式不符 ⇒ 抛错（不静默当空）。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("records"), list):
        raise ValueError(f"{path}: 报告缺少 records 列表")
    return raw["records"]


def probe_records(path) -> list:
    return [r for r in load_records(path) if r.get("kind") == "eval_batch_probe"]


def hardness_losses(path) -> tuple:
    """把 hardness 报告折成 {sid: loss} 与元信息。"""
    recs = [r for r in load_records(path) if r.get("kind") == "hardness_scan"]
    losses, batches, peaks = {}, set(), []
    for r in recs:
        for sid, v in (r.get("losses") or {}).items():
            losses[int(sid)] = float(v)
        if r.get("batch"):
            batches.add(int(r["batch"]))
        if r.get("peak_free_gib") is not None:
            peaks.append(float(r["peak_free_gib"]))
    return losses, {"n_records": len(recs), "batches": sorted(batches),
                    "min_peak_free_gib": (min(peaks) if peaks else None)}


# --------------------------------------------------------------------------- #
# 判定：Eval Batch Probe
# --------------------------------------------------------------------------- #
def verdict_probe(records: list, *, atol: float = PROBE_ATOL, rtol: float = PROBE_RTOL,
                  reserve_gib: float = HARD_RESERVE_GIB) -> dict:
    """PASS 需：每个 probe 组 parity/safe 通过、且真实达到 Batch2 与 Batch4。"""
    problems, blocked = [], []
    if not records:
        return {"status": "BLOCKED", "reason": "no_eval_batch_probe_records",
                "note": "未取得任何 probe 记录（需 AL_EVAL_BATCH_MODE=probe + AL_EVAL_BATCH_PROBE_OUT）"}
    errors = [r for r in records if r.get("kind") == "eval_batch_probe_error"]
    records = [r for r in records if r.get("kind") == "eval_batch_probe"]
    if errors:
        blocked.append("probe_report_write_failed:" + str(errors[0].get("error"))[:80])
    modes = sorted({str(r.get("mode")) for r in records})
    if modes != ["probe"]:
        problems.append(f"unexpected_mode:{modes}")
    if any(r.get("parity") is not True for r in records):
        problems.append("parity_failed")
    if any(r.get("safe") is not True for r in records):
        problems.append("safety_or_vram_guard_failed")
    if any(r.get("peak_free_gib") is not None and float(r["peak_free_gib"]) < reserve_gib
           for r in records):
        problems.append("peak_free_below_reserve")
    seen = sorted({int(r.get("batch") or 0) for r in records})
    for need in REQUIRED_PROBE_BATCHES:
        if need not in seen:
            blocked.append(f"batch{need}_not_reached")
    speedups = [float(r["speedup"]) for r in records if r.get("speedup") is not None]
    max_speed = max(speedups) if speedups else None
    diffs = [t.get("max_abs_diff") for r in records for t in (r.get("per_traj") or [])
             if t.get("max_abs_diff") is not None]
    status = "FAIL" if problems else ("BLOCKED" if blocked else "PASS")
    return {
        "status": status, "kind": "eval_batch_probe_verdict",
        "problems": problems, "blocked": blocked,
        "batches_probed": seen, "n_records": len(records),
        "max_speedup": max_speed, "max_per_traj_abs_diff": (max(diffs) if diffs else None),
        "atol": atol, "rtol": rtol, "reserve_gib": reserve_gib,
        "speedup_note": ("no_speedup_observed_do_not_enable_auto"
                         if (max_speed is None or max_speed <= 1.0) else "speedup_observed"),
        "note": "probe 只记录影子结果、PASS 仍走串行；未通过本判定不得启用 auto",
    }


# --------------------------------------------------------------------------- #
# 判定：Hardness 批量
# --------------------------------------------------------------------------- #
def _rank(losses: dict) -> list:
    return [sid for sid, _ in sorted(losses.items(), key=lambda kv: (kv[1], kv[0]))]


def verdict_hardness(batch_ref: dict, batch_cmp: dict, *, repeat: dict | None = None,
                     atol: float = HARD_ATOL, rtol: float = HARD_RTOL,
                     reserve_gib: float = HARD_RESERVE_GIB) -> dict:
    """PASS 需：逐样本 Loss 一致、排序一致、（可选）重跑逐位一致、显存余量达标。"""
    problems = []
    ref, cmp_ = batch_ref["losses"], batch_cmp["losses"]
    missing = sorted(set(ref) - set(cmp_))
    extra = sorted(set(cmp_) - set(ref))
    if missing or extra:
        problems.append(f"sample_id_mismatch:missing={len(missing)},extra={len(extra)}")
    worst, worst_sid = 0.0, None
    for sid in sorted(set(ref) & set(cmp_)):
        delta = abs(ref[sid] - cmp_[sid])
        if delta > atol + rtol * abs(ref[sid]):
            problems.append(f"loss_mismatch:{sid}")
        if delta > worst:
            worst, worst_sid = delta, sid
    if batch_ref["meta"]["batches"] and batch_cmp["meta"]["batches"]:
        if batch_ref["meta"]["batches"] == batch_cmp["meta"]["batches"]:
            problems.append("both_reports_use_same_batch_size")
    if _rank(ref) != _rank(cmp_):
        problems.append("ordering_changed")
    if repeat is not None:
        rep = repeat["losses"]
        if set(rep) != set(ref):
            problems.append("repeat_sample_id_mismatch")
        elif any(rep[sid] != ref[sid] for sid in ref):
            problems.append("rng_not_reproducible")
    # 显存余量对**每份报告**都必须达标（batch8 与对照各自都要安全）。
    peaks_all = {name: rep["meta"]["min_peak_free_gib"]
                 for name, rep in (("ref", batch_ref), ("cmp", batch_cmp))
                 if rep["meta"]["min_peak_free_gib"] is not None}
    if repeat is not None and repeat["meta"]["min_peak_free_gib"] is not None:
        peaks_all["repeat"] = repeat["meta"]["min_peak_free_gib"]
    peak = batch_ref["meta"]["min_peak_free_gib"]
    if any(v < reserve_gib for v in peaks_all.values()):
        problems.append("peak_free_below_reserve")
    return {
        "status": "FAIL" if problems else "PASS", "kind": "hardness_batch_verdict",
        "problems": problems, "n_samples": len(ref),
        "batches_ref": batch_ref["meta"]["batches"], "batches_cmp": batch_cmp["meta"]["batches"],
        "max_abs_loss_delta": worst, "max_abs_loss_delta_sid": worst_sid,
        "ordering_identical": _rank(ref) == _rank(cmp_),
        "rng_repeat_checked": repeat is not None,
        "min_peak_free_gib": peak, "peaks_all": peaks_all, "reserve_gib": reserve_gib,
        "note": ("默认仍为 fixed/batch8；未通过本判定不得设置 AL_HARDNESS_BATCH_APPROVED=1"
                 " 或启用 auto"),
    }


# --------------------------------------------------------------------------- #
# PLAN ONLY
# --------------------------------------------------------------------------- #
#: run-all 的控制项：一次进程内完成 Eval Batch Probe + Hardness 四阶段（不做 optimizer.step）。
RUN_ALL_STAGE = "ratio-gpu"
RUN_ALL_ENV = {
    "AL_EVAL_BATCH_MODE": "probe",                 # probe 输出恒为串行 ⇒ PASS 不受影响
    "AL_HARDNESS_BATCH_MODE": "fixed",             # 默认 fixed；--include-auto 时才改
    "AL_HARDNESS_FIXED_BATCH": "8",                # 显式 Batch8 基线
    "AL_HARDNESS_REPLAY_BATCH": "1",               # 同进程内 Batch1 对照
    "AL_HARDNESS_REPEAT": "1",                     # 同进程内 Batch8 复测（RNG 可复现）
}


def build_run_all_env(out_dir, *, include_auto: bool = False, base_env=None) -> dict:
    """构造 **一次** 运行的子进程环境：两个报告都落盘、AUTO 仅测试用且只作用于该子进程。"""
    env = dict(base_env or os.environ)
    env.update(RUN_ALL_ENV)
    env["AL_EVAL_BATCH_PROBE_OUT"] = str(Path(out_dir) / "eval_probe.json")
    env["AL_HARDNESS_REPORT_OUT"] = str(Path(out_dir) / "hardness_parity.json")
    # 验收专用：强制覆盖 Batch1/2/4（speedup 不达标也继续数值验收；parity/显存/失败保护照旧）。
    env['AL_EVAL_BATCH_FORCE_COVERAGE'] = '1'
    # 零步 Hardness 自检（仅本次验收子进程；正式训练不会设这两个变量）。
    env['AL_HARDNESS_SELFTEST_IDS'] = '9'
    if include_auto:
        # ⚠️ 仅本次子进程内生效；不写任何配置、不影响其它进程（PLAN/summary 会记录该事实）。
        env["AL_HARDNESS_BATCH_MODE"] = "auto"
        env["AL_HARDNESS_BATCH_APPROVED"] = "1"
    else:
        env.pop("AL_HARDNESS_BATCH_APPROVED", None)
    return env


#: 验收专用 AL 配置（Scout 4 条轨迹 ⇒ 能真实成组到 Batch4；非正式配置）。
ACCEPTANCE_AL_CONFIG = "configs/auto_learning/acceptance_scan_accel_2task.yaml"


def run_all_command(out_dir, *, python="python", steps: int = 3, include_auto: bool = False,
                    al_config: str = ACCEPTANCE_AL_CONFIG) -> list:
    """复用**已有** smoke 入口；Bootstrap 全 PASS ⇒ 零训练步（不执行 optimizer.step）。"""
    return [python, "tools/gpu96_acceptance.py", RUN_ALL_STAGE, "--micro", "24", "--gas", "1",
            "--target-total-passed-tasks", "2", "--max-named-tasks", "2",
            "--steps", str(steps), "--al-config", al_config, "--execute"]


PLAN_TEXT = """\
扫描加速 GPU 验收（**仅计划**；本命令不碰 GPU、不建目录、不设任何 *_APPROVED）

公共设置
  cd /data/code/lingbot-vla-v2
  PY=/data/miniconda3/envs/lingbotvla/bin/python
  OUT=/data/outputs/scan_accel_acceptance        # 全新目录（工具会拒绝同名覆盖）

A. Eval Batch Probe（真实模型 Batch1/2/4 对拍；probe 输出恒为串行 ⇒ PASS 不受影响）
  A1 串行基线（可选，复核指标不变）
      AL_EVAL_BATCH_MODE=serial $PY tools/gpu96_acceptance.py ratio-gpu \\
        --micro 24 --gas 1 --target-total-passed-tasks 4 --max-named-tasks 4 --steps 15 --execute
  A2 probe 验收（同一命令 + 报告输出）
      AL_EVAL_BATCH_MODE=probe AL_EVAL_BATCH_PROBE_OUT=$OUT/eval_probe.json \\
        $PY tools/gpu96_acceptance.py ratio-gpu \\
        --micro 24 --gas 1 --target-total-passed-tasks 4 --max-named-tasks 4 --steps 15 --execute
  A3 判定（纯 CPU）
      $PY tools/scan_accel_gpu_acceptance.py verify-probe $OUT/eval_probe.json
  判定门槛：每个 probe 组 parity/safe 必须为真、peak_free ≥ 10 GiB、且必须真实达到
            Batch2 **与** Batch4（否则 BLOCKED，不得声称多轨迹可用）

B. Hardness 逐样本对拍（Batch8 vs Batch1 + 重跑一致性）
  B1 AL_HARDNESS_BATCH_MODE=fixed AL_HARDNESS_FIXED_BATCH=8 \\
     AL_HARDNESS_REPORT_OUT=$OUT/hardness_b8.json $PY tools/gpu96_acceptance.py ratio-gpu ... --execute
  B2 同上但 AL_HARDNESS_FIXED_BATCH=1 且输出 $OUT/hardness_b1.json
  B3 再跑一次 B1（输出 $OUT/hardness_b8_repeat.json）验随机数可复现
  B4 判定（纯 CPU）
      $PY tools/scan_accel_gpu_acceptance.py verify-hardness \\
        $OUT/hardness_b8.json $OUT/hardness_b1.json --repeat $OUT/hardness_b8_repeat.json
  判定门槛：逐样本 Loss 一致（atol 1e-6 / rtol 1e-4）、候选排序完全一致、
            重跑逐位一致、peak_free ≥ 10 GiB

C. 50-task GMean200 启动预检（无卡，可先做）
  $PY tools/gmean50_preflight.py --config configs/auto_learning/experiment_50task_gmean200.yaml \\
    --thresholds /data/eval_results/open_loop/ref50k/pass_thresholds_gmean200_warn.json \\
    --baseline /data/train/task_splits_50/task_baseline.json --expected-tasks 50
  ⚠️ target=4 不变；若 Bootstrap 初始 PASS ≥ 4，训练会**零步结束**（见 preflight 的 zero_step_risk）

预计 GPU 耗时（单卡 96G，含模型加载）
  A1+A2 约 6–10 min；B1–B3 约 8–15 min；C 无卡。合计约 **15–25 min**（不含排队）

D. 推荐路径：**一次加载、一个进程**完成 Eval Batch + Hardness（run-all）
  $PY tools/scan_accel_gpu_acceptance.py run-all --out-dir $OUT            # PLAN ONLY
  $PY tools/scan_accel_gpu_acceptance.py run-all --out-dir $OUT --execute  # 真正执行（一次）
  该命令内部：① 复用已有 `tools/gpu96_acceptance.py ratio-gpu`（**不新增推理实现**）；
  ② 用 2 个"易任务"配置使 Bootstrap 全部 PASS ⇒ **零训练步**（不执行 optimizer.step）；
  ③ 同时打开 `AL_EVAL_BATCH_MODE=probe`、`AL_EVAL_BATCH_PROBE_OUT`、`AL_HARDNESS_REPORT_OUT`、
     `AL_HARDNESS_REPLAY_BATCH=1`、`AL_HARDNESS_REPEAT=1`；④ 结束后自动跑两个 verify 并写 summary.json。
  加 `--include-auto` 才会在**该子进程内**临时设 `AL_HARDNESS_BATCH_MODE=auto` 与
  `AL_HARDNESS_BATCH_APPROVED=1`（仅测试决策/安全行为；**不写配置、不影响其它进程**，summary 会如实记录）。

安全停止条件（任一命中即停止该专项并保留现场）
  1) parity=False 或 safe=False（数值/显存守卫失败）→ 立即停，不得启用 auto
  2) peak_free < 10 GiB → 停
  3) CUDA OOM / 非有限 loss / 非有限 action → 停
  4) 报告缺失或 sample_id 不一致 → 停并查证据口径
  5) Hardness 排序变化或重跑不一致 → 停
  禁止：设置 AL_EVAL_BATCH_APPROVED / AL_HARDNESS_BATCH_APPROVED、启用 auto、声称加速
"""


def cmd_run_all(a: argparse.Namespace) -> int:
    """PLAN ONLY（默认）或**一次**执行两项验收并汇总。"""
    out = Path(a.out_dir)
    env = build_run_all_env(out, include_auto=a.include_auto)
    cmd = run_all_command(out, python=a.python, steps=a.steps, include_auto=a.include_auto)
    print("RUN-ALL（一次进程内完成 Eval Batch Probe + Hardness 四阶段）")
    print("  command :", " ".join(cmd))
    print("  out_dir :", out)
    print("  env     :", json.dumps({k: env[k] for k in sorted(env) if k.startswith("AL_")},
                                   ensure_ascii=False))
    print("  auto    :", "test-only (子进程内 AL_HARDNESS_BATCH_APPROVED=1，不写配置)"
          if a.include_auto else "off（未设任何 *_APPROVED）")
    if not a.execute:
        print("PLAN ONLY：加 --execute 才会真正使用 GPU；不覆盖已存在的输出目录。")
        return 0
    if out.exists() and any(out.iterdir()):
        print(f"REFUSED：输出目录非空，拒绝覆盖：{out}")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    import subprocess
    log = out / "run_all.log"
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(Path(__file__).resolve().parents[1]))
    summary = {"kind": "scan_accel_run_all", "returncode": rc, "out_dir": str(out),
               "include_auto": bool(a.include_auto), "steps_budget": int(a.steps),
               "note": "optimizer.step 不应发生（Bootstrap 全 PASS ⇒ 零训练步）；见 summary 证据"}
    vp = verdict_probe(probe_records(out / "eval_probe.json")) if (out / "eval_probe.json").exists() else \
        {"status": "BLOCKED", "reason": "eval_probe.json 缺失"}
    summary["eval_probe"] = vp
    if (out / "hardness_parity.json").exists():
        recs = load_records(out / "hardness_parity.json")
        phases = sorted({r.get("phase") for r in recs if r.get("kind") == "hardness_scan"})
        summary["hardness_parity"] = {"status": "PASS" if len(phases) >= 2 else "BLOCKED",
                                      "phases": phases, "note": "四阶段需 main/replay_batch1/repeat"}
    else:
        summary["hardness_parity"] = {"status": "BLOCKED", "reason": "hardness_parity.json 缺失"}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if rc == 0 and summary["eval_probe"]["status"] == "PASS" else 2


def cmd_plan(_: argparse.Namespace) -> int:
    print(PLAN_TEXT)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="扫描加速 GPU 验收（PLAN ONLY + CPU 判定）")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("plan", help="打印 GPU 验收命令/耗时/目录/停止条件（默认）")
    ra = sub.add_parser("run-all", help="一次加载完成 Eval Batch + Hardness（默认 PLAN ONLY）")
    ra.add_argument("--out-dir", default="/data/outputs/scan_accel_acceptance")
    ra.add_argument("--python", default="python")
    ra.add_argument("--steps", type=int, default=3)
    ra.add_argument("--include-auto", action="store_true")
    ra.add_argument("--execute", action="store_true")
    vp = sub.add_parser("verify-probe", help="判定 Eval Batch Probe 报告")
    vp.add_argument("report")
    vp.add_argument("--reserve-gib", type=float, default=HARD_RESERVE_GIB)
    vh = sub.add_parser("verify-hardness", help="判定 hardness 逐样本报告")
    vh.add_argument("batch8")
    vh.add_argument("batch1")
    vh.add_argument("--repeat")
    vh.add_argument("--atol", type=float, default=HARD_ATOL)
    vh.add_argument("--rtol", type=float, default=HARD_RTOL)
    vh.add_argument("--reserve-gib", type=float, default=HARD_RESERVE_GIB)
    a = p.parse_args(argv)

    if a.cmd in (None, "plan"):
        return cmd_plan(a)
    if a.cmd == "run-all":
        return cmd_run_all(a)
    if a.cmd == "verify-probe":
        v = verdict_probe(probe_records(a.report), reserve_gib=a.reserve_gib)
    else:
        v = verdict_hardness(hardness_losses(a.batch8), hardness_losses(a.batch1),
                             repeat=(hardness_losses(a.repeat) if a.repeat else None),
                             atol=a.atol, rtol=a.rtol, reserve_gib=a.reserve_gib)
    print(json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if v["status"] == "PASS" else (2 if v["status"] == "BLOCKED" else 1)


if __name__ == "__main__":
    sys.exit(main())
