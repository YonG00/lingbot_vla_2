#!/usr/bin/env python
"""多 checkpoint + 多卡 RoboTwin 闭环评测调度器。

解决的问题
----------
官方 launcher (`start_robotwin_infer_and_eval.sh`) 一次只评测**一个**
checkpoint。训练会产出多个 checkpoint (global_step_5000/10000/...), 逐个
手动跑既费时又占不满卡; 而一次开卡是按小时计费的。

本脚本在**不改官方调度逻辑**的前提下做编排:

    发现本次训练产出的 checkpoint (含完整性校验)
      ↓
    按并发上限组织成「滑窗」
      ↓
    给每个 checkpoint 启动一份官方 launcher
      ↓
    管理 GPU / port / output / log 隔离
      ↓
    谁先完成, 立刻补下一个 (消除 checkpoint 完成时间不一致的空档)
      ↓
    汇总 clean / randomized 成功率

核心机制 (实测确认)
------------------
* 官方多卡评测**不是模型并行**, 是「多副本 + 任务级并行」:
  每个 slot = 一个独立进程, 各自完整加载一遍模型 → 显存是 N 倍。
* 端口 = `start_port + slot`; GPU = `slot % num_gpus` (在 launcher 内计算)。
* 因此本脚本给每个并发位分配**互不重叠的端口段**, 宽度 = `num_gpus * num_per_gpu`。
* 单张 96G 卡实测可同时容纳 2 个不同 checkpoint 的 FP32 server
  (2 × 25.0 GiB = 49 GiB), 故 `--max-parallel-checkpoints 2` 成立。

边界
----
* **不修改** `experiment/robotwin/robotwin_quick_eval.py` (历史遗留, 已失效)。
* **不重写** 官方 launcher 的 task-level queue, 只通过 `--task_list_file` 注入任务。
* `--dry-run` 不启动任何子进程、不初始化 CUDA。

用法
----
    # 只看计划 (无卡可跑)
    python experiment/robotwin/robotwin_multi_ckpt_eval.py \
        --ckpt-root /data/models/lingbot-vla-v2-6b-robotwin \
        --phase 1 --dry-run

    # 真跑
    python experiment/robotwin/robotwin_multi_ckpt_eval.py \
        --ckpt-root /data/models/lingbot-vla-v2-6b-robotwin \
        --phase 1 --max-parallel-checkpoints 2 --num-gpus 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAUNCHER = REPO_ROOT / "experiment" / "robotwin" / "start_robotwin_infer_and_eval.sh"
DEFAULT_EVAL_LIST_DIR = Path("/data/train/phases")

# condition -> RoboTwin task_config 名
CONDITIONS = {"clean": "demo_clean", "randomized": "demo_randomized"}

# checkpoint 目录布局: <output_dir>/checkpoints/global_step_<N>/hf_ckpt
# 支持把 --ckpt-root 指到 output_dir 本身, 或它的上若干层。
CKPT_PATTERNS = (
    "checkpoints/global_step_*/hf_ckpt",
    "*/checkpoints/global_step_*/hf_ckpt",
    "*/*/checkpoints/global_step_*/hf_ckpt",
)

# HF 转换中断会留下形如 `.model-00002-of-00006.safetensors.XfY3mY` 的临时文件。
# 注意: 这只是**残留**, 不代表 checkpoint 不可用 —— 真实的 global_step_50000
# 就带着这样一个 115MB 残留文件, 但 6 个分片齐全、index 完全对得上。
TEMP_RESIDUE_GLOB = ".*.safetensors.*"

# index.json 里声明的权重文件名, 用于校验分片齐全。
INDEX_FILENAME = "model.safetensors.index.json"

# 传输完成标记 (规范第 13 节工作流 B: rsync 到无卡机时用)。
# 存在该标记时, 跳过「静默期」判定 —— 标记本身就是权威信号。
READY_MARKER = "_READY"


# ---------------------------------------------------------------------------
# checkpoint 发现 + 完整性判据
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    """一个待评测的 HF checkpoint。"""

    hf_ckpt: Path            # .../checkpoints/global_step_<N>/hf_ckpt
    output_dir: Path         # .../checkpoints/global_step_<N>
    exp_name: str            # launcher 用来拼 run_dir 的实验名 (含 /checkpoints 的上一级)
    step: int
    complete: bool
    reasons: list[str] = field(default_factory=list)     # 判为不完整的原因
    warnings: list[str] = field(default_factory=list)    # 只提醒, 不阻塞
    total_bytes: int = 0
    newest_mtime: float = 0.0

    @property
    def tag(self) -> str:
        """人类可读且唯一的短标识, 用作输出目录名。"""
        return f"{self.exp_name}_step{self.step}"

    def fingerprint(self) -> str:
        """内容指纹 —— 变了就说明这个 checkpoint 与上次评测时不同。"""
        return json.dumps(
            {
                "hf_ckpt": str(self.hf_ckpt),
                "complete": self.complete,
                "total_bytes": self.total_bytes,
                "newest_mtime": round(self.newest_mtime, 3),
            },
            sort_keys=True,
        )


def check_hf_ckpt(hf_ckpt: Path, min_age_seconds: float = 0.0,
                  now: float | None = None) -> tuple[bool, list[str], list[str], int, float]:
    """校验一个 hf_ckpt 目录是否「完整且可评测」。

    返回 ``(complete, reasons, warnings, total_bytes, newest_mtime)``。

    判据 (按重要性):
      1. ``model.safetensors.index.json`` 存在且可解析;
      2. index 的 ``weight_map`` 声明的**每一个**分片都存在且非空;
      3. 分片总字节 >= index ``metadata.total_size`` (防截断);
      4. 最新 mtime 距今 >= ``min_age_seconds`` (防「正在写入」被误判为可用)。

    只提醒不阻塞:
      * 残留的 ``.*.safetensors.*`` 临时文件 (HF 转换中断的遗留物);
      * 缺 ``config.json``。

    关于第 4 条: safetensors 文件 = 8 字节头长 + 头 JSON + 张量数据, 所以
    文件总字节必然 >= ``total_size``。截断会让它掉到下面, 因此 ``>=`` 是安全下界。
    """
    reasons: list[str] = []
    warnings: list[str] = []

    if not hf_ckpt.is_dir():
        return False, [f"不是目录: {hf_ckpt}"], [], 0, 0.0

    index_path = hf_ckpt / INDEX_FILENAME
    if not index_path.is_file():
        reasons.append(f"缺少 {INDEX_FILENAME}")
        return False, reasons, warnings, 0, _newest_mtime(hf_ckpt)

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:                       # noqa: BLE001
        reasons.append(f"{INDEX_FILENAME} 解析失败: {type(exc).__name__}: {exc}")
        return False, reasons, warnings, 0, _newest_mtime(hf_ckpt)

    weight_map = index.get("weight_map") or {}
    shards = sorted(set(weight_map.values()))
    if not shards:
        reasons.append(f"{INDEX_FILENAME} 的 weight_map 为空")

    total = 0
    for shard in shards:
        shard_path = hf_ckpt / shard
        if not shard_path.is_file():
            reasons.append(f"缺少分片 {shard}")
            continue
        size = shard_path.stat().st_size
        if size == 0:
            reasons.append(f"分片为空 {shard}")
            continue
        total += size

    declared = (index.get("metadata") or {}).get("total_size")
    if isinstance(declared, int) and declared > 0 and total < declared:
        reasons.append(
            f"分片总字节 {total} < index 声明的 total_size {declared} (疑似截断)"
        )

    # ---- 只提醒 ----
    residue = sorted(p.name for p in hf_ckpt.glob(TEMP_RESIDUE_GLOB))
    if residue:
        warnings.append(
            f"残留临时文件 {residue} (HF 转换中断遗留, 不影响使用, 可删)"
        )
    if not (hf_ckpt / "config.json").is_file():
        warnings.append("缺少 config.json")

    # ---- 静默期 ----
    newest = _newest_mtime(hf_ckpt)
    if not (hf_ckpt / READY_MARKER).exists():
        if min_age_seconds > 0:
            ref = time.time() if now is None else now
            age = ref - newest
            if age < min_age_seconds:
                reasons.append(
                    f"疑似仍在写入 (最新 mtime 距今 {age:.1f}s < 静默期 {min_age_seconds:.0f}s)"
                )

    return (not reasons), reasons, warnings, total, newest


def _newest_mtime(directory: Path) -> float:
    """目录下所有条目的最新 mtime (目录本身也算)。"""
    newest = 0.0
    try:
        newest = directory.stat().st_mtime
    except OSError:
        pass
    try:
        for entry in directory.iterdir():
            try:
                newest = max(newest, entry.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        pass
    return newest


def _parse_step(step_dir_name: str) -> int | None:
    """`global_step_50000` -> 50000。"""
    prefix = "global_step_"
    if not step_dir_name.startswith(prefix):
        return None
    tail = step_dir_name[len(prefix):]
    return int(tail) if tail.isdigit() else None


def discover_checkpoints(ckpt_root: Path, *, min_age_seconds: float = 0.0,
                         now: float | None = None,
                         extra_globs: list[str] | None = None) -> list[Checkpoint]:
    """扫描 ``ckpt_root`` 下的所有 hf_ckpt, 返回按 (exp_name, step) 排序的列表。

    不完整的 checkpoint 也会返回 (``complete=False`` + ``reasons``), 由调用方
    决定是跳过还是报错 —— 这样报告里能说清「为什么没评测它」。
    """
    ckpt_root = Path(ckpt_root)
    found: dict[Path, Checkpoint] = {}

    patterns = list(extra_globs) if extra_globs else list(CKPT_PATTERNS)
    for pattern in patterns:
        for hf_ckpt in sorted(ckpt_root.glob(pattern)):
            hf_ckpt = hf_ckpt.resolve()
            if hf_ckpt in found:
                continue
            step_dir = hf_ckpt.parent                      # .../global_step_N
            step = _parse_step(step_dir.name)
            if step is None:
                continue
            exp_name = step_dir.parent.parent.name          # <exp_name>/checkpoints/global_step_N
            complete, reasons, warnings, total, newest = check_hf_ckpt(
                hf_ckpt, min_age_seconds=min_age_seconds, now=now
            )
            found[hf_ckpt] = Checkpoint(
                hf_ckpt=hf_ckpt,
                output_dir=step_dir,
                exp_name=exp_name,
                step=step,
                complete=complete,
                reasons=reasons,
                warnings=warnings,
                total_bytes=total,
                newest_mtime=newest,
            )

    return sorted(found.values(), key=lambda c: (c.exp_name, c.step))


# ---------------------------------------------------------------------------
# 增量评测状态 (哪些 checkpoint 是本次新增/更新)
# ---------------------------------------------------------------------------

def load_state(path: Path) -> dict:
    """读取上次评测留下的指纹表; 不存在或损坏时返回空表。"""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                              # noqa: BLE001
        return {}
    return data.get("checkpoints", {}) if isinstance(data, dict) else {}


def save_state(path: Path, checkpoints: list[Checkpoint]) -> None:
    """写回指纹表, 供下次判断「新增 / 更新 / 历史不变」。"""
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoints": {
            c.tag: {"fingerprint": c.fingerprint(), "hf_ckpt": str(c.hf_ckpt),
                    "step": c.step, "complete": c.complete}
            for c in checkpoints
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def classify(checkpoints: list[Checkpoint], state: dict) -> dict[str, list[Checkpoint]]:
    """把 checkpoint 分成 new / updated / unchanged / incomplete。"""
    out: dict[str, list[Checkpoint]] = {
        "new": [], "updated": [], "unchanged": [], "incomplete": [],
    }
    for ckpt in checkpoints:
        if not ckpt.complete:
            out["incomplete"].append(ckpt)
            continue
        known = state.get(ckpt.tag)
        if not known:
            out["new"].append(ckpt)
        elif known.get("fingerprint") != ckpt.fingerprint():
            out["updated"].append(ckpt)
        else:
            out["unchanged"].append(ckpt)
    return out


# ---------------------------------------------------------------------------
# 调度计划 (纯函数 —— dry-run 与真实执行共用)
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """一次 launcher 调用 (一个 checkpoint × 一个 condition)。"""

    ckpt_tag: str
    step: int
    hf_ckpt: str
    condition: str                 # clean | randomized
    task_config: str               # demo_clean | demo_randomized
    slot_index: int
    gpus: list[int]
    start_port: int
    end_port: int                  # 闭区间上界
    output_base: str
    log_file: str
    cmd: list[str] = field(default_factory=list)

    # 运行后填充
    returncode: int | None = None
    duration_s: float | None = None
    run_dir: str | None = None
    stats_file: str | None = None
    overall_rate: float | None = None
    success: int | None = None
    episodes: int | None = None


@dataclass
class CkptPlan:
    """一个 checkpoint 的完整作业序列 (condition 之间串行)。"""

    index: int
    tag: str
    step: int
    hf_ckpt: str
    slot_index: int
    jobs: list[Job] = field(default_factory=list)


def make_batches(n_ckpts: int, max_parallel: int) -> list[int]:
    """把 n 个 checkpoint 按并发上限切成批次大小, 用于校验滑窗形状。

    例: n=5, max_parallel=2 -> [2, 2, 1]
    """
    if max_parallel < 1:
        raise ValueError("max_parallel 必须 >= 1")
    if n_ckpts <= 0:
        return []
    return [min(max_parallel, n_ckpts - i)
            for i in range(0, n_ckpts, max_parallel)]


def plan_schedule(
    checkpoints: list[Checkpoint],
    *,
    max_parallel: int,
    conditions: list[str],
    num_gpus: int,
    num_per_gpu: int,
    start_port_base: int,
    output_base: Path,
    task_list_file: Path,
    launcher: Path = DEFAULT_LAUNCHER,
    eval_workdir: str | None = None,
    conda_sh: str | None = None,
    inference_env: str | None = None,
    sim_env: str | None = None,
    use_fp32: bool = True,
    use_compile: bool = False,
    enable_video: bool = False,
    episodes: int | None = None,
    extra_args: list[str] | None = None,
) -> list[CkptPlan]:
    """把 checkpoint 列表展开成确定性的调度计划。

    纯函数: 不碰文件系统 (除了读取任务清单的调用方), 不启动进程。

    slot 分配是**静态**的 —— 第 i 个 checkpoint 用 ``i % max_parallel`` 号并发位。
    因为滑窗保证 ``i`` 与 ``i + max_parallel`` 不会同时在跑, 所以端口段可以安全复用:

        端口段宽度 = num_gpus * num_per_gpu   (launcher 里 num_slots 的上界)
        第 k 号并发位 = [start_port_base + k*band, ... + band - 1]
    """
    if max_parallel < 1:
        raise ValueError("max_parallel 必须 >= 1")
    for cond in conditions:
        if cond not in CONDITIONS:
            raise ValueError(f"未知 condition: {cond} (可选: {sorted(CONDITIONS)})")

    band = num_gpus * num_per_gpu
    output_base = Path(output_base)
    task_list_file = Path(task_list_file)
    plans: list[CkptPlan] = []

    for index, ckpt in enumerate(checkpoints):
        slot_index = index % max_parallel
        plan = CkptPlan(
            index=index,
            tag=ckpt.tag,
            step=ckpt.step,
            hf_ckpt=str(ckpt.hf_ckpt),
            slot_index=slot_index,
        )

        for cond in conditions:
            job_start = start_port_base + slot_index * band
            job_output = output_base / ckpt.tag / cond
            job_log = output_base / "_logs" / f"{ckpt.tag}.{cond}.log"

            cmd = [
                "bash", str(launcher),
                "--model_path", str(ckpt.hf_ckpt),
                "--output_base", str(job_output),
                "--start_port", str(job_start),
                "--task_list_file", str(task_list_file),
                "--task_config", CONDITIONS[cond],
                "--num_gpus", str(num_gpus),
                "--num_per_gpu", str(num_per_gpu),
                "--use_fp32", "true" if use_fp32 else "false",
                "--use_bf16", "false" if use_fp32 else "true",
                "--use_compile", "true" if use_compile else "false",
            ]
            # video 必须**显式双向**传递: launcher 里 enable_video 的默认值是 False,
            # 且原本只有 --no_video、没有能把它打开的开关 —— 只传 --no_video 时,
            # 调度器的 --video 其实什么也没做。现在两个分支都显式传, 行为不依赖
            # 上游默认值 (配套给 launcher 补了对称的 --enable_video)。
            cmd.append("--enable_video" if enable_video else "--no_video")
            if episodes:
                # 透传给 eval client 的 test_num。不传 = 保留 client 自己的默认值
                # (官方 100), 这样不显式指定时行为与官方完全一致。
                cmd += ["--test_num", str(episodes)]
            if eval_workdir:
                cmd += ["--eval_workdir", str(eval_workdir)]
            if conda_sh:
                cmd += ["--conda_sh", str(conda_sh)]
            if inference_env:
                cmd += ["--inference_env", str(inference_env)]
            if sim_env:
                cmd += ["--sim_env", str(sim_env)]
            if extra_args:
                cmd += list(extra_args)

            plan.jobs.append(Job(
                ckpt_tag=ckpt.tag,
                step=ckpt.step,
                hf_ckpt=str(ckpt.hf_ckpt),
                condition=cond,
                task_config=CONDITIONS[cond],
                slot_index=slot_index,
                gpus=list(range(num_gpus)),
                start_port=job_start,
                end_port=job_start + band - 1,
                output_base=str(job_output),
                log_file=str(job_log),
                cmd=cmd,
            ))

        plans.append(plan)

    return plans


def validate_plan(plans: list[CkptPlan], *, max_parallel: int) -> list[str]:
    """检查计划的自洽性, 返回问题列表 (空 = 通过)。

    覆盖: 端口段不重叠、输出目录不重叠、日志不重叠、model_path 不串。
    """
    problems: list[str] = []

    # 同一时刻可能并发的作业 = 每个 slot 上最多一个。
    by_slot: dict[int, list[Job]] = {}
    for plan in plans:
        for job in plan.jobs:
            by_slot.setdefault(job.slot_index, []).append(job)

    for slot, jobs in by_slot.items():
        # 同一个 slot 上, 同一 condition 的端口段必须一致 (复用), 不同 condition 也一样。
        bands = {(j.start_port, j.end_port) for j in jobs}
        if len(bands) != 1:
            problems.append(f"slot {slot} 的端口段不唯一: {sorted(bands)}")

    # 不同 slot 之间的端口段不能重叠。
    slot_bands = {s: next(iter({(j.start_port, j.end_port) for j in jobs}))
                  for s, jobs in by_slot.items()}
    slots = sorted(slot_bands)
    for a, b in zip(slots, slots[1:]):
        if slot_bands[a][1] >= slot_bands[b][0]:
            problems.append(
                f"slot {a} 端口段 {slot_bands[a]} 与 slot {b} 端口段 {slot_bands[b]} 重叠"
            )

    # 输出 / 日志路径唯一。
    outputs = [j.output_base for p in plans for j in p.jobs]
    if len(outputs) != len(set(outputs)):
        problems.append("存在重复的 output_base")
    logs = [j.log_file for p in plans for j in p.jobs]
    if len(logs) != len(set(logs)):
        problems.append("存在重复的 log_file")

    # 同一个 checkpoint 的所有作业必须用同一个 model_path。
    for plan in plans:
        paths = {j.hf_ckpt for j in plan.jobs}
        if len(paths) != 1:
            problems.append(f"{plan.tag} 的作业 model_path 不一致: {sorted(paths)}")

    # 并发的 slot 数不能超过上限。
    if len(by_slot) > max_parallel:
        problems.append(f"并发 slot 数 {len(by_slot)} 超过上限 {max_parallel}")

    return problems


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------

def _stats_from_run_dir(run_dir: Path) -> dict:
    """解析 launcher 生成的 stats.txt。"""
    stats_file = run_dir / "stats.txt"
    result: dict = {"stats_file": str(stats_file)}
    if not stats_file.is_file():
        return result
    text = stats_file.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("Summary:"):
            # Summary: total 261s, success 3/6, overall rate 50.0%
            try:
                tail = line.split("success", 1)[1]
                frac = tail.split(",")[0].strip()
                suc, tot = (int(x) for x in frac.split("/"))
                result["success"] = suc
                result["episodes"] = tot
                rate = line.split("overall rate", 1)[1].strip().rstrip("%")
                result["overall_rate"] = float(rate)
            except Exception:                       # noqa: BLE001
                pass
            break
    return result


def run_job(job: Job, *, dry_run: bool = False) -> Job:
    """执行一个作业 (一个 launcher 调用)。"""
    if dry_run:
        return job

    log_file = Path(job.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    Path(job.output_base).mkdir(parents=True, exist_ok=True)

    started = time.time()
    with log_file.open("w", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(job.cmd)}\n")
        fh.flush()
        proc = subprocess.run(job.cmd, stdout=fh, stderr=subprocess.STDOUT)
    job.returncode = proc.returncode
    job.duration_s = time.time() - started

    # launcher 的 run_dir 带时间戳, 事后按目录里最新的 stats.txt 认领。
    candidates = sorted(
        (p.parent for p in Path(job.output_base).glob("*/stats.txt")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        job.run_dir = str(candidates[0])
        job.__dict__.update(_stats_from_run_dir(candidates[0]))
    return job


def run_plan(plan: CkptPlan, *, dry_run: bool = False, on_done=None) -> CkptPlan:
    """串行执行一个 checkpoint 的所有 condition。"""
    for job in plan.jobs:
        run_job(job, dry_run=dry_run)
    if on_done:
        on_done(plan)
    return plan


def run_scheduler(plans: list[CkptPlan], *, max_parallel: int,
                  dry_run: bool = False, poll_seconds: float = 2.0) -> list[CkptPlan]:
    """滑窗调度: 同时最多跑 ``max_parallel`` 个 checkpoint, 谁先完成立刻补下一个。

    与「静态分批」的区别: 静态分批要等整批都结束才开下一批, checkpoint 完成时间
    不一致时会留空档; 滑窗则是**有一个空位就立刻补位**。
    """
    if dry_run:
        for plan in plans:
            for job in plan.jobs:
                run_job(job, dry_run=True)
        return plans

    pending = deque(plans)
    running: dict[int, threading.Thread] = {}
    free_slots: deque[int] = deque(range(max_parallel))

    def _log(msg: str) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    while pending or running:
        # 有空位就立刻补位 —— 这正是「滑窗」相对「静态分批」的关键:
        # 静态分批要等整批都跑完, checkpoint 完成时间不一致时会留空档。
        while pending and free_slots:
            slot = free_slots.popleft()
            plan = pending.popleft()
            _log(f"启动 {plan.tag} (step {plan.step}) -> slot {slot}, "
                 f"端口 {plan.jobs[0].start_port}~{plan.jobs[-1].end_port}")
            thread = threading.Thread(
                target=run_plan,
                kwargs={"plan": plan, "dry_run": False,
                        "on_done": lambda p: _log(f"完成 {p.tag}")},
                daemon=True,
            )
            running[slot] = thread
            thread.start()

        if not running:
            break

        time.sleep(poll_seconds)
        for slot in [s for s, t in running.items() if not t.is_alive()]:
            running.pop(slot).join()
            free_slots.append(slot)

    return plans


# ---------------------------------------------------------------------------
# 预检 (消除共享状态首次并发写)
# ---------------------------------------------------------------------------

def preflight(eval_workdir: Path, inference_workdir: Path) -> list[str]:
    """在并发启动**之前**, 单进程把 RoboTwin 侧的共享文件准备好。

    官方 launcher 会在「文件不存在」时才从推理仓库拷贝 eval client 与 deploy
    辅助模块。两个 launcher 同时首跑时, 这两次拷贝会互相竞争 (读到半截文件)。
    这里先串行做一次, 让并发阶段无共享写入。

    只做幂等的补齐, 不覆盖已存在的文件 (除非内容不同, 此时用原子替换)。
    返回人类可读的动作列表。
    """
    actions: list[str] = []
    eval_workdir = Path(eval_workdir)
    inference_workdir = Path(inference_workdir)

    pairs = [
        (inference_workdir / "experiment" / "robotwin" / "eval_policy_client_lingbotvla.py",
         eval_workdir / "script" / "eval_policy_client_lingbotvla.py"),
        (inference_workdir / "deploy" / "__init__.py",
         eval_workdir / "script" / "deploy" / "__init__.py"),
        (inference_workdir / "deploy" / "websocket_client_policy.py",
         eval_workdir / "script" / "deploy" / "websocket_client_policy.py"),
        (inference_workdir / "deploy" / "msgpack_numpy.py",
         eval_workdir / "script" / "deploy" / "msgpack_numpy.py"),
    ]

    for src, dst in pairs:
        if not src.is_file():
            actions.append(f"[跳过] 源文件不存在: {src}")
            continue
        if dst.is_file() and dst.read_bytes() == src.read_bytes():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + f".preflight.{os.getpid()}")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)                 # 同目录 rename, 原子 —— 读者不会看到半截文件
        actions.append(f"[同步] {dst}")

    return actions


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def render_plan(plans: list[CkptPlan], *, task_list_file: Path, tasks: list[str],
                max_parallel: int, conditions: list[str],
                episodes: int | None = None) -> str:
    """把计划渲染成人类可读文本。"""
    lines = [
        "=" * 78,
        "  多 checkpoint 评测计划 (dry-run)",
        "=" * 78,
        f"  checkpoint 数 : {len(plans)}",
        f"  并发上限      : {max_parallel}",
        f"  批次形状      : {make_batches(len(plans), max_parallel)}",
        f"  condition     : {' -> '.join(conditions)} (每个 ckpt 内部串行)",
        f"  回合数/任务   : {episodes if episodes else '(不覆盖, 用官方默认 100)'}"
        f"  × {len(conditions)} 个 condition",
        f"  任务清单      : {task_list_file}  ({len(tasks)} 个任务)",
        f"  任务          : {' '.join(tasks)}",
        f"  总 rollout 数 : {len(plans)} ckpt × {len(tasks)} 任务 × "
        f"{(episodes or 100)} 回合 × {len(conditions)} condition",
        "=" * 78,
    ]
    for plan in plans:
        lines.append("")
        lines.append(f"  [{plan.index}] {plan.tag}   slot={plan.slot_index}   "
                     f"step={plan.step}")
        lines.append(f"      model_path : {plan.hf_ckpt}")
        for job in plan.jobs:
            lines.append(
                f"      - {job.condition:<11} port {job.start_port}~{job.end_port}  "
                f"gpu {job.gpus}  cfg {job.task_config}"
            )
            lines.append(f"        output : {job.output_base}")
            lines.append(f"        log    : {job.log_file}")
    lines.append("")
    return "\n".join(lines)


def render_summary(plans: list[CkptPlan], *, skipped: dict[str, list[Checkpoint]],
                   dry_run: bool) -> str:
    """渲染最终结果汇总。"""
    lines = [
        "=" * 78,
        "  多 checkpoint 评测结果" + ("  (dry-run, 未实际执行)" if dry_run else ""),
        "=" * 78,
    ]
    for plan in plans:
        lines.append("")
        lines.append(f"  {plan.tag}  (step {plan.step})")
        for job in plan.jobs:
            if dry_run:
                lines.append(f"    {job.condition:<11} [计划] port {job.start_port}~{job.end_port}")
                continue
            rc = job.returncode
            rate = f"{job.overall_rate:.1f}%" if job.overall_rate is not None else "-"
            frac = (f"{job.success}/{job.episodes}"
                    if job.success is not None else "-")
            dur = f"{job.duration_s:.0f}s" if job.duration_s is not None else "-"
            lines.append(
                f"    {job.condition:<11} rc={rc}  成功率 {rate} ({frac})  耗时 {dur}"
            )
            if job.run_dir:
                lines.append(f"        run_dir : {job.run_dir}")

    if not dry_run:
        lines += ["", "-" * 78, "  Clean vs Randomized 对比", "-" * 78]
        lines.append(f"    {'checkpoint':<40} {'clean':>9} {'randomized':>12}")
        for plan in plans:
            rates = {j.condition: j.overall_rate for j in plan.jobs}
            def _fmt(v): return f"{v:.1f}%" if v is not None else "-"
            lines.append(f"    {plan.tag:<40} "
                         f"{_fmt(rates.get('clean')):>9} "
                         f"{_fmt(rates.get('randomized')):>12}")

    for bucket in ("incomplete", "unchanged"):
        items = skipped.get(bucket) or []
        if not items:
            continue
        lines += ["", "-" * 78, f"  未评测 ({bucket}): {len(items)} 个", "-" * 78]
        for ckpt in items:
            detail = "; ".join(ckpt.reasons) if ckpt.reasons else "指纹未变"
            lines.append(f"    {ckpt.tag:<40} {detail}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="多 checkpoint + 多卡 RoboTwin 闭环评测调度器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n----", 1)[-1],
    )
    ap.add_argument("--ckpt-root", required=True,
                    help="训练输出目录 (或它的上层目录), 其下按 "
                         "<output_dir>/checkpoints/global_step_*/hf_ckpt 发现 checkpoint")
    ap.add_argument("--ckpt-glob", action="append", default=None,
                    help="自定义发现模式 (可重复); 默认覆盖 1~2 层嵌套")

    src = ap.add_argument_group("评测内容")
    src.add_argument("--phase", type=int, choices=[1, 2, 3, 4], default=None,
                     help="阶段号 -> 读取 <eval-list-dir>/phase<N>_eval.txt")
    src.add_argument("--task-list-file", default=None,
                     help="直接指定任务清单文件 (与 --phase 二选一)")
    src.add_argument("--eval-list-dir", default=str(DEFAULT_EVAL_LIST_DIR),
                     help=f"阶段清单目录 (默认 {DEFAULT_EVAL_LIST_DIR})")
    src.add_argument("--conditions", default="clean,randomized",
                     help="评测条件, 逗号分隔 (默认 clean,randomized; 同一 ckpt 内串行)")
    src.add_argument("--episodes", type=int, default=3,
                     help="每个任务每个 condition 的评测回合数 (默认 3, 与 "
                          "curriculum yaml 的 evaluation.protocols.mini_eval 一致)。"
                          "透传为 eval client 的 test_num; 传 0 表示不覆盖 "
                          "(用 client 自带的官方默认 100)")

    sched = ap.add_argument_group("调度")
    sched.add_argument("--max-parallel-checkpoints", type=int, default=2,
                       help="同时评测的 checkpoint 上限 (默认 2; 显存不够就设 1)")
    sched.add_argument("--num-gpus", type=int, default=4)
    sched.add_argument("--num-per-gpu", type=int, default=1)
    sched.add_argument("--start-port-base", type=int, default=9330)
    sched.add_argument("--output-base", default="/data/eval_results/multi_ckpt")

    sel = ap.add_argument_group("选择哪些 checkpoint")
    sel.add_argument("--all", action="store_true",
                     help="评测全部完整 checkpoint (默认只评测新增/更新的)")
    sel.add_argument("--only-steps", default=None,
                     help="只评测这些 step, 逗号分隔 (例如 10000,20000)")
    sel.add_argument("--min-age-seconds", type=float, default=120.0,
                     help="最新 mtime 距今不足该秒数视为仍在写入, 跳过 (默认 120)")
    sel.add_argument("--state-file", default=None,
                     help="增量状态文件 (默认 <output-base>/eval_state.json)")

    env = ap.add_argument_group("环境")
    env.add_argument("--launcher", default=str(DEFAULT_LAUNCHER))
    env.add_argument("--eval-workdir", default="/data/code/RoboTwin-lingbot")
    env.add_argument("--inference-workdir", default=None,
                     help="推理侧工作目录 (默认仓库根)")
    env.add_argument("--conda-sh", default="/data/miniconda3/etc/profile.d/conda.sh")
    env.add_argument("--inference-env", default=None)
    env.add_argument("--sim-env", default=None)
    env.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                     help="推理精度 (默认 fp32 = 发布复现设置)")
    env.add_argument("--use-compile", action="store_true",
                     help="开启模型 compile (默认关, 省启动时间)")
    env.add_argument("--video", action="store_true",
                     help="录制评测视频 (默认关, 省 IO 与时间)")

    ap.add_argument("--dry-run", action="store_true",
                    help="只打印计划, 不启动任何子进程、不初始化 CUDA")
    ap.add_argument("--no-preflight", action="store_true",
                    help="跳过并发前的共享文件预检")
    ap.add_argument("--poll-seconds", type=float, default=2.0,
                    help="滑窗轮询间隔 (默认 2s)")
    return ap


def resolve_task_list(args) -> tuple[Path, list[str]]:
    """确定任务清单路径 + 任务名列表。"""
    if args.task_list_file:
        path = Path(args.task_list_file)
    elif args.phase:
        path = Path(args.eval_list_dir) / f"phase{args.phase}_eval.txt"
    else:
        raise SystemExit("必须给出 --phase 或 --task-list-file 之一")

    if not path.is_file():
        raise SystemExit(
            f"任务清单不存在: {path}\n"
            f"  阶段清单由 `python tools/prepare_phase.py --phase all` 生成"
        )

    tasks = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tasks.append(line)
    if not tasks:
        raise SystemExit(f"任务清单为空: {path}")
    return path, tasks


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    task_list_file, tasks = resolve_task_list(args)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    output_base = Path(args.output_base)
    state_file = Path(args.state_file) if args.state_file else output_base / "eval_state.json"
    inference_workdir = Path(args.inference_workdir) if args.inference_workdir else REPO_ROOT

    print("=" * 78)
    print("  多 checkpoint 多卡 RoboTwin 评测调度器")
    print("=" * 78)
    print(f"  checkpoint 根 : {args.ckpt_root}")
    print(f"  任务清单      : {task_list_file}  ({len(tasks)} 任务)")
    print(f"  输出根        : {output_base}")
    print(f"  并发上限      : {args.max_parallel_checkpoints}")
    print(f"  精度          : {args.precision}"
          f"{'  (dry-run)' if args.dry_run else ''}")
    print()

    # ---- 1. 发现 ----
    checkpoints = discover_checkpoints(
        Path(args.ckpt_root),
        min_age_seconds=0.0 if args.dry_run else args.min_age_seconds,
        extra_globs=args.ckpt_glob,
    )
    if not checkpoints:
        print(f"  未发现任何 checkpoint: {args.ckpt_root}")
        print("  期望布局: <output_dir>/checkpoints/global_step_<N>/hf_ckpt")
        return 1

    print(f"  发现 {len(checkpoints)} 个 checkpoint:")
    for ckpt in checkpoints:
        mark = "完整" if ckpt.complete else "不完整"
        print(f"    [{'x' if ckpt.complete else ' '}] {ckpt.tag:<40} {mark}"
              f"  {ckpt.total_bytes/2**30:.1f} GiB")
        for reason in ckpt.reasons:
            print(f"          - 跳过原因: {reason}")
        for warning in ckpt.warnings:
            print(f"          - 提醒: {warning}")
    print()

    # ---- 2. 分类 ----
    state = load_state(state_file)
    buckets = classify(checkpoints, state)

    selected = list(buckets["new"]) + list(buckets["updated"])
    if args.all:
        selected = [c for c in checkpoints if c.complete]
    if args.only_steps:
        wanted = {int(s) for s in args.only_steps.split(",") if s.strip()}
        selected = [c for c in selected if c.step in wanted]

    # 去重 + 稳定排序
    seen: set[str] = set()
    ordered: list[Checkpoint] = []
    for ckpt in sorted(selected, key=lambda c: (c.exp_name, c.step)):
        if ckpt.tag not in seen:
            seen.add(ckpt.tag)
            ordered.append(ckpt)

    print(f"  增量分类: 新增 {len(buckets['new'])} / 更新 {len(buckets['updated'])} / "
          f"未变 {len(buckets['unchanged'])} / 不完整 {len(buckets['incomplete'])}")
    if args.all:
        print("  (--all: 评测全部完整 checkpoint)")
    print(f"  本次将评测: {len(ordered)} 个 checkpoint")
    for ckpt in ordered:
        print(f"    - {ckpt.tag}")
    print()

    if not ordered:
        print("  没有需要评测的 checkpoint。")
        print("  提示: 用 --all 强制全量, 或用 --only-steps 指定。")
        return 0

    # ---- 3. 计划 ----
    plans = plan_schedule(
        ordered,
        max_parallel=args.max_parallel_checkpoints,
        conditions=conditions,
        num_gpus=args.num_gpus,
        num_per_gpu=args.num_per_gpu,
        start_port_base=args.start_port_base,
        output_base=output_base,
        task_list_file=task_list_file,
        launcher=Path(args.launcher),
        eval_workdir=args.eval_workdir,
        conda_sh=args.conda_sh,
        inference_env=args.inference_env,
        sim_env=args.sim_env,
        use_fp32=(args.precision == "fp32"),
        use_compile=args.use_compile,
        enable_video=args.video,
        episodes=args.episodes or None,
    )

    problems = validate_plan(plans, max_parallel=args.max_parallel_checkpoints)
    if problems:
        print("  计划自检失败:")
        for problem in problems:
            print(f"    - {problem}")
        return 2
    print("  计划自检通过 (端口段/输出/日志互不重叠, model_path 不串)")
    print()

    print(render_plan(plans, task_list_file=task_list_file, tasks=tasks,
                      max_parallel=args.max_parallel_checkpoints,
                      conditions=conditions,
                      episodes=args.episodes or None))

    # ---- 4. 预检 ----
    if not args.dry_run and not args.no_preflight:
        actions = preflight(Path(args.eval_workdir), inference_workdir)
        print("  预检 (并发前的共享文件同步):")
        if actions:
            for action in actions:
                print(f"    {action}")
        else:
            print("    无需改动")
        print()

    # ---- 5. 执行 ----
    started = time.time()
    run_scheduler(plans, max_parallel=args.max_parallel_checkpoints,
                  dry_run=args.dry_run, poll_seconds=args.poll_seconds)
    elapsed = time.time() - started

    # ---- 6. 报告 ----
    report = render_summary(plans, skipped=buckets, dry_run=args.dry_run)
    print(report)

    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "summary.txt").write_text(report, encoding="utf-8")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
        "elapsed_seconds": round(elapsed, 1),
        "task_list_file": str(task_list_file),
        "tasks": tasks,
        "max_parallel_checkpoints": args.max_parallel_checkpoints,
        "conditions": conditions,
        "batches": make_batches(len(plans), args.max_parallel_checkpoints),
        "plans": [asdict(p) for p in plans],
        "skipped": {
            key: [{"tag": c.tag, "reasons": c.reasons} for c in items]
            for key, items in buckets.items()
        },
    }
    (output_base / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"  summary.txt : {output_base / 'summary.txt'}")
    print(f"  summary.json: {output_base / 'summary.json'}")

    if not args.dry_run:
        save_state(state_file, checkpoints)
        print(f"  state       : {state_file}")

    failed = [j for p in plans for j in p.jobs
              if not args.dry_run and j.returncode not in (0, None)]
    if failed:
        print(f"  有 {len(failed)} 个作业返回非 0, 详见各自的 log。")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
