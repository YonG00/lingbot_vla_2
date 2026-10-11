#!/usr/bin/env python3
"""AL 训练仪表盘 v2 —— 一条命令看：进度 / 通关 / 谁通关 / 门槛(GMean) / 还差多少。

用法（训练机上直接跑）：
    python3 /workspace/runtime/al_panel.py                 # 自动选最新 run
    python3 /workspace/runtime/al_panel.py al_v41          # 指定 run
    python3 /workspace/runtime/al_panel.py al_v41 --all    # 列出全部 50 个任务
    watch -n 30 python3 /workspace/runtime/al_panel.py     # 每 30 秒刷新

判定口径（实测确认）：
    任务 PASS  ⇔  active_val_gmean_mse ≤ pass_threshold_gmean_mse
    现成比值  =  val_to_pass_threshold （≤1 ⇒ PASS，越小越好）

数据来源（全部只读）：
  · <root>/logs/train_<run>.log                     训练进度（Step / s/it）
  · <root>/train_<run>/auto_learning_events.jsonl   AL 事件（PASS 数、逐任务指标）
  · <root>/train_<run>/lingbotvla_cli.yaml          本次 run 的 AL 配置路径
  · 该 AL 配置里的 pass_thresholds_file             门槛表（50 个任务的线）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: al_runs 根目录；换机器用 `AL_ROOT=<al_runs 路径>` 覆盖。
ROOT = Path(os.environ.get('AL_ROOT', '/models/robotwin-persistent/al_runs'))
W = 82


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def sh(cmd: List[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return ''


def rank_count() -> int:
    """只数**真正的训练 rank**（判据：该进程是 `torch.distributed.run` 的直接子进程）。

    ⚠️ 不能用"环境里有没有 `WORLD_SIZE`"——dataloader / 编译 worker 都**继承**父进程环境
    （实测把 6 个 rank 数成 54 个）。也不能只靠 cmdline 匹配（bash / train.sh / torchrun
    的命令行里都含脚本名）。
    """
    all_pids = _pids()
    torchrun_pids = set()
    for p in all_pids:
        if 'torch.distributed.run' in _cmdline(p):
            torchrun_pids.add(p)
    if not torchrun_pids:
        return 0
    n = 0
    for p in all_pids:
        if _ppid(p) in torchrun_pids:
            n += 1
    return n


def _pids() -> List[str]:
    if not Path('/proc').is_dir():
        return []
    return [p for p in os.listdir('/proc') if p.isdigit()]


def _ppid(pid: str) -> str:
    try:
        with open(f'/proc/{pid}/stat', 'rb') as fh:
            parts = fh.read().decode('utf-8', 'replace').rsplit(')', 1)[1].split()
        return parts[1]          # state 之后第一个字段是 ppid
    except (OSError, IndexError):
        return ''


def _cmdline(pid: str) -> str:
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as fh:
            return fh.read().decode('utf-8', 'replace')
    except OSError:
        return ''


def pick_run(arg: Optional[str]) -> str:
    if arg:
        return arg
    try:
        cands = sorted((p.name for p in ROOT.glob('logs/train_al_*.log')),
                       key=lambda n: (ROOT / 'logs' / n).stat().st_mtime, reverse=True)
        if cands:
            return cands[0][len('train_'):-len('.log')]
    except OSError:
        pass
    return ''            # 找不到任何 run ⇒ 空（下面会提示怎么指定）


def run_cfg(run: str) -> Dict[str, object]:
    """从 run 目录的 CLI 配置里取：AL 配置路径 / 目标 PASS / 评测间隔。"""
    out: Dict[str, object] = {'al_cfg': None, 'target': None, 'interval': None,
                             'thresholds': None}
    p = ROOT / f'train_{run}' / 'lingbotvla_cli.yaml'
    if not p.is_file():
        return out
    txt = p.read_text(encoding='utf-8', errors='replace')
    m = re.search(r'^\s*auto_learning:\s*(\S+)\s*$', txt, re.M)
    if m:
        out['al_cfg'] = m.group(1)
    # AL 配置里的目标与阈值表
    al_path = Path(str(out['al_cfg'])).expanduser() if out['al_cfg'] else None
    if al_path and al_path.is_file():
        a = al_path.read_text(encoding='utf-8', errors='replace')
        m = re.search(r'^target_total_passed_tasks:\s*(\d+)', a, re.M)
        if m:
            out['target'] = int(m.group(1))
        m = re.search(r'^pass_thresholds_file:\s*(\S+)', a, re.M)
        if m:
            out['thresholds'] = m.group(1)
        m = re.search(r'^eval_interval_steps:\s*(\d+)', a, re.M)
        if m:
            out['interval'] = int(m.group(1))
    return out


def load_thresholds(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    try:
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        t = d.get('tasks') or {}
        return {k: float(v) for k, v in t.items() if isinstance(v, (int, float))}
    except Exception:
        return {}


def read_events(events: Path) -> Tuple[Dict[str, object], Dict[str, Dict[str, float]]]:
    """返回 (每个 metric 名的最新值, 每个任务的最新指标)。"""
    last: Dict[str, object] = {}
    per_task: Dict[str, Dict[str, float]] = {}
    if not events.is_file():
        return last, per_task
    try:
        with events.open(encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if '"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                name = d.get('name')
                if not name:
                    continue
                val = d.get('value')
                last[name] = val
                m = re.match(r'task/(.+)/(.+)$', name)
                if m and isinstance(val, (int, float)):
                    per_task.setdefault(m.group(1), {})[m.group(2)] = float(val)
    except OSError:
        pass
    return last, per_task


def step_progress(log: Path, interval: Optional[int]) -> Tuple[Optional[int], Optional[float], int]:
    """返回 (最新 step, 稳态 s/it 中位数, 采样到的步数)。"""
    if not log.is_file():
        return None, None, 0
    try:
        tail = log.read_text(encoding='utf-8', errors='replace')[-400_000:]
    except OSError:
        return None, None, 0
    hits = re.findall(r'Step:\s*(\d+)/\d+\s*\[[^]]*?([\d.]+)s/it', tail)
    if not hits:
        m = re.findall(r'Step:\s*(\d+)/', tail)
        return (int(m[-1]) if m else None), None, 0
    steps = [int(a) for a, _ in hits]
    rates = [float(b) for _, b in hits]
    steady = rates[-10:] if len(rates) >= 3 else rates
    return steps[-1], statistics.median(steady), len(hits)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description='AL 训练仪表盘')
    ap.add_argument('run', nargs='?', default=None, help='run 名（默认自动选最新）')
    ap.add_argument('--all', action='store_true', help='列出全部任务（默认只列最接近通关的 15 个）')
    ap.add_argument('--top', type=int, default=15, help='默认显示多少个任务（按倍数升序）')
    args = ap.parse_args()

    run = pick_run(args.run)
    if not run or not (ROOT / f'train_{run}').is_dir():
        print(f'未找到 run（ROOT={ROOT}）。请指定：al_panel.py <run名>；'
              '或用 AL_ROOT=<al_runs 目录> 覆盖根目录。')
        return 2
    cfg = run_cfg(run)
    thresholds = load_thresholds(cfg['thresholds'])          # type: ignore[arg-type]
    target = int(cfg['target'] or 15)
    last, per_task = read_events(ROOT / f'train_{run}' / 'auto_learning_events.jsonl')
    nrank = rank_count()
    step, sit, nsamples = step_progress(ROOT / 'logs' / f'train_{run}.log', cfg['interval'])  # type: ignore[arg-type]

    passed = int(last.get('curriculum/passed_tasks') or 0)     # type: ignore[arg-type]
    newly = last.get('curriculum/newly_passed_count')
    boot = last.get('curriculum/bootstrap_pass_count')
    print('=' * W)
    print(f' AL 仪表盘  run={run}   目标 PASS={target}   门槛={Path(str(cfg["thresholds"] or "?")).name}')
    print('=' * W)
    if nrank:
        line = f' 状态: 训练中（{nrank} 个 rank / {nrank} 张卡）'
        if step is not None:
            line += f' | Step {step}'
        if sit:
            line += f' | 稳态 {sit:.1f}s/it（近 {min(nsamples, 10)} 次采样中位数）'
        print(line)
    else:
        print(' 状态: 未在训练（无 rank 进程）')
    pct = (100.0 * passed / target) if target else 0.0
    extra = ''
    if isinstance(newly, (int, float)):
        extra += f' | 本 run 新增 {int(newly)}'
    if isinstance(boot, (int, float)) and boot:
        extra += f' | Bootstrap {int(boot)}'
    print(f' 通关: {passed}/{target}（{pct:.0f}%）  还差 {max(0, target - passed)} 个{extra}')
    # 下一次判定还有多少步（unit = eval_interval_steps；判定只在 unit 末尾发生）
    if cfg['interval'] and step is not None:
        rem = int(cfg['interval']) - (step % int(cfg['interval']))
        print(f' 下一次判定: 再过 {rem} 步（Step {step + rem}；unit 末才 REVIEW）')
    print(f' 门槛表: {cfg["thresholds"] or "（未找到）"}（{len(thresholds)} 个任务有门槛）')
    if cfg['interval']:
        print(f' 评测节奏: 每 {cfg["interval"]} 步一个 unit（unit 末尾才判定 PASS）')

    # 逐任务：以**阈值表**为准（50 个全有），指标取已评测到的最新值
    rows = []
    for task, th in thresholds.items():
        m = per_task.get(task) or {}
        g = m.get('active_val_gmean_mse')
        src = 'val'
        if g is None:
            g = m.get('scout_gmean_mse')
            src = 'scout'
        rows.append((task, g, float(th), src))
    if not rows:      # 阈值表拿不到时退回事件里的指标
        for task, m in per_task.items():
            g = m.get('active_val_gmean_mse') or m.get('scout_gmean_mse')
            th = m.get('pass_threshold_gmean_mse')
            if g is not None and th:
                rows.append((task, g, th, 'val'))
    rated = [r for r in rows if r[1] is not None]
    unrated = [r for r in rows if r[1] is None]
    rated.sort(key=lambda r: r[1] / r[2] if r[2] else 9e9)

    if rated:
        print('-' * W)
        print(f' {"任务":<28}{"GMean":>10}{"门槛":>10}{"倍数":>8}  {"数据":<6} 距过线')
        print('-' * W)
        shown = rated if args.all else rated[:max(1, args.top)]
        for task, g, th, src in shown:
            ratio = g / th if th else float('inf')
            # 🔴 措辞修正（用户 2026-07 反馈）：行内**不再**写 "PASS"（会被读成"已通关"）。
            #    这里只表达"该指标是否已过线"，且必须标明数据来源（实测 / 估值）。
            verdict = '已过线' if g <= th else ('接近' if ratio <= 1.3 else '差距大')
            data = '实测' if src == 'val' else 'scout估值'
            print(f' {task:<28}{g:>10.4f}{th:>10.4f}{ratio:>7.2f}x  {data:<6} {verdict}')
        if not args.all and len(rated) > len(shown):
            print(f' … 另有 {len(rated) - len(shown)} 个已评测任务未列出（--all 看全部）')
        n_pass = sum(1 for _, g, th, _ in rated if g <= th)
        n_val = sum(1 for _, _, _, s in rated if s == 'val')
        print('-' * W)
        print(f' 表内 {len(rated)}/{len(rows)} 个任务有数（其中**实测** {n_val} 个，其余为 scout 估值）；')
        print(f' 按门槛"已过线"的 {n_pass} 个 —— ⚠️ 这只是**指标过线**，不等于已通关；')
        print(' 正式通关以 AL registry 为准（unit 末尾判定，见上面"通关"行）。')
    else:
        print('-' * W)
        print(' 暂无逐任务评测数据（还没到第一次评测）')
    if unrated and args.all:
        print(f' 未评测: {", ".join(t for t, *_ in unrated[:20])}' +
              (' …' if len(unrated) > 20 else ''))
    print('=' * W)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
