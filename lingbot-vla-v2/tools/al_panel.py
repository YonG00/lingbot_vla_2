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
    return ''            # 找不到任何 run ⇒ 空（下面给提示）


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


def read_series(events: Path) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """收集**带 step 的逐任务指标序列**：`{指标名: {task: [(step, value), ...]}}`。

    用于画"成长轨迹"：`active_val_gmean_mse`（实测 val GMean）、
    `pass_threshold_gmean_mse`（当时的门槛）、`val_to_pass_threshold`（现成倍率）。
    """
    out: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    if not events.is_file():
        return out
    try:
        with events.open(encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if '"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                name = d.get('name') or ''
                step = d.get('step')
                val = d.get('value')
                if not name.startswith('task/') or step is None or not isinstance(val, (int, float)):
                    continue
                parts = name.split('/', 2)
                if len(parts) != 3:
                    continue
                metric, task = parts[2], parts[1]
                out.setdefault(metric, {}).setdefault(task, []).append((int(step), float(val)))
    except OSError:
        pass
    for metric in out:
        for task in out[metric]:
            out[metric][task].sort()
    return out


def derive_status(events: Path) -> Tuple[Dict[str, str], List[str]]:
    """从事件流**推导**每个任务的当前状态，返回 (task -> status, 当前 PASS 的任务列表)。

    为什么需要它：`curriculum/passed_tasks` 这类**指标**只在 **unit 末尾**记录
    （每 `eval_interval_steps` 步一次），而**状态变更**（bootstrap/select/review 的 result）
    是**即时**的。真机踩过：bootstrap 阶段 3 个任务已判 `pass`（免费 PASS），
    但指标还没写 ⇒ 面板显示"通关 0/15"，被误读成"扫描没过"。

    状态来源（按时间顺序覆盖）：
      * `bootstrap.result == 'pass'`  ⇒ PASS（bootstrap 免费 PASS，含 4-traj confirm 通过）
      * `bootstrap.result == 'confirm_failed'` / `'candidate'` ⇒ CANDIDATE
      * `review.outcomes[*].forgotten == true` ⇒ FORGOTTEN（PASS 被遗忘）
      * `train_unit` / `select` 的 task ⇒ TRAINING
    """
    status: Dict[str, str] = {}
    if not events.is_file():
        return status, []
    try:
        with events.open(encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if '"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                action = d.get('action')
                task = d.get('task')
                if action == 'bootstrap' and task:
                    status[str(task)] = 'PASS' if d.get('result') == 'pass' else 'CANDIDATE'
                elif action == 'review':
                    for o in (d.get('outcomes') or []):
                        t = o.get('task')
                        if not t:
                            continue
                        if o.get('forgotten'):
                            status[str(t)] = 'FORGOTTEN'
                        elif o.get('confirmed'):
                            status[str(t)] = 'PASS'
                elif action in ('select', 'train_unit') and task:
                    if status.get(str(task)) != 'PASS':
                        status[str(task)] = 'TRAINING'
    except OSError:
        pass
    passed = sorted(t for t, st in status.items() if st == 'PASS')
    return status, passed


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
    ev_path = ROOT / f'train_{run}' / 'auto_learning_events.jsonl'
    last, per_task = read_events(ev_path)
    derived_status, derived_passed = derive_status(ev_path)
    series = read_series(ev_path)
    nrank = rank_count()
    step, sit, nsamples = step_progress(ROOT / 'logs' / f'train_{run}.log', cfg['interval'])  # type: ignore[arg-type]

    # 🔴 "通关数"优先用**事件推导**（即时），registry 指标（unit 末尾才有）用于交叉校验
    reg_passed = last.get('curriculum/passed_tasks')
    reg_passed = int(reg_passed) if isinstance(reg_passed, (int, float)) else None
    passed = len(derived_passed) if derived_passed else (reg_passed or 0)
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
    if reg_passed is not None and reg_passed != passed:
        print(f'  ⚠️ registry 指标 curriculum/passed_tasks={reg_passed} 与事件推导 {passed} 不同'
              '（前者只在 unit 末尾记录，后者即时；unit 末应自动对齐）')
    if derived_passed:
        print(f' 已通关任务: {", ".join(derived_passed)}')
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
    # ---- 当前任务的成长轨迹（**按任务绑定**；最近 3 次对照）----
    # 🔴 用户要求：这三次历史必须**针对正在评测的那个任务**；任务切换后要看**新任务自己的**历史，
    #    新任务若还没有历史 ⇒ 明确标"没有"（不要拿别的任务的数据顶上）。
    val_all = series.get('active_val_gmean_mse') or {}
    loss_all = series.get('unit_loss') or {}
    # 🔴 选"被查看的任务"：**最近一次走完 unit 的任务** = `unit_loss` 出现过的最大 step 对应的任务。
    #    为什么不用 `curriculum/current_task_name`：任务切换发生在 unit 边界，同一 step 里
    #    它会被**写两次**（先旧任务、后新任务），取"最新值"会得到**刚被选中、还没历史**的新任务。
    #    实测（step 150）：`pick_dual_bottles` 的 unit_loss 先写，随后
    #    `current_task_name` 先= pick_dual_bottles 再= shake_bottle_horizontally。
    cand = [(seq[-1][0], t) for t, seq in loss_all.items() if seq]
    cur_task = max(cand)[1] if cand else None
    if cur_task is None:      # 还没走完任何 unit ⇒ 退化为"有 val 数据的任务"
        cand2 = [(seq[-1][0], t) for t, seq in val_all.items() if seq]
        cur_task = max(cand2)[1] if cand2 else None
    named = last.get('curriculum/current_task_name')
    if isinstance(named, str) and named and named != cur_task:
        print(f' 注: AL 当前正在训 {named}（本 unit 还没走完 ⇒ 无历史）；'
              f'下面展示的是**最近走完 unit 的任务** {cur_task} 的历史')
    print('-' * W)
    if not cur_task:
        print(' 任务成长轨迹: 没有（事件流里还没有任何 active_val 指标）')
        print('=' * W)
        return 0
    val_seq = list(val_all.get(str(cur_task)) or [])
    th_map = dict((series.get('pass_threshold_gmean_mse') or {}).get(str(cur_task)) or [])
    ra_map = dict((series.get('val_to_pass_threshold') or {}).get(str(cur_task)) or [])
    lo_map = dict((series.get('unit_loss') or {}).get(str(cur_task)) or [])
    print(f' 任务成长轨迹: {cur_task}（**只统计该任务自己的历史**；每 unit 一个点）')
    if not val_seq:
        print('   没有（该任务还没有 active_val 记录）')
    else:
        print(f'   {"step":>7}{"GMean(val)":>12}{"门槛":>11}{"倍率":>8}{"unit_loss":>11}  {"类型":<4} 相对上一次')
        prev_ratio = None
        shown = 0
        for st, g in val_seq:
            th = th_map.get(st)
            ratio = ra_map.get(st) or ((g / th) if th else None)
            loss = lo_map.get(st)
            if prev_ratio is None:
                delta = '—（首次记录）'
            else:
                d = (ratio - prev_ratio) if ratio is not None else None
                if d is None:
                    delta = '—'
                elif d < -0.005:
                    delta = f'↓ 改善 {abs(d):.3f}'
                elif d > 0.005:
                    delta = f'↑ 变差 +{d:.3f}'
                else:
                    delta = '→ 持平'
            kind = '基线' if loss is None else '训后'
            print(f'   {st:>7}{g:>12.4f}{(f"{th:.4f}" if th else "—"):>11}'
                  f'{(f"{ratio:.2f}x" if ratio is not None else "—"):>8}'
                  f'{(f"{loss:.4f}" if loss is not None else "—"):>11}  {kind}  {delta}')
            prev_ratio = ratio if ratio is not None else prev_ratio
            shown += 1
        print(f'   （该任务共 {shown} 次评测记录'
              + ('；不足 3 次 ⇒ 上上上次/上上次显示"没有"' if shown < 3 else '') + '）')
        # 明确列出"上上次/上上次/上上上次"三个槽位（用户口径）
        slots = []
        for k, label in ((1, '上一次'), (2, '上上次'), (3, '上上上次')):
            if len(val_seq) >= k:
                st, g = val_seq[-k]
                th = th_map.get(st)
                ratio = ra_map.get(st) or ((g / th) if th else None)
                slots.append(f'{label}=step {st} / {ratio:.2f}x' if ratio is not None
                             else f'{label}=step {st} / 门槛缺失')
            else:
                slots.append(f'{label}：没有')
        print('   三次对照: ' + ' | '.join(slots))
        if len(val_seq) >= 2:
            r0 = ra_map.get(val_seq[0][0])
            r1 = ra_map.get(val_seq[-1][0])
            if r0 and r1:
                gain = (r0 - r1) / r0 * 100
                print(f'   合计: {r0:.2f}x → {r1:.2f}x（{"改善" if gain > 0 else "变差"} {abs(gain):.1f}%）')

    if unrated and args.all:
        print(f' 未评测: {", ".join(t for t, *_ in unrated[:20])}' +
              (' …' if len(unrated) > 20 else ''))
    print('=' * W)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
