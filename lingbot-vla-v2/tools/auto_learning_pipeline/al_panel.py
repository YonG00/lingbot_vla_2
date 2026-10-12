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
from collections import Counter
import json
import os
import re
import statistics
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(os.environ.get('AL_ROOT', '/models/robotwin-persistent/al_runs'))
W = 76          # 行宽：中文按 2 列算，76 + 中文行留余量


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def sh(cmd: List[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return ''


def discover_trainers() -> List[Dict[str, object]]:
    """发现机上**所有**训练器（torchrun 进程组），每个一档。

    为什么不能只数一个数字：一台机器上**可以同时跑多个 run**（实测 al_v41 六卡 + al_v42 两卡），
    旧实现把「所有 torchrun 的直接子进程」加在一起 ⇒ 显示 `8 rank`，对任何单个 run 都是错的。

    判据（全部来自 /proc 事实，不写死 run 名/卡数）：
      * **组**   = 一个 `torch.distributed.run` 进程；
      * **ranks** = 该 torchrun 的**直接子进程**数（并与其 `WORLD_SIZE` 交叉校验）；
      * **run 名** = rank 子进程环境里 `TRAIN_OUT` 的 basename（如 `train_al_v42` ⇒ `al_v42`），
        缺失时退到 `AL_CFG` / cmdline 里的 CONFIG 路径；
      * 另收 `CUDA_VISIBLE_DEVICES`（占哪几张卡）、`AL_MODEL_NAME`（基底）、`PYTHONPATH`（哪个代码副本）、
        `/proc/<pid>/stat` 的启动时间（用于"最新"兜底）。
    """
    pids = _pids()
    parent: Dict[str, str] = {}
    for pid in pids:
        parent[pid] = _ppid(pid)
    out: List[Dict[str, object]] = []
    for tp in pids:
        if 'torch.distributed.run' not in _cmdline(tp):
            continue
        kids = [c for c in pids if parent.get(c) == tp]
        if not kids:
            continue
        env = _environ(kids[0])
        train_out = env.get('TRAIN_OUT', '')
        # 🔴 判据必须是 **basename**：TRAIN_OUT 是绝对路径（`/models/…/train_al_v42`），
        #    对整串做 `startswith('train_')` 恒为 False ⇒ 会掉进下面的回退正则，
        #    而回退正则会被 `al_eval2_*.yaml` 里的 `al_eval2` 命中 ⇒ 解析成错误的 `al_v2`。
        _base = Path(train_out).name
        run = _base[len('train_'):] if _base.startswith('train_') else ''
        al_cfg = env.get('AL_CFG', '')
        if not run and al_cfg:
            # 回退只认"**以 al_v<数字> 结尾**"的运行名目录，避免误吃 al_eval2 这类配置名
            m = re.search(r'(?:^|/)(al_v\d+)(?:/|$)', al_cfg)
            run = m.group(1) if m else ''
        world = env.get('WORLD_SIZE', '')
        started = min((_start_ticks(c) for c in kids), default=0)
        out.append({
            'pgid': tp,
            'run': run,
            'ranks': len(kids),
            'world_size': int(world) if str(world).isdigit() else None,
            'gpus': env.get('CUDA_VISIBLE_DEVICES', ''),
            'model': env.get('AL_MODEL_NAME', ''),
            'workspace': env.get('PYTHONPATH', ''),
            'al_cfg': al_cfg,
            'started': started,
        })
    out.sort(key=lambda t: -int(t['started']))          # 最新启动的排前面
    return out


def find_trainer(trainers: List[Dict[str, object]], run: str) -> Optional[Dict[str, object]]:
    """挑出属于 `run` 的训练器；挑不到返回 None（**不拿别的 run 的数字充数**）。"""
    for t in trainers:
        if t['run'] and t['run'] == run:
            return t
    for t in trainers:                                   # 退一步：按 run 目录出现在哪条环境里匹配
        # 退一步：按"run 目录路径"精确匹配（**不做宽子串**：`al_v4` 会误命中 `al_v41`）
        if run:
            want = f'/train_{run}'
            for t in trainers:
                hay = ' '.join(str(t.get(k, '')) for k in ('al_cfg', 'workspace', 'model'))
                if want in hay:
                    return t
    return None


def _fmt_trainer(t: Dict[str, object]) -> str:
    src = '进程' if t['run'] else '（run 名未识别）'
    head = f"{src} {t['ranks']} rank"
    if t['gpus']:
        head += f" · GPU{t['gpus']}"
    if t['run']:
        head += f" · {t['run']}"
    return head


def print_trainers(trainers: List[Dict[str, object]], run: str) -> None:
    """列出机上所有训练器（多 run 并存时一眼看清谁在跑、几个 rank、占哪几张卡）。"""
    print(f' 训练器: 共 {len(trainers)} 个' if trainers else ' 训练器: 无（未发现 torchrun）')
    for t in trainers:
        mark = ' ← 本 run' if t['run'] == run else ''
        extra = f" | 基底 {Path(str(t['model'])).name}" if t['model'] else ''
        ws = Path(str(t['workspace'])).name if t['workspace'] else ''
        extra += f' | 代码 {ws}' if ws else ''
        print(f"   · {_fmt_trainer(t)}{extra}{mark}")


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


def _environ(pid: str) -> Dict[str, str]:
    """只读**训练 rank 子进程**的 environ（2~8 个），不做全表扫描 —— 快且省。"""
    try:
        with open(f'/proc/{pid}/environ', 'rb') as fh:
            raw = fh.read().decode('utf-8', 'replace')
    except OSError:
        return {}
    out: Dict[str, str] = {}
    for kv in raw.split('\0'):
        if '=' in kv:
            k, v = kv.split('=', 1)
            out[k] = v
    return out


def _start_ticks(pid: str) -> int:
    """进程启动时刻（/proc/<pid>/stat 第 22 字段，单位 tick）—— 用于"最新训练器"兜底。"""
    try:
        with open(f'/proc/{pid}/stat', 'rb') as fh:
            body = fh.read().decode('utf-8', 'replace').rsplit(')', 1)[1].split()
        return int(body[19])                              # state(0) 之后第 20 项 = starttime
    except (OSError, IndexError, ValueError):
        return 0


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


# --------------------------------------------------------------------------- #
# DCP 注册表（池子状态的**权威**来源；2026-10-11 加）
# --------------------------------------------------------------------------- #
#: 注册表里的状态 → 展示顺序（越靠前越"需要关注"）
_STATUS_ORDER = ('TRAINING', 'DEFER', 'PASS', 'CANDIDATE', 'EXHAUSTED', 'FORGOTTEN')


def latest_dcp(run_dir: Path) -> Optional[Path]:
    """取 `<run_dir>/checkpoints` 下**步数最大**的那份 DCP（没有则 None）。"""
    cks = [p for p in (run_dir / 'checkpoints').glob('global_step_*') if p.is_dir()]
    if not cks:
        return None
    try:
        return sorted(cks, key=lambda p: int(p.name.rsplit('_', 1)[-1]))[-1]
    except (ValueError, IndexError):
        return cks[-1]


def load_registry(run_dir: Path) -> Tuple[Optional[Dict[str, dict]], Optional[int], Optional[Path]]:
    """读 DCP `extra_state` 里的 AL 注册表 → (records, dcp_step, dcp_dir)。

    🔴 为什么读 DCP 而不是从事件流推导（真机实测）：事件流推不出 `DEFER` / `EXHAUSTED`
    （只有 PASS/CANDIDATE 两类），也读不到 `attempt_count` / `lp50` / `overfit` /
    `last_transition_reason`；且**重启后事件流叠加**会让推导错乱。
    DCP 里 `auto_learning.registry.records[task]` 字段齐全，是唯一权威源。

    ⚠️ 只读 + `map_location='cpu'`：不占 GPU、不触碰正在跑的训练；
    取文件失败/半写状态一律**降级**（返回 None，由调用方退回事件流并提示）。
    """
    dcp = latest_dcp(run_dir)
    if dcp is None:
        return None, None, None
    f = dcp / 'extra_state' / 'extra_state_rank_0.pt'
    if not f.is_file():
        return None, None, dcp
    try:
        import torch  # 惰性 import：只有真要读 DCP 时才需要
        d = torch.load(f, map_location='cpu', weights_only=False)
        al = (d or {}).get('auto_learning') or {}
        reg = (al.get('registry') or {}).get('records') or {}
        step = (al.get('scheduler') or {}).get('global_step')
        return (reg or None), (int(step) if isinstance(step, (int, float)) else None), dcp
    except Exception as exc:  # noqa: BLE001 —— 读不到不该让面板失败
        print(f'  ⚠️ 读 DCP 注册表失败（{type(exc).__name__}: {str(exc)[:60]}）⇒ 池子改用事件流推导')
        return None, None, dcp


def pool_counts(records: Dict[str, dict]) -> Dict[str, int]:
    """状态计数（只统计已知状态；未知状态归入原样 key）。"""
    out: Dict[str, int] = {}
    for rec in records.values():
        st = str((rec or {}).get('status') or '?')
        out[st] = out.get(st, 0) + 1
    return out


def _ratio(rec: dict, thresholds: Dict[str, float], *, key: str = 'current_val_gmean_mse') -> Optional[float]:
    """倍率 = GMean / 门槛（越小越好；≤1 ⇒ 过线）。

    🔴 字段口径（真机核对 DCP 后的结论）：
      * `current_val_gmean_mse` = **最近一次实测** val GMean ⇒ 进行中任务也有值（首选）；
      * `best_gmean_mse` = **历史最好**，但**只在判 PASS 后才会被填**（进行中为 None）；
      * `scout_gmean_mse` = 初筛缓存值（未训过的任务可用）。
    按 current → best → scout 依次回退 ⇒ 任何阶段都有可显示的数。
    """
    th = thresholds.get(str(rec.get('task_name') or ''))
    if not th:
        return None
    for k in (key, 'best_gmean_mse', 'scout_gmean_mse'):
        g = rec.get(k)
        if isinstance(g, (int, float)) and g:
            return float(g) / float(th)
    return None


def _cur_task_line(records: Dict[str, dict], thresholds: Dict[str, float]) -> str:
    """`当前任务: X（attempt 1/2，已训 100 步，best 倍率 0.86x）`"""
    for task, rec in records.items():
        if str((rec or {}).get('status')) != 'TRAINING':
            continue
        att = rec.get('attempt_count')
        steps = rec.get('attempt_step')
        r = _ratio(dict(rec, task_name=task), thresholds)
        bits = []
        if att is not None:
            # ⚠️ 注册表**不含** max_attempts_per_task（那是 config 项）⇒ 不写 `/2`，
            #    改写"第 N 次尝试"，需要上限时看配置（默认 2）。
            bits.append(f'第 {int(att)} 次尝试')
        if steps is not None:
            bits.append(f'已训 {int(steps)} 步')
        if r is not None:
            bits.append(f'当前倍率 {r:.2f}x')
        if rec.get('overfit'):
            bits.append('⚠️过拟合')
        return f' 当前任务: {task}（{("，".join(bits)) if bits else "—"}）'
    # 没有 TRAINING 记录（例如正处 transition 的瞬间）⇒ 退回 scheduler 的 current_task 提示
    return ''


def print_pool_summary(records: Dict[str, dict], thresholds: Dict[str, float],
                       target: int) -> None:
    """默认视图的 2 行池子摘要（**权威口径**：来自 DCP 注册表）。"""
    cnt = pool_counts(records)
    total = sum(cnt.values()) or 1
    order = [st for st in _STATUS_ORDER if st in cnt] + [st for st in cnt if st not in _STATUS_ORDER]
    parts = [f'{st} {cnt[st]}' for st in order]
    print(f' 池子: {" | ".join(parts)}   （共 {total} 个任务）')
    line = _cur_task_line(records, thresholds)
    if line:
        print(line)


def print_pool_detail(records: Dict[str, dict], thresholds: Dict[str, float]) -> None:
    """`--pool`：逐任务明细（状态 / attempts / 已训步 / best倍率 / 进步 / 过拟合 / DEFER原因）。"""
    print('-' * 78)
    print(' 池子明细（来源：DCP 注册表；倍率 = best_gmean_mse / 门槛，≤1 过线）')
    print(f' {"任务":28}{"状态":10}{"att":>4}{"步":>6}{"当前":>8}{"best":>8}{"lp50":>8}{"lp_tr":>8}{"gap":>7}  过拟合/原因')
    order = {st: i for i, st in enumerate(_STATUS_ORDER)}
    rows = sorted(records.items(),
                  key=lambda kv: (order.get(str((kv[1] or {}).get('status')), 99), kv[0]))
    for task, rec in rows:
        rec = rec or {}
        st = str(rec.get('status') or '?')
        att = rec.get('attempt_count')
        steps = rec.get('attempt_step')
        r = _ratio(dict(rec, task_name=task), thresholds)
        rb = _ratio(dict(rec, task_name=task), thresholds, key='best_gmean_mse')
        lp50 = rec.get('lp50')
        lptr = rec.get('lp_train')
        gap = rec.get('train_val_gap_ratio')
        reason = str(rec.get('last_transition_reason') or '')
        if len(reason) > 40:
            reason = reason[:38] + '…'
        print(f' {task:28}{st:10}'
              f'{int(att) if isinstance(att, (int, float)) else "-":>4}'
              f'{int(steps) if isinstance(steps, (int, float)) else "-":>6}'
              f'{(f"{r:.2f}x" if r is not None else "-"):>8}'
              f'{(f"{rb:.2f}x" if rb is not None else "-"):>8}'
              f'{(f"{lp50:+.3f}" if isinstance(lp50, (int, float)) else "-"):>8}'
              f'{(f"{lptr:+.3f}" if isinstance(lptr, (int, float)) else "-"):>8}'
              f'{(f"{gap:.2f}" if isinstance(gap, (int, float)) else "-"):>7}'
              f'  {"⚠️" if rec.get("overfit") else "  "} {reason}')


def print_progress(records: Dict[str, dict], thresholds: Dict[str, float]) -> None:
    """`--progress`：只列**被训练过**的任务（attempt ≥ 1），按 lp50 降序。"""
    rows = [(t, r or {}) for t, r in records.items()
            if isinstance((r or {}).get('attempt_count'), (int, float))
            and int((r or {}).get('attempt_count')) >= 1]
    rows.sort(key=lambda kv: (kv[1].get('lp50') if isinstance(kv[1].get('lp50'), (int, float))
                              else -9e9), reverse=True)
    print('-' * 78)
    print(' 进步情况（只列被训练过的任务；lp50 = val 进步幅度，越大越好；gap = train/val 差距）')
    print(f' {"任务":28}{"状态":10}{"步":>6}{"lp50":>9}{"lp_train":>10}{"gap":>7}  过拟合')
    for task, rec in rows:
        lp50 = rec.get('lp50')
        lptr = rec.get('lp_train')
        gap = rec.get('train_val_gap_ratio')
        print(f' {task:28}{str(rec.get("status") or "?"):10}'
              f'{int(rec.get("attempt_step") or 0):>6}'
              f'{(f"{lp50:+.4f}" if isinstance(lp50, (int, float)) else "-"):>9}'
              f'{(f"{lptr:+.4f}" if isinstance(lptr, (int, float)) else "-"):>10}'
              f'{(f"{gap:.2f}" if isinstance(gap, (int, float)) else "-"):>7}'
              f'  {"⚠️" if rec.get("overfit") else " "}')
    if not rows:
        print('  （还没有被训练过的任务）')


def print_traces(records: Dict[str, dict], thresholds: Dict[str, float],
                 series: Dict[str, Dict[str, List[Tuple[int, float]]]], n: int) -> None:
    """`--traces N`：对**每个被训练过的任务**画最近 N 个 unit 的 GMean/倍率。"""
    g = series.get('active_val_gmean_mse') or {}
    ratio = series.get('val_to_pass_threshold') or {}
    tasks = [t for t, r in records.items()
             if isinstance((r or {}).get('attempt_count'), (int, float))
             and int((r or {}).get('attempt_count')) >= 1]
    print('-' * 78)
    print(f' 多任务成长轨迹（最近 {n} 个 unit；倍率 ≤1 表示过线）')
    if not tasks:
        print('  （还没有被训练过的任务）')
        return
    for task in tasks:
        pts = (g.get(task) or [])[-n:]
        if not pts:
            continue
        st = str((records.get(task) or {}).get('status') or '?')
        print(f'  {task}（{st}）')
        rm = dict(ratio.get(task) or [])
        for stp, val in pts:
            r = rm.get(stp)
            extra = f'  倍率 {r:.2f}x' if isinstance(r, (int, float)) else ''
            print(f'    step {stp:>6}  GMean {val:.4f}{extra}')


def watch_loop(seconds: int, fn) -> int:
    """`--watch S`：每 S 秒重绘一次（只读；Ctrl-C 退出）。"""
    import time as _t
    try:
        while True:
            print('\033[2J\033[H', end='')      # 清屏 + 光标归位
            fn()
            _t.sleep(max(2, int(seconds)))
    except KeyboardInterrupt:
        print('（已退出 watch）')
        return 0


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
                elif action in ('pass', 'apply_pass') and task:
                    status[str(task)] = 'PASS'
                elif action in ('select', 'train_unit') and task:
                    if status.get(str(task)) != 'PASS':
                        status[str(task)] = 'TRAINING'
    except OSError:
        pass
    passed = sorted(t for t, st in status.items() if st == 'PASS')
    return status, passed


def step_progress(log: Path, interval: Optional[int]) -> Tuple[Optional[int], Optional[float], int, Optional[float]]:
    """返回 (最新 step, **稳态** s/it 中位数, 采样步数, 最新一步的 s/it)。

    🔴 为什么必须区分"稳态"和"最新一步"（真机踩坑）：
        `s/it` 是**平均**步时；unit 边界那一步要跑开环评测 ⇒ 实测被抬到 **276.95s/it**
        （同期稳态是 13 s/it 量级）。只看最新一步会误判成"训练变慢 20 倍"。
        判据：`interval`（评测间隔，默认 50）—— step % interval == 0 的样本算**边界步**，剔除后取中位数。

    🔴 另一个坑：日志里进度条用 `\\r` 分隔，切片 400KB **可能只含 1 个样本** ⇒
        "中位数"退化成单点。这里先把 `\\r` 归一成 `\\n` 再匹配，并把窗口放大到 2 MB。
    """
    if not log.is_file():
        return None, None, 0, None
    try:
        tail = log.read_text(encoding='utf-8', errors='replace')[-2_000_000:]
    except OSError:
        return None, None, 0, None
    tail = tail.replace('\r', '\n')                       # 进度条分隔符归一
    hits = re.findall(r'Step:\s*(\d+)/\d+\s*\[[^]]*?([\d.]+)s/it', tail)
    if not hits:
        m = re.findall(r'Step:\s*(\d+)/', tail)
        return (int(m[-1]) if m else None), None, 0, None
    latest = float(hits[-1][1])
    iv = int(interval) if interval else 50
    steady_rates = [float(b) for a, b in hits if iv and int(a) % iv != 0]   # 剔掉边界步
    if len(steady_rates) < 3:                            # 稳态样本太少 ⇒ 退回全部
        steady_rates = [float(b) for _, b in hits]
    return int(hits[-1][0]), statistics.median(steady_rates[-20:]), len(hits), latest


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description='AL 训练仪表盘')
    ap.add_argument('run', nargs='?', default=None, help='run 名（默认自动选最新）')
    ap.add_argument('--all', action='store_true', help='列出全部任务（默认只列最接近通关的若干）')
    ap.add_argument('--trainers', action='store_true',
                    help='只列出机上所有训练器（多 run 并存时用；每个一行）')
    ap.add_argument('--top', type=int, default=4,
                    help='默认显示多少个任务（按倍数升序；默认 4 以便整屏可见；--all 看全部）')
    ap.add_argument('--pool', action='store_true',
                    help='池子明细（状态/attempts/已训步/倍率/进步/过拟合/DEFER 原因；来源 DCP 注册表）')
    ap.add_argument('--progress', action='store_true',
                    help='进步情况（只列被训练过的任务，按 lp50 降序）')
    ap.add_argument('--traces', type=int, nargs='?', const=3, default=None,
                    metavar='N', help='多任务成长轨迹（每个被训练过的任务最近 N 个 unit，默认 3）')
    ap.add_argument('--watch', type=int, nargs='?', const=30, default=None, metavar='S',
                    help='每 S 秒刷新一次（默认 30；Ctrl-C 退出）')
    args = ap.parse_args()

    if getattr(args, 'trainers', False):
        _ts = discover_trainers()
        print('=' * 78)
        print(' AL 训练器（torchrun 进程组）')
        print('=' * 78)
        print_trainers(_ts, '')
        return 0

    run = pick_run(args.run)
    run_dir = (ROOT / f'train_{run}') if run else None
    if not run or run_dir is None or not run_dir.is_dir():
        # 🔴 三种情况必须分开说（真机踩坑：新 run 在预扫阶段**目录尚未创建**，
        #    旧提示语一律说"未找到 run"，让人误以为 run 名写错）：
        #    ① 进程在跑但目录没建 ⇒ 仍在启动/预扫；
        #    ② 进程不在跑且目录没建 ⇒ 真的没有这个 run；
        #    ③ 目录在但事件流为空 ⇒ 刚起，还没走到第一个 unit。
        running = [t for t in discover_trainers() if t.get('run') == run]
        existing = sorted(q.name[len('train_'):] for q in ROOT.glob('train_*') if q.is_dir())
        print('=' * 78)
        print(f' AL 仪表盘  run={run or "(未指定)"}')
        print('=' * 78)
        if running:
            t0 = running[0]
            print(' 状态: 已启动但**尚未进入训练**（run 目录未创建 ⇒ 仍在预扫/初始化）')
            print(f'   · {_fmt_trainer(t0)} | 代码 {Path(str(t0["workspace"])).name}')
            log = Path('/workspace/runtime/logs') / f'launch_{run}.log'
            if log.is_file():
                try:
                    tail = log.read_text(encoding='utf-8', errors='replace')[-6000:]
                except OSError:
                    tail = ''
                prog = re.findall(r'合计 (\d+/\d+)', tail)
                if prog:
                    print(f'   · 预扫进度: {prog[-1]}（日志 {log}）')
                else:
                    print(f'   · 启动日志: {log}')
            else:
                print(f'   · 启动日志: 未找到 {log}')
            print(f'   · 进入训练后（目录出现 {run_dir}）本面板自动有数据')
        else:
            print(f' 状态: 未找到 run（ROOT={ROOT}）')
            if existing:
                print(f' 现有 run: {", ".join(existing[-8:])}')
            print(' 用法: al_panel.py <run名>；或用 AL_ROOT=<al_runs 目录> 覆盖根目录')
        print('=' * 78)
        return 2
    cfg = run_cfg(run)
    thresholds = load_thresholds(cfg['thresholds'])          # type: ignore[arg-type]
    target = int(cfg['target'] or 15)
    ev_path = ROOT / f'train_{run}' / 'auto_learning_events.jsonl'
    last, per_task = read_events(ev_path)
    derived_status, derived_passed = derive_status(ev_path)
    series = read_series(ev_path)
    trainers = discover_trainers()
    trainer = find_trainer(trainers, run)
    step, sit, nsamples, latest_sit = step_progress(ROOT / 'logs' / f'train_{run}.log', cfg['interval'])  # type: ignore[arg-type]

    # 🔴 通关数口径（真机纠错两次后的结论）：
    #    * **registry 指标 `curriculum/passed_tasks` 是权威**（scheduler 自己算，含训练中新增 PASS）；
    #    * 事件推导只覆盖 bootstrap/select 的 pass，**会漏掉训练中新增的 PASS** ⇒ 只作交叉校验，
    #      且在它**更大**时以它为准（真机实测：step 200 时 registry=4、推导=3）。
    reg_passed = last.get('curriculum/passed_tasks')
    reg_passed = int(reg_passed) if isinstance(reg_passed, (int, float)) else None
    passed = reg_passed if reg_passed is not None else len(derived_passed)
    newly = last.get('curriculum/newly_passed_count')
    boot = last.get('curriculum/bootstrap_pass_count')
    print('=' * 78)
    print(f' AL 仪表盘  run={run}   目标 PASS={target}')
    _th = Path(str(cfg['thresholds'] or '?')).name
    _iv = f'；每 {cfg["interval"]} 步判定一次' if cfg['interval'] else ''
    print(f' 门槛: {_th.replace("pass_thresholds_", "")}（{len(thresholds)} 任务）{_iv}')
    # 🔴 状态行只报**本 run 自己的训练器**；机上有多个训练器时**绝不合并计数**
    #    （旧实现把 v41 的 6 rank + v42 的 2 rank 加成 8 ⇒ 对两个 run 都错）。
    if trainer:
        _r = int(trainer['ranks'])                        # type: ignore[arg-type]
        ws = trainer['world_size']
        st_line = f' 状态: 训练中（{_r} rank'
        if ws is not None and int(ws) != _r:              # type: ignore[arg-type]
            st_line += f'/{ws}⚠️'
        st_line += f' · GPU{trainer["gpus"]}）' if trainer['gpus'] else '）'
    elif not trainers:
        st_line = ' 状态: 未在训练（未发现 torchrun 进程）'
    else:
        st_line = ' 状态: 本 run 无进程（机上另有训练器在跑，见下方"训练器"）'
    if step is not None:
        st_line += f' | Step {step}'
    # 🔴 步时必须**分开报**：`s/it` 是平均值，边界步含评测（实测 276.9s/it vs 稳态 13s/it）；
    #    只报一个数会被读成"训练变慢 20 倍"。
    if sit:
        st_line += f' | 稳态 {sit:.1f}s/it'
        if latest_sit is not None and sit and latest_sit > sit * 1.6:
            st_line += f'（当前步 {latest_sit:.0f}s/it：含评测/扫描）'
    pct = (100.0 * passed / target) if target else 0.0
    _extra = ''
    if isinstance(newly, (int, float)):
        _extra += f' | 新增 {int(newly)}'
    if isinstance(boot, (int, float)) and boot:
        _extra += f' | Bootstrap {int(boot)}'
    print(st_line)
    print(f' 通关: {passed}/{target}（{pct:.0f}%）还差 {max(0, target - passed)}{_extra}'
          + ('' if reg_passed is not None else '（本 run 尚未产生判定）'))
    # 新 run（如 al_v42）自己的事件流里还没有 curriculum/* ⇒ 上面那个数字来自**别的 run/继承口径**，
    # 必须点明，不能让读者以为是本 run 的战绩。
    if reg_passed is None and derived_passed:
        print('   ⚠️ 本 run 尚无 curriculum 指标；下列通关记为**其它 run/历史**，仅供对照')
    if derived_passed:
        _tail = ''
        if reg_passed is not None and len(derived_passed) < reg_passed:
            _tail = f' +{reg_passed - len(derived_passed)} 个训练中新增'
        _names = ', '.join(derived_passed[:2])
        _more = f' 等{len(derived_passed)}个' if len(derived_passed) > 2 else ''
        print(f' 已通关: {_names}{_more}{_tail}')
    if cfg['interval'] and step is not None:
        _rem = int(cfg['interval']) - (step % int(cfg['interval']))
        print(f' 下一次判定: 再过 {_rem} 步（Step {step + _rem}）')
    if len(trainers) > 1 or (trainers and not trainer):
        print_trainers(trainers, run)

    # ---- 池子（**权威来源 = DCP 注册表**；读不到则退回事件流并提示）----
    records, dcp_step, dcp_dir = load_registry(run_dir)
    if records:
        print(f' 池子来源: DCP {dcp_dir.name if dcp_dir else "?"}'
              + (f'（global_step {dcp_step}）' if dcp_step is not None else ''))
        print_pool_summary(records, thresholds, target)
        if getattr(args, 'pool', False):
            print_pool_detail(records, thresholds)
        if getattr(args, 'progress', False):
            print_progress(records, thresholds)
        if getattr(args, 'traces', None):
            print_traces(records, thresholds, series, int(args.traces))
    else:
        _st = Counter(derived_status.values()) if derived_status else {}
        if _st:
            print(f' 池子（事件流推导）: ' + ' | '.join(f'{k} {v}' for k, v in sorted(_st.items())))
            print('   ⚠️ 事件流推不出 DEFER/EXHAUSTED 与 attempts；要看权威池子请等 run 落下第一份 DCP')

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
            print(f' {task:<26}{g:>9.4f}{th:>9.4f}{ratio:>6.2f}x  {data:<6} {verdict}')
        if not args.all and len(rated) > len(shown):
            print(f' … 另有 {len(rated) - len(shown)} 个已评测任务未列出（--all 看全部）')
        n_pass = sum(1 for _, g, th, _ in rated if g <= th)
        n_val = sum(1 for _, _, _, s in rated if s == 'val')
        print('-' * W)
        print(f' 表内 {len(rated)}/{len(rows)} 有数（实测 {n_val}）｜"已过线" {n_pass} 个'
              f'（仅指标，非通关）')
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
        print(f' 注: 在训 {named}（unit 未完）；轨迹={cur_task}')
    print('-' * W)
    if not cur_task:
        print(' 任务成长轨迹: 没有（事件流里还没有任何 active_val 指标）')
        print('=' * W)
        return 0
    val_seq = list(val_all.get(str(cur_task)) or [])
    th_map = dict((series.get('pass_threshold_gmean_mse') or {}).get(str(cur_task)) or [])
    ra_map = dict((series.get('val_to_pass_threshold') or {}).get(str(cur_task)) or [])
    lo_map = dict((series.get('unit_loss') or {}).get(str(cur_task)) or [])
    print(f' 成长轨迹: {cur_task}（该任务自身历史，每 unit 一点）')
    if not val_seq:
        print('   没有（该任务还没有 active_val 记录）')
    else:
        print(f'   {"step":>5}{"GMean":>9}{"门槛":>9}{"倍率":>7}{"loss":>9}  {"类型":<4}变化')
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
            print(f'   {st:>5}{g:>9.4f}{(f"{th:.4f}" if th else "—"):>9}'
                  f'{(f"{ratio:.2f}x" if ratio is not None else "—"):>7}'
                  f'{(f"{loss:.4f}" if loss is not None else "—"):>9}  {kind:<4}{delta}')
            prev_ratio = ratio if ratio is not None else prev_ratio
            shown += 1
        if shown < 3:
            print(f'   （仅 {shown} 次记录 ⇒ 缺的槽位显示"没有"）')
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
        print('   三次: ' + '｜'.join(v.replace('step ', '') for v in slots))
        last_r = ra_map.get(val_seq[-1][0])
        prev_r = ra_map.get(val_seq[-2][0]) if len(val_seq) >= 2 else None
        trend = '—'
        if last_r is not None and prev_r is not None:
            trend = ('↓ 在改善' if last_r < prev_r - 0.005
                     else ('↑ 在变差' if last_r > prev_r + 0.005 else '→ 停滞'))
        # 合计（首点 → 末点的倍率变化）先算好，再一行摘要输出
        gain_txt = ''
        if len(val_seq) >= 2:
            r0 = ra_map.get(val_seq[0][0])
            r1 = ra_map.get(val_seq[-1][0])
            if r0 and r1:
                gain = (r0 - r1) / r0 * 100
                gain_txt = f' | {r0:.2f}x→{r1:.2f}x（{"改善" if gain > 0 else "变差"}{abs(gain):.0f}%）'
        print('-' * W)
        _r = f'{last_r:.2f}x' if last_r is not None else '—'
        print(f' 摘要: 通关 {passed}/{target} | {_r} {trend}{gain_txt}')

    if unrated and args.all:
        print(f' 未评测: {", ".join(t for t, *_ in unrated[:20])}' +
              (' …' if len(unrated) > 20 else ''))
    print('=' * W)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
