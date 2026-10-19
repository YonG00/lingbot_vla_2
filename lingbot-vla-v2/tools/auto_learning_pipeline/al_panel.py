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
import math
import os
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
      * **组**   = `torchrun` 可执行文件或 `python -m torch.distributed.run` 进程；
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
        if not _is_torchrun_cmdline(_cmdline(tp)):
            continue
        kids = [c for c in pids if parent.get(c) == tp]
        if not kids:
            continue
        env = _environ(kids[0])
        train_out = env.get('TRAIN_OUT', '')
        # Dedicated launchers pass output_dir on argv, without legacy TRAIN_OUT.
        argv = [a for a in _cmdline(kids[0]).split('\0') if a]
        for i, arg in enumerate(argv):
            if arg == '--train.output_dir' and i + 1 < len(argv):
                train_out = argv[i + 1]
            elif arg.startswith('--train.output_dir='):
                train_out = arg.split('=', 1)[1]
        # 🔴 判据必须是 **basename**：TRAIN_OUT 是绝对路径（`/models/…/train_al_v42`），
        #    对整串做 `startswith('train_')` 恒为 False ⇒ 会掉进下面的回退正则，
        #    而回退正则会被 `al_eval2_*.yaml` 里的 `al_eval2` 命中 ⇒ 解析成错误的 `al_v2`。
        _base = Path(train_out).name
        # 🔴 2026-10-12 修：basename **不一定带 `train_` 前缀**。
        #    旧布局（不传 `--train-out`）是 `train_al_v42`；新布局（显式 `--train-out …/al_v69`）
        #    直接就是 `al_v69` ⇒ 旧代码只认带前缀的 ⇒ `run=''` ⇒ 回退正则匹配不上
        #    ⇒ **进程在跑却显示"无进程!"**（v69 实测）。
        if _base.startswith('train_'):
            run = _base[len('train_'):]
        elif re.fullmatch(r'al_[A-Za-z0-9_]+', _base):
            run = _base
        else:
            run = ''
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


def _is_torchrun_cmdline(cmdline: str) -> bool:
    """兼容 torchrun console script 与等价的 `python -m torch.distributed.run`。"""
    argv = [arg for arg in cmdline.split('\0') if arg]
    if not argv:
        return False
    if 'torch.distributed.run' in argv:
        return True
    return any(Path(arg).name in ('torchrun', 'torchrun.exe') for arg in argv[:2])


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


def resolve_run_dir(run: str) -> Optional[Path]:
    """把 run 名解析成**实际 run 目录**；找不到返回 None。

    🔴 两种布局都要认（2026-10-12 用户实测踩坑：v69 面板报"未找到 run"）：
      · 旧：`<ROOT>/train_<run>/` —— 启动器**不传 `--train-out`** 时（v65 及更早）
      · 新：`<ROOT>/<run>/`       —— 启动器**显式传 `--train-out <ROOT>/<run>`** 时（v69 起）
        两者内部结构相同（`lingbotvla_cli.yaml` / `auto_learning_events.jsonl` / `runs`）。
    判据：`lingbotvla_cli.yaml` 存在（面板取 AL 配置/目标的唯一来源）；目录在但配置未写 ⇒ 也认
    （好让面板能报"启动中"而不是"run 名写错"）。
    """
    if not run:
        return None
    cands = [ROOT / f'train_{run}', ROOT / run]
    for c in cands:
        if (c / 'lingbotvla_cli.yaml').is_file():
            return c
    for c in cands:
        if c.is_dir():
            return c
    return None


def run_cfg(run: str) -> Dict[str, object]:
    """从 run 目录的 CLI 配置里取：AL 配置路径 / 目标 PASS / 评测间隔。"""
    out: Dict[str, object] = {'al_cfg': None, 'target': None, 'interval': None,
                             'thresholds': None, 'max_steps': None,
                             'pass_enabled': True, 'task_scan': None,
                             'hardness_scan': None, 'save_steps': None}
    rd = resolve_run_dir(run)
    p = (rd if rd else ROOT / f'train_{run}') / 'lingbotvla_cli.yaml'
    if not p.is_file():
        return out
    txt = p.read_text(encoding='utf-8', errors='replace')
    for key in ('max_steps', 'save_steps'):
        m = re.search(r'^\s*' + key + r':\s*(\d+)\s*$', txt, re.M)
        if m and int(m.group(1)) > 0:
            out[key] = int(m.group(1))
    m = re.search(r'^\s*auto_learning:\s*(\S+)\s*$', txt, re.M)
    if m:
        out['al_cfg'] = m.group(1).strip("\"'")
    # AL 配置里的目标与阈值表
    al_path = Path(str(out['al_cfg'])).expanduser() if out['al_cfg'] else None
    if al_path and not al_path.is_absolute():
        al_path = next((q for q in (p.parent / al_path, Path(__file__).resolve().parents[2] / al_path)
                        if q.is_file()), al_path)
    if al_path and al_path.is_file():
        a = al_path.read_text(encoding='utf-8', errors='replace')
        try:
            values = json.loads(a)
        except ValueError:
            # Legacy AL files are flat YAML; retain dependency-free parsing.
            values = {}
            for key, value in re.findall(r'^([A-Za-z_]+):\s*([^#\n]+)', a, re.M):
                value = value.strip().strip("\"'")
                values[key] = (value.lower() == 'true' if value.lower() in ('true', 'false')
                               else int(value) if value.isdigit() else value)
        for source, dest in (('target_total_passed_tasks', 'target'),
                             ('pass_thresholds_file', 'thresholds'),
                             ('eval_interval_steps', 'interval'),
                             ('pass_transition_enabled', 'pass_enabled'),
                             ('train_all_episodes', 'train_all_episodes'),
                             ('task_scan_enabled', 'task_scan'),
                             ('hardness_scan_enabled', 'hardness_scan')):
            if source in values:
                out[dest] = values[source]
        if not out['max_steps'] and values.get('max_global_steps', 0) > 0:
            out['max_steps'] = values['max_global_steps']
        if out['thresholds']:
            th = Path(str(out['thresholds']))
            if not th.is_absolute():
                out['thresholds'] = str(next((q for q in (al_path.parent / th,
                    Path(__file__).resolve().parents[2] / th) if q.is_file()), th))
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
                if d.get('action') == 'select' and d.get('task'):
                    last['curriculum/current_task_name'] = str(d['task'])
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


def ee_val_text(series, task, *, kind='active_val'):
    """Read one matching EE eval step from events, never from stale pool state."""
    positions = (series.get(kind + '_ee_position_rmse_mm') or {}).get(task) or []
    if not positions:
        return None
    step, position = positions[-1]
    def value(key):
        return dict((series.get(kind + '_ee_' + key) or {}).get(task) or []).get(step)
    angle, grip = value('orientation_mean_deg'), value('gripper_mae')
    if angle is None or grip is None:
        return None  # tolerate a partially appended event group
    return f'Step {step}  位置 {position:.2f} mm  姿态 {angle:.2f}°  夹爪 {grip:.4f}'


def ee_training_loss_text(series, step):
    """Exact-step, unweighted training losses; never substitute a later step.

    The caller only attaches this to a history row with a task unit_loss.
    TB reports the last micro-batch when GAS>1; this is not a unit mean.
    """
    source = series.get('_training_ee') or {}
    values = []
    for name in ('position', 'orientation', 'gripper'):
        value = dict(source.get(f'ee_fk/{name}_loss') or []).get(step)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        values.append(value)
    return 'EE ' + '/'.join(f'{value:.5f}' for value in values)


def history_threshold_text(steps, thresholds):
    """A fixed threshold belongs in the heading; preserve changing history."""
    values = {thresholds.get(step) for step in steps}
    if len(values) == 1:
        value = next(iter(values))
        return (f'门槛 {value:.4g}' if isinstance(value, (int, float)) else '门槛 --'), False
    return '门槛有变化，逐行标注T', True


def print_train_monitor_history(task, series):
    """全训练模式只展示真实 active_train，绝不借用 val 字段或参考阈值。"""
    print('  模式    全轨迹训练；固定4条训练拟合监控，无独立val')
    ee_text = ee_val_text(series, task, kind='active_train')
    if ee_text:
        print(f'  末端train  {ee_text}')
    for side, label in (('left', '左'), ('right', '右')):
        seq = (series.get(f'active_train_ee_{side}_position_rmse_mm') or {}).get(task) or []
        if seq:
            print(f'  {label}臂train  {seq[-1][1]:.2f} mm (step {seq[-1][0]})')
    seq = (series.get('active_train_gmean_mse') or {}).get(task) or []
    losses = dict((series.get('unit_loss') or {}).get(task) or [])
    if not seq:
        print('  历史    尚无训练拟合记录')
    for step, gmse in seq[-3:]:
        loss = losses.get(step)
        loss_text = f'{loss:.4f}' if isinstance(loss, (int, float)) else '--'
        ee_loss = ee_training_loss_text(series, step) if step in losses else None
        print(f'    {step:>5} train G {gmse:.6f} loss {loss_text}'
              f'{"  " + ee_loss if ee_loss else ""}')
    print('  说明    EE训练项为未加权/该步；训练拟合不代表闭环成功')


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


def read_tb_latest(run_dir: Optional[Path], *, ee_series=None) -> Dict[str, Any]:
    """读 TensorBoard 的**最新标量** → `{tag: (step, value)}`（面板的"全实时"来源）。

    🔴 为什么加它（2026-10-12 用户要求"面板全部实时"）：
      · `pool_state.json` / DCP `extra_state` **只在存档点导出**（`SAVE_EVERY`，本项目 400 步）
        ⇒ 面板的"池子/在训"两行最长落后 400 步（真机 v65 实测：主循环已训
        `place_object_basket`，面板仍显示 2400 快照里的 `click_bell`、`pass=8`，
        而 TB 里 `skill_overview/pass_count` 已经是 14）；
      · TB 是**每 unit 末（≈50 步）实时写**的，且**池子级计数齐全**：
        `skill_overview/{pass,defer,exhausted,candidate}_count`、
        `curriculum/{passed_tasks,newly_passed_count,bootstrap_pass_count,reopened_count,
        forgotten_tasks,units_completed}`。
    ⇒ 池子计数改从 TB 读 ⇒ 无需任何训练侧改动、也无需额外导出文件。

    ⚠️ 只读**最新一个** event 文件（TB 会把超大文件轮转）；读不到 ⇒ 返回空 dict
    （调用方退回旧路径：pool_state.json / DCP / 事件流推导），**绝不让面板失败**。
    """
    out: Dict[str, Any] = {}
    if run_dir is None:
        return out
    try:
        cands = sorted((run_dir / 'runs').glob('events.out.tfevents.*'),
                       key=lambda f: f.stat().st_mtime)
        if not cands:
            return out
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(cands[-1]), size_guidance={'scalars': 0})
        ea.Reload()
        for t in (ea.Tags().get('scalars') or []):
            try:
                vals = ea.Scalars(t)
            except Exception:  # noqa: BLE001
                continue
            if vals:
                out[t] = (int(vals[-1].step), vals[-1].value)
                if ee_series is not None and t in (
                        'ee_fk/position_loss', 'ee_fk/orientation_loss', 'ee_fk/gripper_loss'):
                    ee_series[t] = [(int(point.step), float(point.value)) for point in vals]
    except Exception as exc:  # noqa: BLE001 —— 读 TB 失败不该让面板失败
        print(f'  ⚠️ 读 TensorBoard 失败（{type(exc).__name__}: {str(exc)[:60]}）')
    return out


def latest_dcp(run_dir: Path) -> Optional[Path]:
    """取 `<run_dir>/checkpoints` 下**步数最大**的那份 DCP（没有则 None）。"""
    cks = [p for p in (run_dir / 'checkpoints').glob('global_step_*') if p.is_dir()]
    if not cks:
        return None
    try:
        return sorted(cks, key=lambda p: int(p.name.rsplit('_', 1)[-1]))[-1]
    except (ValueError, IndexError):
        return cks[-1]


def latest_pool_state(run_dir: Path) -> Optional[Path]:
    """取 `<run_dir>/checkpoints/global_step_*` 下**步数最大**的池子存档（没有则 None）。

    🔴 为什么需要它（2026-10-12 真机）：`--train.dcp_save_mode=final_only` 时
    **周期只写 HF、收尾才写 DCP** ⇒ 整个 run 期间 `checkpoints/global_step_N/`
    里**没有 `extra_state`**，面板只能退化成"事件流推导"，于是把
    `DEFER`/`PASS` 全当成 `CANDIDATE`（实测显示 `candidate=49`，真值 27）。

    而**同一次存档**其实已经把权威注册表写在了 HF 旁边：
      `<global_step_N>/pool_state.json` 或 `<global_step_N>/hf_ckpt/pool_state.json`
      （由 `train_lingbotvla.py::_export_pool_state_here` 导出，格式与 DCP 同构）。
    本函数就是为了让面板能找到它 —— 池子/在训/倍率不必等收尾写 DCP。
    """
    cks = [p for p in (run_dir / 'checkpoints').glob('global_step_*') if p.is_dir()]
    if not cks:
        return None
    try:
        cks = sorted(cks, key=lambda p: int(p.name.rsplit('_', 1)[-1]))
    except (ValueError, IndexError):
        pass
    for d in reversed(cks):                       # 步数最大的优先
        for cand in (d / 'pool_state.json', d / 'hf_ckpt' / 'pool_state.json'):
            if cand.is_file():
                return cand
    return None


def load_registry(run_dir: Path) -> Tuple[Optional[Dict[str, dict]], Optional[int], Optional[Path]]:
    """读权威 AL 注册表 → (records, step, 来源路径)。

    读取顺序（**三级回退**，2026-10-12 起）：
      1. **DCP `extra_state`**（`checkpoints/global_step_N/extra_state/extra_state_rank_0.pt`）
         —— 最完整；
      2. **`pool_state.json`**（`global_step_N/pool_state.json` 或 `.../hf_ckpt/pool_state.json`）
         —— `dcp_save_mode=final_only` 时**唯一**的权威来源（见 `latest_pool_state`）；
      3. 都没有 ⇒ 返回 None，由调用方退回"事件流推导"并打警告。

    🔴 为什么不用事件流推导当主源（真机实测）：事件流推不出 `DEFER` / `EXHAUSTED`
    （只有 PASS/CANDIDATE 两类），也读不到 `attempt_count` / `lp50` / `overfit` /
    `last_transition_reason`；且**重启后事件流叠加**会让推导错乱。
    注册表（无论来自 DCP 还是 pool_state.json）字段齐全，是唯一权威源。

    ⚠️ 只读 + `map_location='cpu'`：不占 GPU、不触碰正在跑的训练；
    取文件失败/半写状态一律**降级**（返回 None，由调用方退回事件流并提示）。
    """
    dcp = latest_dcp(run_dir)
    f = (dcp / 'extra_state' / 'extra_state_rank_0.pt') if dcp is not None else None
    if f is not None and f.is_file():
        try:
            import torch  # 惰性 import：只有真要读 DCP 时才需要
            d = torch.load(f, map_location='cpu', weights_only=False)
            al = (d or {}).get('auto_learning') or {}
            reg = (al.get('registry') or {}).get('records') or {}
            step = (al.get('scheduler') or {}).get('global_step')
            return (reg or None), (int(step) if isinstance(step, (int, float)) else None), dcp
        except Exception as exc:  # noqa: BLE001 —— 读不到不该让面板失败
            print(f'  ⚠️ 读 DCP 注册表失败（{type(exc).__name__}: {str(exc)[:60]}）⇒ 尝试 pool_state.json')

    # ---- 回退 2：pool_state.json（final_only 模式下的唯一权威源）----
    pool = latest_pool_state(run_dir)
    if pool is not None:
        try:
            import json
            d = json.loads(pool.read_text(encoding='utf-8'))
            al = (d or {}).get('auto_learning') or {}
            reg = (al.get('registry') or {}).get('records') or {}
            step = (al.get('scheduler') or {}).get('global_step') if al else None
            # pool_state.json 顶层还有 global_step（导出时写入）⇒ 兜底
            if not isinstance(step, (int, float)):
                step = (d or {}).get('global_step')
            if reg:
                return (reg or None), (int(step) if isinstance(step, (int, float)) else None), pool
        except Exception as exc:  # noqa: BLE001
            print(f'  ⚠️ 读 pool_state.json 失败（{type(exc).__name__}: {str(exc)[:60]}）⇒ 退回事件流推导')

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


def _to_sec_any(hms: str) -> Optional[float]:
    """`HH:MM:SS` / `MM:SS` / `SS` → 秒；失败返回 None。"""
    try:
        parts = [int(x) for x in str(hms).split(':')]
    except (ValueError, AttributeError):
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    return float(parts[0] * 3600 + parts[1] * 60 + parts[2])


def _hms_h(sec: Optional[float]) -> str:
    """秒 → `X.Xh`（用户要求以 h 为单位）；None → `?`。"""
    if not isinstance(sec, (int, float)) or sec <= 0:
        return '?'
    return f'{sec / 3600.0:.1f}h'


def train_log_path(run: str) -> Path:
    """Use an explicit log override, then the common launcher log layouts."""
    explicit = os.environ.get('AL_LOG_PATH')
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.is_file():
            return explicit_path
    log_root = Path(os.environ.get('AL_LOG_ROOT', '/data/tmp')).expanduser()
    candidates = [ROOT / 'logs' / f'train_{run}.log',
                  log_root / f'{run}.log']
    existing = [path for path in candidates if path.is_file()]
    if existing:
        return max(existing, key=lambda path: path.stat().st_mtime)
    return candidates[0]


def prefer_tensorboard_progress(log_progress, tb: Dict[str, Any]):
    """Prefer the newest TB scalar step, retaining parsed-log timing as fallback."""
    step, sit, nsamples, latest_sit, elapsed_s, eta_s = log_progress
    point = tb.get('training/loss')
    tb_step = point[0] if isinstance(point, tuple) and len(point) == 2 else None
    if isinstance(tb_step, (int, float)) and math.isfinite(tb_step):
        tb_step = int(tb_step)
        if step is None or tb_step >= step:
            step = tb_step
            tb_time = tb.get('steptime')
            if (isinstance(tb_time, tuple) and len(tb_time) == 2
                    and tb_time[0] == tb_step
                    and isinstance(tb_time[1], (int, float))
                    and math.isfinite(tb_time[1]) and tb_time[1] > 0):
                latest_sit = float(tb_time[1])
                if sit is None:
                    sit = latest_sit
    return step, sit, nsamples, latest_sit, elapsed_s, eta_s


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
        return None, None, 0, None, None, None
    try:
        tail = log.read_text(encoding='utf-8', errors='replace')[-2_000_000:]
    except OSError:
        return None, None, 0, None, None, None
    tail = tail.replace('\r', '\n')                       # 进度条分隔符归一
    # 取 (step, 该行 elapsed 秒, 累计均值 s/it)
    hits = re.findall(r'Step:\s*(\d+)/\d+\s*\[([\d:]+)<[^]]*?([\d.]+)s/it', tail)
    if not hits:
        m = re.findall(r'Step:\s*(\d+)/', tail)
        return (int(m[-1]) if m else None), None, 0, None, None, None
    latest = float(hits[-1][2])
    elapsed_s = _to_sec_any(hits[-1][1])
    eta_s = None
    _m_eta = re.findall(r'Step:\s*\d+/\d+\s*\[[\d:]+<([\d:]+)', tail)
    if _m_eta:
        eta_s = _to_sec_any(_m_eta[-1])

    def _to_sec(hms: str):
        try:
            parts = [int(x) for x in hms.split(':')]
        except ValueError:
            return None
        while len(parts) < 3:
            parts.insert(0, 0)
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])

    # 🔴 稳态必须用**每步增量**（相邻两行 elapsed 之差）。为什么不能用累计均值：
    #    unit 边界那一步含评测/扫描（实测评测 45s + hardness 466s），把**累计均值**抬高，
    #    之后每步分母只 +1 ⇒ 要 ~50 步才回落（实测出现 258/184/132… 的虚高序列）
    #    ⇒ 直接取"最近 20 个累计均值的中位数"会把稳态高估（真机 20.3 vs 真实 10.2–10.9）。
    deltas = []
    prev_step = prev_sec = None
    for stp, hms, _cum in hits:
        cur = _to_sec(hms)
        if None not in (prev_sec, cur) and int(stp) - int(prev_step) == 1:
            d = cur - prev_sec
            if 0.0 < d < 3600.0:
                deltas.append(d)
        prev_step, prev_sec = int(stp), cur
    if not deltas:
        return int(hits[-1][0]), None, len(hits), latest, elapsed_s, eta_s
    # 边界步**自动识别**：明显大于典型步（> max(3×中位, 25s)）的那些
    med_all = statistics.median(deltas)
    cutoff = max(3.0 * med_all, 25.0)
    steady = [d for d in deltas if d <= cutoff] or deltas
    return (int(hits[-1][0]), statistics.median(steady[-60:]), len(hits), latest,
            elapsed_s, eta_s)


# --------------------------------------------------------------------------- #
# 紧凑视图（默认输出；目标 ≤22 行，终端一屏可见）
# --------------------------------------------------------------------------- #
_CLR = {'TRAINING': '36', 'DEFER': '33', 'PASS': '32', 'CANDIDATE': '90', 'EXHAUSTED': '31'}
_SPARK = '▁▂▃▄▅▆▇█'
#: 池子分段用的**字符**（不靠颜色也能分辨）
#: 池子分段字符 —— **只用 ASCII**（`+`/`=`/`-`/`.`）：
#    框线与方块字符在部分终端（CJK 宽度判定不准）会错位成乱码，真机踩过。
_POOL_CHAR = {'TRAINING': '+', 'DEFER': '=', 'PASS': '#', 'CANDIDATE': '-', 'EXHAUSTED': '!'}
_POOL_TAG = {'TRAINING': 'T', 'DEFER': 'D', 'PASS': 'P', 'CANDIDATE': 'C', 'EXHAUSTED': 'E'}



def _use_color() -> bool:
    if os.environ.get('NO_COLOR'):
        return False
    try:
        import sys
        return bool(sys.stdout.isatty())
    except Exception:  # noqa: BLE001
        return False


def _c(text: str, code: str) -> str:
    return f'\033[{code}m{text}\033[0m' if _use_color() else text


def _bar(frac: float, width: int = 18, fill: str = '█', empty: str = '░') -> str:
    f = max(0.0, min(1.0, float(frac)))
    n = int(round(f * width))
    return fill * n + empty * (width - n)


def _rstr(r, width: int = 6) -> str:
    """倍率着色：≤1 绿 / ≤1.3 黄 / 其余红（无颜色时纯文本）。"""
    if not isinstance(r, (int, float)):
        return ' ' * width
    txt = f'{r:.2f}x'
    col = '32' if r <= 1.0 else ('33' if r <= 1.3 else '31')
    return _c(txt.rjust(width), col)


def _spark(vals) -> str:
    vals = [float(v) for v in vals]
    if not vals:
        return ''
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return _SPARK[3] * len(vals)
    return ''.join(_SPARK[int((v - lo) / (hi - lo) * 7)] for v in vals)


def _progress_bit(att: dict) -> str:
    """进步一行摘要：`训练过15  lp50>0 7 / ≤0 8  过拟合 6  最好 X(+0.200)`。"""
    if not att:
        return _c('（还没有被训练过的任务）', '33')
    lp = [(r.get('lp50'), t) for t, r in att.items() if isinstance(r.get('lp50'), (int, float))]
    pos = [x for x in lp if x[0] > 0]
    of = [t for t, r in att.items() if r.get('overfit')]
    best = max(lp) if lp else (None, None)
    bt = f'  最好 {best[1]}({best[0]:+.3f})' if best[0] is not None else ''
    return (f'训练过{len(att)}  lp50>0 {_c(str(len(pos)), "32")} / ≤0 {_c(str(len(lp) - len(pos)), "33")}'
            f'  过拟合 {_c(str(len(of)), "31") if of else "0"}{bt}')


def _pbar(frac: float, width: int = 16) -> str:
    """纯 ASCII 比例条：`###....`（避免框线/方块字符在不支持 CJK-宽字符的终端里错位）。"""
    f = max(0.0, min(1.0, float(frac)))
    n = int(round(f * width))
    return '#' * n + '.' * (width - n)


def _pool_line(records: dict, derived_status: dict) -> str:
    """池子一行：**`pass=7  defer=12  candidate=31` 形式**（用户 2026-10-12：不要条形）。

    两处用户口径：
      * 用**可读键名**（`D12` 这种单字母标签要对照才能读懂）；
      * **`exhausted` 固定显示**（即使是 0）—— 它是"任务被判死"的信号，必须一眼可见；
        其余状态为 0 时省略以省版面。
    """
    _NAME = {'TRAINING': 'training', 'DEFER': 'defer', 'PASS': 'pass',
             'CANDIDATE': 'candidate', 'EXHAUSTED': 'exhausted'}
    if records:
        cnt = pool_counts(records)
        order = [k for k in _STATUS_ORDER if cnt.get(k)] + [k for k in cnt if k not in _STATUS_ORDER]
        if 'EXHAUSTED' not in order:
            order.append('EXHAUSTED')
        return '  '.join(_c(f'{_NAME.get(k, k.lower())}={cnt.get(k, 0)}', _CLR.get(k, '0'))
                         for k in order)
    st = Counter(derived_status.values()) if derived_status else {}
    return ('  '.join(f'{k.lower()}={v}' for k, v in sorted(st.items()))
            + _c('  (事件流推导，无 defer/exhausted 细分)', '33'))


def _trace_lines(task: str, series: dict, rec: dict, n: int = 4) -> list:
    """在训任务的**历史轨迹**：每 unit 一行 `step / GMean / 门槛 / 倍率 / loss / 相对上次`。"""
    val = (series.get('active_val_gmean_mse') or {}).get(task) or []
    thr = dict((series.get('pass_threshold_gmean_mse') or {}).get(task) or [])
    rat = dict((series.get('val_to_pass_threshold') or {}).get(task) or [])
    los = dict((series.get('unit_loss') or {}).get(task) or [])
    if not val:
        return [f'  {task}: 尚无实测记录（等第一个 unit 末的判定）']
    out = []
    prev = None
    rows = val[-n:]
    for st, g in rows:
        r = rat.get(st) or ((g / thr[st]) if thr.get(st) else None)
        d = ''
        if prev is not None and r is not None:
            diff = r - prev
            d = ('↓改善 %.3f' % abs(diff)) if diff < -0.005 else (
                ('↑变差 +%.3f' % diff) if diff > 0.005 else '→持平')
        prev = r if r is not None else prev
        th_txt = f'{thr[st]:.4f}' if isinstance(thr.get(st), (int, float)) else '  --  '
        lo_txt = f'{los[st]:.4f}' if isinstance(los.get(st), (int, float)) else '  --  '
        out.append(f'  {st:>5}  G {g:.4f}  门槛 {th_txt}  倍率 {_rstr(r)}  loss {lo_txt}  {d}')
    if len(val) < n:
        out.append(f'  （仅 {len(val)} 次记录；更早的没有）')
    return out


def _term_width(default: int = 64, cap: int = 100) -> int:
    """当前终端可用列数。

    🔴 真机踩过：写死 74 列时，窄终端会把 `GPU0,2,3,...` 这类长行**折行**，
    看起来就是"缩进/换行全乱"。所以先探测（`COLUMNS` → `os.get_terminal_size`
    → `shutil`），再取 min(探测值, cap)；非 TTY（重定向）用 default。
    """
    w = None
    try:
        env = int(os.environ.get('COLUMNS') or 0)
        if env > 0:
            w = env
    except ValueError:
        w = None
    if w is None and _use_color():          # 只在真 TTY 上探测（重定向时探测值无意义）
        try:
            w = os.get_terminal_size().columns
        except OSError:
            try:
                import shutil
                w = shutil.get_terminal_size((default, 24)).columns
            except Exception:  # noqa: BLE001
                w = None
    if not w or w <= 20:
        w = default
    return max(40, min(int(w), cap))


def _fit(text: str, width: int) -> str:
    """截断到 width（含省略号）。"""
    return text if len(text) <= width else text[:max(1, width - 1)] + '…'


def print_compact(*, run: str, target: int, passed: int, newly, boot, step, sit, latest_sit,
                  elapsed_s, eta_s, trainer, trainers, records, thresholds, derived_status,
                  per_task, series, dcp_name, dcp_step, tb=None,
                  wide_hint: bool = True, cfg=None, current_task=None) -> None:
    """紧凑视图 v5：**每行一个信息**、缩进统一两格、宽度自适应（窄终端不折行）。

    结构（约 15 行）：
        AL <run>                     ← 标题行
          通关    <bar>  7/50 (14%)  ← 每个指标一行，标签 4 字对齐
          进度    Step 2152/20000
          步时    11.0s/it（稳态）
          用时    已训 11.3h / 剩余 189.6h / 总计 200.9h
          池子    ====####----  D12 P7 C31
          训练    15 个（lp50>0 7 / ≤0 8，过拟合 6）
        -- 在训 stack_blocks_two
          状态    CANDIDATE   已训 0 步
          倍率    当前 1.64x   best 1.64x
          历史    step 2150  G 0.0093  门槛 0.0062  1.50x  ↓改善
    """
    LBL = 6                                  # 标签列宽（`通关` + 两空格）
    W = _term_width()

    def row(label: str, text: str) -> None:
        print('  ' + _fit(f'{label:<{LBL}}{text}', W))

    # ---- 标题 ----
    cfg = cfg or {}
    pass_enabled = cfg.get('pass_enabled', True)
    total_steps = cfg.get('max_steps')
    goal = f'目标 PASS {target}' if pass_enabled else f'固定步数 {total_steps or "未知"}（PASS关闭）'
    print('AL ' + _fit(f'{run}   {goal}', W - 3))
    # ---- 通关 ----
    pct = (passed / target) if target else 0.0
    extra = ''
    if isinstance(newly, (int, float)):
        extra += f'  新增{int(newly)}'
    if isinstance(boot, (int, float)) and boot:
        extra += f'  免费{int(boot)}'
    # 用户要求：通关**不要条形**（`##....`），只要数字与百分比
    row('通关', f'{passed}/{target} ({pct * 100:.0f}%){extra}' if pass_enabled else '关闭状态转换；开环指标仅观察')
    # ---- 进度 + 步时（用户要求合并成一行）----
    p_bits = []
    if step is not None:
        p_bits.append(f'Step {step}' + (f'/{total_steps}' if total_steps else ''))
    if trainer:
        p_bits.append(f'{int(trainer["ranks"])}卡')
        if trainer.get('gpus'):
            p_bits.append(f'GPU{trainer["gpus"]}')
    else:
        p_bits.append(_c('无进程!', '31'))
    if sit:
        p_bits.append(f'{sit:.1f}s/it')
        if latest_sit is not None and sit and latest_sit > sit * 1.8:
            p_bits.append(_c(f'当前步 {latest_sit:.0f}s(含评测/扫描)', '33'))
    row('进度', '  '.join(p_bits))
    # ---- 用时 ----
    t_bits = []
    if elapsed_s:
        t_bits.append(f'已训 {_hms_h(elapsed_s)}')
    # ⚠️ 日志里的 `<ETA` 字段**不可靠**：它与累计均值同源，会被边界步（评测+扫描）抬高，
    #    实测同一时刻连续读到 61:27 → 63:52 → 69:45（在涨）。所以**用稳态步速自己算**：
    #    剩余 = (总步数 - 当前步) × 稳态 s/it —— 单调、可解释、不被单次边界步带偏。
    eta_est = None
    if sit and step is not None and total_steps:
        eta_est = max(0, total_steps - int(step)) * float(sit)
    if eta_est:
        t_bits.append(f'剩余约 {_hms_h(eta_est)}')
    if elapsed_s and eta_est:
        t_bits.append(f'总计约 {_hms_h(elapsed_s + eta_est)}')
    if t_bits:
        row('用时', ' / '.join(t_bits) + ('（纯训练估计，另加评测/存档）' if (sit and step) else ''))
    switches = [f'{label}={"开" if cfg[key] else "关"}' for key, label in
                (('task_scan', '任务扫描'), ('hardness_scan', '样本扫描')) if cfg.get(key) is not None]
    if cfg.get('interval'):
        switches.append(f'评测每{cfg["interval"]}步')
    if cfg.get('save_steps'):
        switches.append(f'存档每{cfg["save_steps"]}步')
    if switches:
        row('配置', '  '.join(switches))
    # ---- 池子（**实时优先**：TensorBoard 的计数 vs 注册表快照）----
    # 🔴 2026-10-12（用户要求"面板全部实时"）：池子计数改从 **TB** 读。
    #   为什么：注册表只在存档点导出（`SAVE_EVERY`=400 步）⇒ 真机 v65 出现
    #   "通关 14（实时）与池子 pass=8（2400 快照）同屏矛盾"。
    #   TB 每 unit 末（≈50 步）实时写 `skill_overview/{pass,defer,exhausted,candidate}_count`。
    _tb_counts = {}
    for _key, _tag in (('PASS', 'skill_overview/pass_count'),
                       ('DEFER', 'skill_overview/defer_count'),
                       ('EXHAUSTED', 'skill_overview/exhausted_count'),
                       ('CANDIDATE', 'skill_overview/candidate_count')):
        _v = (tb or {}).get(_tag)
        if _v is not None and isinstance(_v[1], (int, float)):
            _tb_counts[_key] = int(_v[1])
    _tb_step = (tb or {}).get('curriculum/units_completed')
    if _tb_counts:
        _order = ('PASS', 'DEFER', 'CANDIDATE', 'EXHAUSTED')
        _body = '  '.join(f'{k.lower()}={_tb_counts[k]}'
                          for k in _order if k in _tb_counts)
        row('池子', f'{_body}   {_c("(实时·TB)", "32")}')
    else:
        row('池子', _pool_line(records, derived_status))
    if records and not cfg.get('train_all_episodes', False):
        att = {t: r for t, r in records.items()
               if isinstance((r or {}).get('attempt_count'), (int, float))
               and int((r or {}).get('attempt_count')) >= 1}
        lp = [(r.get('lp50'), t) for t, r in att.items() if isinstance(r.get('lp50'), (int, float))]
        pos = [x for x in lp if x[0] > 0]
        of = [t for t, r in att.items() if r.get('overfit')]
        best = max(lp) if lp else (None, None)
        row('训练', f'{len(att)} 个 (lp50>0 {len(pos)} / ≤0 {len(lp) - len(pos)}, '
                    f'过拟合 {len(of)}'
                    + (f', 最好 {best[1]}({best[0]:+.3f})' if best[0] is not None else '') + ')')
    # ================= 在训任务（重点） =================
    # 🔴 2026-10-12 修：**实时优先，快照只作兜底与详情**。
    #   为什么（真机 v65 实测）：`pool_state.json` 只在**存档点**导出（每 SAVE_EVERY 步），
    #   于是注册表里的 `status=TRAINING` 会长时间停在**上一个任务**（实测：主循环已在训
    #   `stack_bowls_two`，面板却仍显示 `click_bell`）⇒ 用户看到"在训任务不对"。
    #   而事件流的 `task/<任务>/unit_loss` 是**每 unit 末实时写**的 ⇒ 用它判"当前在训"。
    _loss_all = series.get('unit_loss') or {}
    _cand = [(seq[-1][0], str(t)) for t, seq in _loss_all.items() if seq]
    cur = current_task or next((t for t, s in reversed(list(derived_status.items())) if s == 'TRAINING'), '')
    cur = cur or (max(_cand)[1] if _cand else '')
    _live = bool(cur)
    if not cur:                                    # 事件流还没 unit ⇒ 只能用快照
        if records:
            for t, r in records.items():
                if str((r or {}).get('status')) == 'TRAINING':
                    cur = str(t)
                    break
    # 快照里 TRAINING 的那个任务（用于判断"cur 是否与快照一致"）
    _reg_training = ''
    if records:
        for t, r in records.items():
            if str((r or {}).get('status')) == 'TRAINING':
                _reg_training = str(t)
                break
    print('-' * W)
    if not cur:
        row('在训', '（预扫/初始化阶段，还没有在训任务）')
        print('=' * W)
        return
    rec = (records or {}).get(cur) or {}
    st = str(rec.get('status') or '?')
    # 🔴 任务名以**实时**为准 ⇒ 状态/倍率也随之标注来源，避免"1 分钟前缓存的中间值"被当成当前值
    _stale = bool(_live and _reg_training and _reg_training != cur)
    if _stale:
        st = f'实时(事件流)；注册表快照仍停在 {_reg_training}'
    elif not st or st == '?':
        st = '事件流（无注册表快照）' if _live else '?'
    if cfg.get('train_all_episodes', False):
        row('在训', f'{cur}   [{st}]')
        if cfg.get('auxiliary_manifest'):
            ratio = float(cfg.get('auxiliary_primary_ratio', .7))
            text = f'配额 主{ratio:.0%}/辅{1-ratio:.0%}；辅助按episode均衡'
            primary_seen = (tb or {}).get('sampling/primary_samples_per_unit')
            auxiliary_seen = (tb or {}).get('sampling/auxiliary_samples_per_unit')
            if primary_seen and auxiliary_seen and primary_seen[0] == auxiliary_seen[0]:
                text += f'；最近unit实际 {int(primary_seen[1])}/{int(auxiliary_seen[1])}'
            row('辅助池', text)
        print_train_monitor_history(cur, series)
        print('=' * W)
        return
    r_now = _ratio(dict(rec, task_name=cur), thresholds) if rec else None
    r_best = _ratio(dict(rec, task_name=cur), thresholds, key='best_gmean_mse') if rec else None
    row('在训', f'{cur}   [{st}]')
    # 🔴 若当前任务与注册表快照里的 TRAINING 任务**不是同一个** ⇒ 快照的
    #   「本任务步数 / 倍率 / 进步」全是**别的任务的旧值** ⇒ 直接不显示（免得误读）。
    #   （2026-10-12：v65 实测出现"倍率 当前 10.30x"其实是快照里 click_bell 的值，
    #    而当前在训 place_object_basket ⇒ 用户会以为新任务倍率很差。）
    if _stale:
        row('本任务', _c('（快照未更新，暂无该任务的重建详情；倍率/进步见下方历史）', '90'))
        _show_detail = False
    else:
        _show_detail = True
    # 「已训多少步」属于**进度**，不要混进「倍率」行（真机反馈：标签配错会误读）
    if _show_detail and isinstance(rec.get('attempt_step'), (int, float)):
        _lag = (int(step) - int(dcp_step)) if (step is not None and dcp_step is not None) else 0
        _t = f'{int(rec["attempt_step"])} 步'
        if _lag > 0:
            _t += _c(f'（注册表快照落后 {_lag} 步）', '90')
        row('本任务', _t)
    m_bits = []
    if _show_detail and r_now is not None:
        m_bits.append(f'当前 {_rstr(r_now)}')
    if _show_detail and r_best is not None:
        m_bits.append(f'best {_rstr(r_best)}')
    if m_bits:
        row('倍率', '  '.join(m_bits))
    ee_text = ee_val_text(series, cur)
    if ee_text is not None:
        row('末端val', ee_text)
        ee_sides = []
        for side, label in (('left', '左'), ('right', '右')):
            values = (series.get(f'active_val_ee_{side}_position_rmse_mm') or {}).get(cur) or []
            if values:
                ee_sides.append(f'{label} {values[-1][1]:.2f} mm (step {values[-1][0]})')
        if ee_sides:
            row('左右定位', '  '.join(ee_sides))
    lp50, lptr, gap = rec.get('lp50'), rec.get('lp_train'), rec.get('train_val_gap_ratio')
    f_bits = []
    if isinstance(lp50, (int, float)):
        f_bits.append(f'lp50 {lp50:+.4f}')
    if isinstance(lptr, (int, float)):
        f_bits.append(f'lp_train {lptr:+.4f}')
    if isinstance(gap, (int, float)):
        f_bits.append(f'gap {gap:.2f}')
    if f_bits:
        row('进步', '  '.join(f_bits) + ('   过拟合' if rec.get('overfit') else '   未过拟合'))
    # ---- 历史轨迹（每 unit 一行，字段对齐）----
    val = (series.get('active_val_gmean_mse') or {}).get(cur) or []
    thr = dict((series.get('pass_threshold_gmean_mse') or {}).get(cur) or [])
    rat = dict((series.get('val_to_pass_threshold') or {}).get(cur) or {})
    los = dict((series.get('unit_loss') or {}).get(cur) or {})
    if not val:
        row('历史', '尚无实测记录（等第一个 unit 末的判定）')
    else:
        # 用户口径：固定看**最近 3 个 unit**
        threshold_text, changing_threshold = history_threshold_text([step for step, _ in val[-3:]], thr)
        row('历史', f'最近 {min(len(val), 3)} 个 unit（倍率 ≤1.0 过线；{threshold_text}）')
        if series.get('_training_ee'):
            row('说明', 'EE=位置/姿态/夹爪，未乘权重，取该训练步')
        prev = None
        for stp, g in val[-3:]:
            r = rat.get(stp) or ((g / thr[stp]) if thr.get(stp) else None)
            d = ''
            if prev is not None and r is not None:
                diff = r - prev
                d = ('↓%.3f' % abs(diff)) if diff < -0.005 else (
                    ('↑+%.3f' % diff) if diff > 0.005 else '→')
            prev = r if r is not None else prev
            lo_txt = f'{los[stp]:.4f}' if isinstance(los.get(stp), (int, float)) else '--'
            r_txt = f'{r:.2f}x' if isinstance(r, (int, float)) else '--'
            ee_loss = ee_training_loss_text(series, stp) if stp in los else None
            threshold = (f' T {thr[stp]:.4g}' if isinstance(thr.get(stp), (int, float)) else ' T --') if changing_threshold else ''
            # Keep the three EE values on one logical line, without truncation.
            print(f'    {stp:>5} G {g:.4f}{threshold} {r_txt:>6} loss {lo_txt}'
                  f'{"  " + ee_loss if ee_loss else ""}  {d}'.rstrip())
    print('=' * W)
    if trainers and len(trainers) > 1:
        print(_c(f'  机上另有 {len(trainers) - 1} 个训练器在跑（--trainers 看全部）', '33'))
    if wide_hint:
        # 用户要求：参数提示用**逗号**分隔（`|` 在窄终端容易被读成折行符）
        print(_c(_fit('  参数: --pool 明细, --progress 进步, --traces N 轨迹, '
                      '--wide 表格, --watch S', W), '90'))


def main() -> int:
    ap = argparse.ArgumentParser(description='AL 训练仪表盘')
    ap.add_argument('run', nargs='?', default=None, help='run 名（默认自动选最新）')
    ap.add_argument('--all', action='store_true', help='列出全部任务（默认只列最接近通关的若干）')
    ap.add_argument('--trainers', action='store_true',
                    help='只列出机上所有训练器（多 run 并存时用；每个一行）')
    ap.add_argument('--top', type=int, default=4,
                    help='默认显示多少个任务（按倍数升序；默认 4 以便整屏可见；--all 看全部）')
    ap.add_argument('--wide', action='store_true',
                    help='完整表格视图（任务明细 + 成长轨迹；默认是紧凑视图）')
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
    run_dir = resolve_run_dir(run)
    if not run or run_dir is None or not run_dir.is_dir():
        # 🔴 三种情况必须分开说（真机踩坑：新 run 在预扫阶段**目录尚未创建**，
        #    旧提示语一律说"未找到 run"，让人误以为 run 名写错）：
        #    ① 进程在跑但目录没建 ⇒ 仍在启动/预扫；
        #    ② 进程不在跑且目录没建 ⇒ 真的没有这个 run；
        #    ③ 目录在但事件流为空 ⇒ 刚起，还没走到第一个 unit。
        running = [t for t in discover_trainers() if t.get('run') == run]
        # 候选 run 名：两种布局都列（旧 `train_<run>`、新 `<run>`）
        existing = sorted({(q.name[len('train_'):] if q.name.startswith('train_') else q.name)
                           for q in ROOT.glob('train_*')
                           if q.is_dir()} |
                          {q.name for q in ROOT.glob('al_v*') if q.is_dir()})
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
    ev_path = run_dir / 'auto_learning_events.jsonl'
    last, per_task = read_events(ev_path)
    derived_status, derived_passed = derive_status(ev_path)
    series = read_series(ev_path)
    trainers = discover_trainers()
    trainer = find_trainer(trainers, run)
    ee_series = {}
    tb = read_tb_latest(run_dir, ee_series=ee_series)
    series['_training_ee'] = ee_series
    log_progress = step_progress(train_log_path(run), cfg['interval'])  # type: ignore[arg-type]
    step, sit, nsamples, latest_sit, elapsed_s, eta_s = prefer_tensorboard_progress(
        log_progress, tb)

    # 🔴 通关数口径（真机纠错两次后的结论）：
    #    * **registry 指标 `curriculum/passed_tasks` 是权威**（scheduler 自己算，含训练中新增 PASS）；
    #    * 事件推导只覆盖 bootstrap/select 的 pass，**会漏掉训练中新增的 PASS** ⇒ 只作交叉校验，
    #      且在它**更大**时以它为准（真机实测：step 200 时 registry=4、推导=3）。
    reg_passed = last.get('curriculum/passed_tasks')
    reg_passed = int(reg_passed) if isinstance(reg_passed, (int, float)) else None
    passed = reg_passed if reg_passed is not None else len(derived_passed)
    newly = last.get('curriculum/newly_passed_count')
    boot = last.get('curriculum/bootstrap_pass_count')
    # ---- 池子（**权威 = DCP 注册表**；读不到就退回事件流并提示）----
    records, dcp_step, dcp_dir = load_registry(run_dir)
    # ---- TensorBoard（**实时**；进度优先读标量 step，池子计数避免存档延迟）----
    if not records:
        _st = Counter(derived_status.values()) if derived_status else {}
        if _st:
            print(' 池子（事件流推导）: ' + ' | '.join(f'{k} {v}' for k, v in sorted(_st.items())))
            print('   ⚠️ 事件流仅作参考；完整池详情待 HF/pool_state 或 DCP 快照（不影响训练显示）')

    # ---- 输出：默认紧凑视图；`--wide` 走原完整表格 ----
    _common = dict(run=run, target=target, passed=passed, newly=newly, boot=boot,
                   step=step, sit=sit, latest_sit=latest_sit,
                   elapsed_s=elapsed_s, eta_s=eta_s,
                   trainer=trainer, trainers=trainers, thresholds=thresholds,
                   derived_status=derived_status, per_task=per_task, series=series,
                   current_task=last.get('curriculum/current_task_name'),
                   records=records, dcp_name=(dcp_dir.name if dcp_dir else None),
                   dcp_step=dcp_step, tb=tb)
    if getattr(args, 'wide', False):
        return render_wide(args=args, cfg=cfg, **{k: v for k, v in _common.items()
                                                  if k not in ('dcp_name', 'dcp_step')},
                           dcp_name=(dcp_dir.name if dcp_dir else None), dcp_step=dcp_step,
                           derived_passed=derived_passed, reg_passed=reg_passed)
    print_compact(**_common, cfg=cfg)
    return 0



def render_wide(*, run: str, args, cfg, thresholds, target, trainer, trainers,
                passed, newly, boot, step, sit, latest_sit, per_task, series,
                derived_status, records, dcp_name, dcp_step,
                derived_passed, reg_passed, elapsed_s=None, eta_s=None, tb=None,
                current_task=None) -> int:
    """`--wide`：原来的**完整表格**视图（任务明细 + 成长轨迹 + 三次槽位 + 摘要）。

    2026-10-12 起默认输出改为紧凑视图（终端只放得下一屏）；本函数保留全部细节，
    供需要逐任务核对时使用。**代码是从原 main() 整段搬来的，逻辑一字未改。**
    """
    if cfg.get('train_all_episodes', False):
        # 全训练模式没有逐任务val/门槛表，复用专用监控视图防止显示虚假倍率。
        print_compact(run=run, target=target, passed=passed, newly=newly, boot=boot,
            step=step, sit=sit, latest_sit=latest_sit, elapsed_s=elapsed_s, eta_s=eta_s,
            trainer=trainer, trainers=trainers, records=records, thresholds=thresholds,
            derived_status=derived_status, per_task=per_task, series=series,
            dcp_name=dcp_name, dcp_step=dcp_step, tb=tb, cfg=cfg,
            current_task=current_task, wide_hint=False)
        return 0
    W = 76
    print('=' * 78)
    goal = f'目标 PASS={target}' if cfg.get('pass_enabled', True) else f'固定步数={cfg.get("max_steps") or "未知"}（PASS关闭）'
    print(f' AL 仪表盘  run={run}   {goal}')
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
    if cfg.get('pass_enabled', True):
        print(f' 通关: {passed}/{target}（{pct:.0f}%）还差 {max(0, target - passed)}{_extra}'
              + ('' if reg_passed is not None else '（本 run 尚未产生判定）'))
    else:
        print(' 通关: 关闭状态转换；开环指标仅观察')
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
    named = current_task
    ee_text = ee_val_text(series, named or cur_task)
    if ee_text:
        print(f' 末端val: {named or cur_task}  {ee_text}')
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
    threshold_text, changing_threshold = history_threshold_text([step for step, _ in val_seq], th_map)
    print(f' 成长轨迹: {cur_task}（该任务自身历史，每 unit 一点；倍率 ≤1.0 过线；{threshold_text}）')
    if series.get('_training_ee'):
        print(' 说明: EE=位置/姿态/夹爪，未乘权重，取该训练步')
    if not val_seq:
        print('   没有（该任务还没有 active_val 记录）')
    else:
        print(f'   {"step":>5}{"GMean":>9}{"倍率":>7}{"loss":>9}  EE(位置/姿态/夹爪)  变化')
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
            ee_loss = ee_training_loss_text(series, st) if loss is not None else None
            threshold = (f' T {th:.4g}' if isinstance(th, (int, float)) else ' T --') if changing_threshold else ''
            print(f'   {st:>5}{g:>9.4f}{threshold}'
                  f'{(f"{ratio:.2f}x" if ratio is not None else "—"):>7}'
                  f'{(f"{loss:.4f}" if loss is not None else "—"):>9}'
                  f'{"  " + ee_loss if ee_loss else ""}  {delta}')
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
