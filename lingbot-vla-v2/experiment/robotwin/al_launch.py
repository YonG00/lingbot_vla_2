#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""al_launch.py —— Auto Learning 统一启动程序（缓存检查 + 并行扫描 + 启动训练）。

一句话
------
「有缓存就直接训练；缓存不全就只并行补齐缺失的任务；显式要求就全量并行重扫」，
扫完（且覆盖确实完整）才启动训练；覆盖不全时以非零退出码结束并点名缺失任务。

为什么需要它（今天踩过的坑）
----------------------------
* **进程内 bootstrap 扫描很慢**：7 卡 FSDP2 同步扫 50 个任务约 18 分钟，任一 rank
  掉队会拖死全组。**并行扫描**（N 个独立单卡进程，各持完整模型，任务按
  ``tasks[i::N]`` 切片）约 5 分钟，进程间没有集合通信，天然失败隔离。
* 少一次校验就会出事，因此本工具把这些检查**做进流程**（今天的真实事故）：
  ① 分片并集必须等于任务全集（无重复、无遗漏）；
  ② worker 启动 30 秒后核对「进程数」与「已就绪：N 任务」，不符立刻报错退出；
  ③ 运行中按分片打印进度，**进度以缓存条目数为准**（事件文件可能有残留数据）；
  ④ 收尾只统计目标指纹目录，缺失清单非空 ⇒ 非零退出码 + 打印缺失任务名（绝不
     打印假的「50/50」）；
  ⑤ 清场只按**精确 PID**（`/proc/<pid>/cmdline` 核对身份），绝不 `pkill -f`。

实现约束
--------
* **纯标准库**（`argparse`/`hashlib`/`ast`/`json`/`subprocess`/`socket`…），
  便于在无 GPU 的机器上做语法检查与 `--dry-run` 预演。
* 真正要跑训练/扫描的机器上，用训练环境解释器启动本脚本
  （默认 `/opt/robotwin-env/bin/python`），这样指纹里的 ``.py`` 语义 hash
  与训练进程用的是同一个 Python 版本（`ast.dump` 的输出与版本相关）。
  若两者版本不一致，本工具会自动改用 `--python` 指到的解释器做一次
  **子进程指纹计算**，并在报告里注明。

退出码
------
===  ==================================================
0    成功（dry-run 预演完成，或训练已后台启动）
2    前置检查失败（参数/路径/AL 配置/机器被占用）
3    扫描未完成（目标指纹目录里仍有缺失任务）
4    worker 未就绪或提前退出（进程数 / 「已就绪：N 任务」不符）
5    扫描超时
6    内部错误（未预期异常）
===  ==================================================

用法
----
    /opt/robotwin-env/bin/python experiment/robotwin/al_launch.py --dry-run
    /opt/robotwin-env/bin/python experiment/robotwin/al_launch.py
    /opt/robotwin-env/bin/python experiment/robotwin/al_launch.py --no-cache
    /opt/robotwin-env/bin/python experiment/robotwin/al_launch.py --selfcheck

完整说明见 ``docs/auto_learning_launch_guide_zh.md``。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

#: 可用卡（4 号卡已挂死：利用率恒 100%、温度 ~30 度、任何 H2D 拷贝都挂住）
DEFAULT_GPUS = '0,1,2,3,5,6,7'

#: scout 缓存记录 schema —— 必须与 `lingbotvla/auto_learning/scout_cache.py` 的 VERSION 一致
VERSION = 1
MAX_JSON_BYTES = 1_000_000
NO_SEMANTIC_HASH_MARKER = 'scout-fingerprint: no-semantic-hash'

#: 训练启动脚本（环境变量驱动）
DEFAULT_LAUNCH_SCRIPT = 'experiment/robotwin/al_50task_bf16.sh'
DEFAULT_TRAIN_CONFIG = 'configs/rocm/robotwin_official_paths_rocm.yaml'
DEFAULT_EVAL_CONFIG = 'configs/auto_learning/al_eval2.yaml'
DEFAULT_PYTHON = '/opt/robotwin-env/bin/python'

DEFAULT_CACHE_ROOT = '/workspace/al/scout_cache'
DEFAULT_CHECKPOINT = '/workspace/models/robbyant_lingbot-vla-v2-6b-bf16'
DEFAULT_SPLIT_DIR = '/workspace/al/task_splits_50'
DEFAULT_PHASES = '/workspace/al/phases_al'
DEFAULT_LEROBOT_ROOT = '/workspace/lerobot'
DEFAULT_NORM_REL = 'assets/norm_stats/robotwin_competition_clean.json'
DEFAULT_QWEN3VL = '/workspace/models/Qwen3-VL-4B-Instruct-config-tokenizer'
#: TMPDIR **绝不能**是 /tmp（那里只有 4 GB tmpfs）
DEFAULT_TMPDIR = '/models/robotwin-persistent/tmp/al_launch'
DEFAULT_TRITON_CACHE = '/workspace/runtime/triton'
DEFAULT_TORCHINDUCTOR_CACHE = '/workspace/runtime/torchinductor'
DEFAULT_WORKER_OUT_ROOT = '/workspace/al/al_launch_runs'

#: 评测链 source 清单 —— **必须与 scout_cache.EVAL_SOURCES 逐字同步**。
#: (key, 相对仓库根的路径, required)。指纹口径：宁可多列，绝不少列。
EVAL_SOURCES: Tuple[Tuple[str, str, bool], ...] = (
    ('eval', 'lingbotvla/utils/open_loop_validation.py', True),
    ('model', 'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py', True),
    ('transform', 'lingbotvla/data/vla_data/transform.py', True),
    ('dataset_builder', 'lingbotvla/data/dataset.py', True),
    ('multi_vla_dataset', 'lingbotvla/data/vla_data/multi_vla_dataset.py', True),
    ('base_dataset', 'lingbotvla/data/vla_data/base_dataset.py', True),
    ('data_utils', 'lingbotvla/data/vla_data/utils.py', True),
    ('ee_pose_transform', 'lingbotvla/data/vla_data/ee_pose_transform.py', False),
    ('video_utils', 'lingbotvla/data/vla_data/video_utils.py', False),
    ('eval_precision', 'lingbotvla/utils/eval_precision.py', True),
    ('evaluator', 'lingbotvla/auto_learning/evaluator.py', True),
    ('gmean', 'lingbotvla/auto_learning/decision/gmean.py', True),
    ('scan_accel', 'lingbotvla/auto_learning/scan_accel.py', True),
    ('eval_batch_policy', 'lingbotvla/auto_learning/eval_batch_policy.py', True),
)

DTYPE_ALIASES = {'bf16': 'bfloat16', 'fp16': 'float16', 'half': 'float16',
                 'fp32': 'float32', 'float': 'float32', 'double': 'float64'}

RE_READY = re.compile(r'已就绪：\s*(\d+)\s*任务')
RE_FATAL = re.compile(
    r'^\s*(RuntimeError|ValueError|TypeError|KeyError|AssertionError|ImportError|'
    r'FileNotFoundError|OSError|DistNetworkError)\b', re.M)
FATAL_SUBSTRINGS = (
    'Traceback (most recent call last)',
    'CUDA out of memory',
    'HIP out of memory',
    'HSA_STATUS_ERROR',
    'hipErrorNoBinaryForGpu',
    'EADDRINUSE',
    'invalid provenance fingerprint',
    'Scout cache checkpoint must exactly equal',
    'Scout cache dtype mismatch',
)

EXIT_OK = 0
EXIT_PREFLIGHT = 2
EXIT_SCAN_INCOMPLETE = 3
EXIT_WORKER_NOT_READY = 4
EXIT_SCAN_TIMEOUT = 5
EXIT_INTERNAL = 6


class LaunchError(RuntimeError):
    """前置检查失败（带明确人话 + 退出码）。"""

    def __init__(self, message: str, code: int = EXIT_PREFLIGHT) -> None:
        super().__init__(message)
        self.code = int(code)


# --------------------------------------------------------------------------- #
# 日志（同时写文件与 stdout）
# --------------------------------------------------------------------------- #
class Tee:
    """所有日志同时写文件与 stdout（`--json` 时 stdout 让给 JSON，人话走 stderr）。"""

    def __init__(self, path: Optional[Path] = None, *, to_stream: bool = True,
                 json_mode: bool = False) -> None:
        self.json_mode = bool(json_mode)
        self._to_stream = bool(to_stream)
        self._fh = None
        self.file_error: Optional[str] = None
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(path, 'a', encoding='utf-8')
            except OSError as exc:
                # 不因为「日志目录建不出来」而崩溃：退回只写 stdout / stderr
                self.file_error = f'{path}: {exc}'

    def write(self, message: str = '') -> None:
        line = str(message)
        if self._fh is not None:
            self._fh.write(line + '\n')
            self._fh.flush()
        if self._to_stream:
            if self.json_mode:
                print(line, file=sys.stderr, flush=True)
            else:
                print(line, flush=True)

    __call__ = write

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _stamp() -> str:
    return time.strftime('%Y%m%d-%H%M%S', time.gmtime())


def _human(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f'{seconds:.0f}s'
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 90:
        return f'{minutes}m{sec:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h{minutes:02d}m'


# --------------------------------------------------------------------------- #
# 指纹：**逐行精确复刻** scout_cache.py（sha256_file / semantic_sha256 /
# source_sha256 / normalize_dtype / provenance / source_manifest / scout_key）
# --------------------------------------------------------------------------- #
def sha256_file(path: Any) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _is_docstring_stmt(node: ast.stmt) -> bool:
    return (isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


class _DocstringStripper(ast.NodeTransformer):
    """删除模块/类/函数 docstring（只删每个 body 的第一条 Expr(Constant(str))）。"""

    @staticmethod
    def _strip_body(body: List[ast.stmt]) -> List[ast.stmt]:
        if body and _is_docstring_stmt(body[0]):
            return body[1:]
        return body

    def visit_Module(self, node: ast.Module):  # noqa: N802 (ast API)
        self.generic_visit(node)
        node.body = self._strip_body(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef):  # noqa: N802
        self.generic_visit(node)
        node.body = self._strip_body(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):  # noqa: N802
        self.generic_visit(node)
        node.body = self._strip_body(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):  # noqa: N802
        self.generic_visit(node)
        node.body = self._strip_body(node.body)
        return node


def semantic_sha256(path: Any) -> str:
    """`.py` 语义 hash（注释/空行/docstring 不生效；逻辑改动必生效）。"""
    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        return hashlib.sha256(raw).hexdigest()
    if NO_SEMANTIC_HASH_MARKER in text:
        return hashlib.sha256(raw).hexdigest()
    try:
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, ValueError):
        return hashlib.sha256(raw).hexdigest()
    tree = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    canonical = ast.dump(tree, include_attributes=False, annotate_fields=True)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def source_sha256(path: Any) -> str:
    path = Path(path)
    if path.suffix == '.py':
        return semantic_sha256(path)
    return sha256_file(path)


def normalize_dtype(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith('torch.'):
        text = text[len('torch.'):]
    return DTYPE_ALIASES.get(text, text)


def provenance(*, weight_files: Sequence[Any], sources: Dict[str, Any],
               options: Dict[str, Any]) -> str:
    """与 `scout_cache.provenance` 逐字一致：权重分片内容 + 源码 + 选项 的联合 SHA256。"""
    if not weight_files or not sources or 'inference_dtype' not in options:
        raise ValueError('Scout cache requires checkpoint shards, eval sources and dtype')
    opts = dict(options)
    opts['inference_dtype'] = normalize_dtype(opts['inference_dtype'])
    manifest = {
        'schema': VERSION,
        'weights': [(str(Path(p).name), sha256_file(p)) for p in sorted(weight_files, key=str)],
        'sources': {k: source_sha256(v) for k, v in sorted(sources.items())},
        'options': opts,
    }
    if len({n for n, _ in manifest['weights']}) != len(manifest['weights']):
        raise ValueError('ambiguous checkpoint shard names')
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def source_manifest(repo_root: Path,
                    extra_sources: Optional[Dict[str, Any]] = None
                    ) -> Tuple[Dict[str, Path], List[Tuple[str, str]]]:
    """`EVAL_SOURCES` + 额外数据/配置文件 ⇒ (存在的清单, 缺失清单)。"""
    sources: Dict[str, Path] = {}
    missing: List[Tuple[str, str]] = []
    for key, rel, required in EVAL_SOURCES:
        path = Path(repo_root) / rel
        if path.is_file():
            sources[key] = path
        else:
            missing.append((key, f'{rel} 不存在（required={required}）'))
    for key, value in dict(extra_sources or {}).items():
        if value is None:
            continue
        path = Path(value).expanduser()
        if path.is_file():
            sources[key] = path
        else:
            missing.append((key, f'{path} 不存在'))
    return sources, missing


def scout_key(task: str, episode_ids: Sequence[int]) -> str:
    ids = list(map(int, episode_ids))
    if not task or not ids or len(set(ids)) != len(ids):
        raise ValueError('invalid task/episode ID set')
    return hashlib.sha256(json.dumps([task, ids], separators=(',', ':')).encode()).hexdigest()


def compute_fingerprint(*, repo_root: Path, checkpoint: Path, manifest: Path, norm: Path,
                        baseline: Path, thresholds: Path, dtype: str, scout_trajs: int,
                        image_augment: bool) -> Tuple[str, Dict[str, Any]]:
    """复刻 `real/build.py` 的指纹构造（含 `src_extra` 六个数据/配置文件）。"""
    shards = sorted(checkpoint.glob('*.safetensors'))
    if not shards:
        raise LaunchError(f'checkpoint 目录下没有 *.safetensors：{checkpoint}'
                          '（--checkpoint 指到 HF 权重目录，不是训练输出目录）')
    extra: Dict[str, Any] = {
        'manifest': manifest,
        'norm': norm,
        'thresholds': thresholds,
        'baseline': baseline,
        'checkpoint_config': checkpoint / 'config.json',
        'checkpoint_tokenizer': checkpoint / 'tokenizer.json',
    }
    sources, missing = source_manifest(repo_root, extra)
    options = {
        'inference_dtype': normalize_dtype(dtype),
        'noise_seed': 1234,
        'scout_trajs': int(scout_trajs),
        'stride': 'per_episode',
        'image_augment': bool(image_augment),
    }
    fp = provenance(weight_files=shards, sources=sources, options=options)
    return fp, {
        'sources_included': len(sources),
        'sources_missing': [{'key': k, 'reason': r} for k, r in missing],
        'weight_shards': [p.name for p in shards],
        'weight_bytes': sum(p.stat().st_size for p in shards),
        'options': options,
    }


# --------------------------------------------------------------------------- #
# 缓存覆盖检查（复刻 BootstrapScoutCache.load 的校验 ⇒ 「真会命中」才算覆盖）
# --------------------------------------------------------------------------- #
def entry_is_hit(path: Path, *, fingerprint: str, task: str, ids: Sequence[int]
                 ) -> Tuple[bool, str]:
    """该 JSON 会不会被 `BootstrapScoutCache.load()` 当成命中返回。"""
    want = list(map(int, ids))
    if not path.is_file():
        return False, '文件不存在'
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            return False, '超过 MAX_JSON_BYTES'
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False, 'JSON 解析失败'
    if raw.get('version') != VERSION:
        return False, f"version={raw.get('version')!r} != {VERSION}"
    if raw.get('fingerprint') != fingerprint:
        return False, 'fingerprint 不匹配'
    if raw.get('task') != task:
        return False, f"task={raw.get('task')!r} 不匹配"
    if raw.get('episode_ids') != want:
        return False, f"episode_ids={raw.get('episode_ids')!r} 不匹配"
    m = raw.get('metrics')
    if not isinstance(m, dict):
        return False, 'metrics 不是字典'
    if m.get('task') != task or m.get('episode_ids') != want:
        return False, 'metrics.task/episode_ids 不匹配'
    per = m.get('per_traj_mse')
    if not isinstance(per, dict) or set(per) != set(map(str, want)):
        return False, 'per_traj_mse 键集合不匹配'
    try:
        vals = [float(v) for v in per.values()]
    except (TypeError, ValueError):
        return False, 'per_traj_mse 不是数字'
    if any(not math.isfinite(v) or v < 0 for v in vals):
        return False, 'per_traj_mse 含非法值'
    try:
        nmse, mse, base = (float(m[k]) for k in ('nmse', 'mse', 'baseline_mse'))
    except (KeyError, TypeError, ValueError):
        return False, 'metrics 缺 nmse/mse/baseline_mse'
    if (m.get('n_trajs') != len(want) or m.get('metric_valid') is not True
            or not all(math.isfinite(v) for v in (nmse, mse, base))
            or min(nmse, mse) < 0 or base <= 0
            or not math.isclose(nmse, mse / base, rel_tol=1e-5, abs_tol=1e-8)
            or not math.isclose(mse, sum(vals) / len(vals), rel_tol=1e-4, abs_tol=1e-7)):
        return False, 'metrics 自洽性校验未通过'
    gmean = m.get('gmean_mse')
    if gmean is None or not math.isfinite(float(gmean)):
        return False, 'gmean_mse 缺失/非有限（GMean 模式不算命中）'
    return True, 'ok'


class Coverage:
    """某个指纹目录对一组任务的覆盖情况。"""

    def __init__(self, cache_dir: Path, fingerprint: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.fingerprint = fingerprint
        self.rows: Dict[str, Tuple[bool, str]] = {}

    def add(self, task: str, ids: Sequence[int]) -> None:
        if not ids:
            self.rows[task] = (False, '任务没有可用的 val ids（无法确定 scout 回合）')
            return
        path = self.cache_dir / f'{scout_key(task, list(ids))}.json'
        self.rows[task] = entry_is_hit(path, fingerprint=self.fingerprint, task=task, ids=ids)

    @property
    def covered(self) -> List[str]:
        return [t for t, (ok, _) in self.rows.items() if ok]

    @property
    def missing(self) -> List[str]:
        return [t for t, (ok, _) in self.rows.items() if not ok]

    @property
    def reasons(self) -> Dict[str, str]:
        return {t: r for t, (ok, r) in self.rows.items() if not ok}

    def json_files(self) -> int:
        try:
            return sum(1 for p in self.cache_dir.glob('*.json') if p.is_file())
        except OSError:
            return 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            'cache_dir': str(self.cache_dir),
            'covered': len(self.covered),
            'total': len(self.rows),
            'missing': list(self.missing),
            'missing_reasons': self.reasons,
            'entries_in_dir': self.json_files(),
        }

    def summary(self) -> str:
        return f'{len(self.covered)}/{len(self.rows)}'


def check_coverage(cache_dir: Path, fingerprint: str,
                   task_ids: Dict[str, List[int]]) -> Coverage:
    cov = Coverage(cache_dir, fingerprint)
    for task, ids in task_ids.items():
        cov.add(task, ids)
    return cov


# --------------------------------------------------------------------------- #
# 任务全集与 val ids
# --------------------------------------------------------------------------- #
def load_task_ids(*, manifest: Path, split_dir: Path, tasks_override: Optional[str],
                  task_source: str, lerobot_root: Path
                  ) -> Tuple[List[str], Dict[str, List[int]], List[str]]:
    """返回 (任务名列表, 任务→val_ids, 警告列表)。"""
    warnings: List[str] = []
    val_ids: Dict[str, List[int]] = {}
    names: List[str] = []

    manifest_tasks: Dict[str, Any] = {}
    if manifest.is_file():
        try:
            man = json.loads(manifest.read_text(encoding='utf-8'))
            raw = man.get('tasks')
            if isinstance(raw, dict) and raw:
                manifest_tasks = raw
            else:
                warnings.append(f'manifest 里没有 tasks 字典：{manifest}')
        except (OSError, ValueError) as exc:
            warnings.append(f'manifest 解析失败（{exc}）：{manifest}')
    else:
        warnings.append(f'manifest 不存在：{manifest}')

    for name, rec in manifest_tasks.items():
        ids = rec.get('val_ids') if isinstance(rec, dict) else None
        if ids is None:
            side = Path(split_dir) / f'{name}.val_ids.json'
            if side.is_file():
                try:
                    ids = json.loads(side.read_text(encoding='utf-8'))
                except (OSError, ValueError) as exc:
                    warnings.append(f'{side} 解析失败（{exc}）')
        if isinstance(ids, list):
            val_ids[str(name)] = [int(x) for x in ids]

    if task_source == 'lerobot':
        names = sorted(p.name[: -len('_joint_v30')] for p in Path(lerobot_root).glob('*_joint_v30')
                       if p.is_dir())
        if not names:
            warnings.append(f'--task-source lerobot 但 {lerobot_root} 下没有 *_joint_v30 目录')
    else:
        names = list(manifest_tasks.keys())
        if not names and val_ids:
            names = list(val_ids.keys())

    if tasks_override:
        names = [t.strip() for t in str(tasks_override).split(',') if t.strip()]

    unknown = [t for t in names if t not in val_ids]
    for t in unknown:
        warnings.append(f'任务 {t} 在 manifest 里没有 val ids ⇒ 无法确定 scout 回合（会被判缺失）')
        val_ids.setdefault(t, [])
    return names, val_ids, warnings


# --------------------------------------------------------------------------- #
# AL 配置读取（YAML 扁平标量块 / JSON，两行以内不引入 PyYAML）
# --------------------------------------------------------------------------- #
def _strip_yaml_comment(line: str) -> str:
    out: List[str] = []
    quote: Optional[str] = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == '\\' and quote == '"' and i + 1 < len(line):
                out.append(ch)
                out.append(line[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            out.append(ch)
            i += 1
            continue
        if ch in ('"', "'"):
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == '#':
            break
        out.append(ch)
        i += 1
    return ''.join(out)


def _parse_scalar(text: str) -> Any:
    t = text.strip()
    if t == '':
        return None
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ('"', "'"):
        return t[1:-1]
    low = t.lower()
    if low in ('null', '~', 'none'):
        return None
    if low == 'true':
        return True
    if low == 'false':
        return False
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    return t


def read_al_config(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    """读 AL 配置的「扁平标量」正文（支持顶层 + `auto_learning:` 包装 + `.json`）。"""
    warnings: List[str] = []
    text = Path(path).read_text(encoding='utf-8')
    if Path(path).suffix == '.json':
        raw = json.loads(text)
        body = raw.get('auto_learning', raw) if isinstance(raw, dict) else {}
        return dict(body), warnings

    lines = text.splitlines()
    wrapper_at: Optional[int] = None
    for idx, raw_line in enumerate(lines):
        stripped = _strip_yaml_comment(raw_line)
        if not stripped.strip():
            continue
        if re.match(r'^auto_learning\s*:\s*$', stripped):
            wrapper_at = idx
        break

    if wrapper_at is None:
        base_indent = 0
        body_lines = list(lines)
    else:
        body_lines = []
        base_indent = -1
        for raw_line in lines[wrapper_at + 1:]:
            stripped = _strip_yaml_comment(raw_line)
            if not stripped.strip():
                continue
            indent = len(stripped) - len(stripped.lstrip(' '))
            if base_indent < 0:
                base_indent = indent
            if indent < base_indent:
                break
            body_lines.append(raw_line)

    body: Dict[str, Any] = {}
    nested: List[str] = []
    current_nested: Optional[str] = None
    for raw_line in body_lines:
        stripped = _strip_yaml_comment(raw_line)
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(' '))
        if indent != base_indent:
            if current_nested and current_nested not in nested:
                nested.append(current_nested)
            continue
        text_line = stripped.strip()
        if ':' not in text_line:
            warnings.append(f'无法解析的行（已跳过）：{text_line[:60]}')
            current_nested = None
            continue
        key, _, value = text_line.partition(':')
        key = key.strip()
        if value.strip() == '' and key != 'auto_learning':
            current_nested = key
            continue
        current_nested = None
        body[key] = _parse_scalar(value)
    if nested:
        warnings.append('AL 配置里的嵌套块不会被复制到分片配置（本工具只搬运扁平标量）：'
                        + ', '.join(nested))
    if not body:
        raise LaunchError(f'无法从 {path} 解析出 AL 配置正文（既不是 JSON，也不是扁平 YAML 标量块）')
    return body, warnings


def write_shard_config(path: Path, *, index: int, total: int, body: Dict[str, Any],
                       tasks: Sequence[str], source_config: Path) -> None:
    """生成单片 worker 的 AL 配置（JSON 也是合法 YAML，`yaml.safe_load` 能读）。"""
    payload = {
        'description': (f'al_launch 分片配置 {index + 1}/{total}：仅 task_names 不同，'
                        f'其余标量原样来自 {source_config}'),
        'auto_learning': {**body, 'task_names': list(tasks)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


# --------------------------------------------------------------------------- #
# 进程工具（只用精确 PID；Linux /proc）
# --------------------------------------------------------------------------- #
def _proc_dir(pid: int) -> Path:
    return Path('/proc') / str(pid)


def proc_alive(pid: int) -> bool:
    if sys.platform != 'linux':
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ValueError):
            return False
    return _proc_dir(pid).exists()


def proc_cmdline(pid: int) -> List[str]:
    """`/proc/<pid>/cmdline` 的 argv 列表（不存在 ⇒ 空列表）。"""
    path = _proc_dir(pid) / 'cmdline'
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    return [part.decode('utf-8', 'replace') for part in raw.split(b'\0') if part]


def proc_pgid(pid: int) -> Optional[int]:
    """进程组 id（`/proc/<pid>/stat` 第 5 个字段）。"""
    try:
        stat = (_proc_dir(pid) / 'stat').read_text(encoding='utf-8')
    except OSError:
        return None
    try:
        tail = stat[stat.rindex(')') + 1:].split()
        return int(tail[2])
    except (ValueError, IndexError):
        return None


def proc_identity(pid: int) -> str:
    argv = proc_cmdline(pid)
    return ' '.join(argv)[:200] if argv else '<no cmdline>'.replace('<no cmdline>', '')


def _identity_matches(pid: int, expect_tokens: Sequence[str]) -> Tuple[bool, str]:
    """按 `/proc/<pid>/cmdline` 核对身份：argv[0] 必须是解释器/shell，且 argv 里出现目标脚本。"""
    argv = proc_cmdline(pid)
    if not argv:
        return False, 'cmdline 读不到（进程可能已退出）'
    head = Path(argv[0]).name
    joined = ' '.join(argv)
    ok_head = (head in ('bash', 'sh', 'dash', 'setsid', 'nohup')
               or head.startswith('python'))
    if not ok_head:
        return False, f'argv[0]={head!r} 不是 shell/解释器'
    for token in expect_tokens:
        if token and token in joined:
            return True, f'argv[0]={head}；命中 {token}'
    return False, f'cmdline 里没有 {list(expect_tokens)}：{joined[:160]}'


def terminate_own(pid: int, *, expect_tokens: Sequence[str], role: str,
                  log: Tee, grace: float = 20.0) -> bool:
    """精确终止自己启动的会话（先核对身份，再 SIGTERM 进程组，最后 SIGKILL）。

    绝不按名字/正则批量杀 —— 今天正是 `pkill -ABRT -f train_lingbotvla` 误杀了
    刚启动的扫描 worker，导致 50 个任务只写入 44 条。
    """
    if not proc_alive(pid):
        return False
    ok, why = _identity_matches(pid, expect_tokens)
    if not ok:
        log.write(f'[cleanup] 拒绝发信号：pid={pid}（{role}）身份核对未通过 —— {why}')
        return False
    pgid = proc_pgid(pid)
    if pgid != pid:
        log.write(f'[cleanup] pid={pid}（{role}）不是会话首进程（pgid={pgid}）'
                  '⇒ 只对单进程发 SIGTERM，不碰进程组')
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return False
    else:
        log.write(f'[cleanup] 精确终止 pid={pid} / 会话 {pgid}（{role}；身份核对通过：{why}）')
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            return False
    deadline = time.time() + grace
    while time.time() < deadline:
        if not proc_alive(pid):
            log.write(f'[cleanup] pid={pid} 已退出（SIGTERM）')
            return True
        time.sleep(1.0)
    if proc_alive(pid) and pgid == pid:
        log.write(f'[cleanup] pid={pid} 在 {grace:.0f}s 内没有退出 ⇒ SIGKILL 同一会话')
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
    return True


def list_other_runs(extra_names: Sequence[str]) -> List[Dict[str, Any]]:
    """只读扫描 `/proc`：列出在跑的 `train_lingbotvla.py` / 本工具进程（抢卡预警）。"""
    if sys.platform != 'linux':
        return []
    tokens = ['train_lingbotvla.py', *extra_names]
    found: List[Dict[str, Any]] = []
    me = os.getpid()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        argv = proc_cmdline(pid)
        if not argv:
            continue
        joined = ' '.join(argv)
        for token in tokens:
            if token in joined:
                found.append({'pid': pid, 'kind': token, 'cmdline': joined[:200]})
                break
    return found


def free_port(preferred: int) -> int:
    """优先用 preferred；被占用就随机取一个空闲端口。"""
    for candidate in (int(preferred), 0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', candidate))
            port = sock.getsockname()[1]
            sock.close()
            return int(port)
        except OSError:
            sock.close()
            continue
    return int(preferred)


def spawn_background(*, script: Path, env: Dict[str, str], log_path: Path, cwd: Path,
                     shell_env: Optional[Dict[str, str]] = None,
                     log: Optional['Tee'] = None) -> int:
    """`setsid nohup env K=V … bash <script> > 日志 2>&1 < /dev/null &` 并取回真实 PID。

    让 `/bin/sh` 起后台作业并 `echo $!` —— `setsid` 只有在「自己已是进程组组长」时才
    fork，本场景不会 fork，所以 `$!` 就是最终的 `bash <script>`（会话首进程），
    后续清场可以按这个精确 PID（以及它自己的会话）办事。

    机器上没有 `setsid`/`nohup` 时（例如在 macOS 上做本地彩排）退回
    `Popen(..., start_new_session=True)`：脱离子会话的语义等价，PID 同样可用。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    base_env = dict(shell_env or {})
    pairs = ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in env.items())
    have_setsid = shutil.which('setsid', path=base_env.get('PATH')) is not None
    if have_setsid:
        inner = (f'setsid nohup env {pairs} bash {shlex.quote(str(script))} '
                 f'> {shlex.quote(str(log_path))} 2>&1 < /dev/null & echo $!')
        proc = subprocess.run(['/bin/sh', '-c', inner], cwd=str(cwd), text=True,
                              capture_output=True, timeout=120, env=shell_env)
        if proc.returncode != 0:
            raise LaunchError(f'后台启动失败（rc={proc.returncode}）：{proc.stderr.strip()[:400]}')
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines or not lines[-1].isdigit():
            raise LaunchError(f'后台启动没有取到 PID（stdout={proc.stdout!r}）')
        return int(lines[-1])
    if log is not None:
        log('[warn] 本机没有 setsid ⇒ 退回 Popen(start_new_session=True) 的等价脱离子会话方式')
    handle = open(log_path, 'ab')
    try:
        proc = subprocess.Popen(['bash', str(script)], env=env, cwd=str(cwd),
                                stdin=subprocess.DEVNULL, stdout=handle,
                                stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        handle.close()
    return int(proc.pid)


def shell_env_for_launch(python: Path) -> Dict[str, str]:
    path = f'{python.parent}:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin'
    return {'PATH': path, 'HOME': os.environ.get('HOME', '/root'), 'LANG': 'C.UTF-8'}


def equivalent_command(script: Path, env: Dict[str, str], log_path: Path, cwd: Path) -> str:
    pairs = ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in env.items())
    return (f'cd {shlex.quote(str(cwd))} && setsid nohup env {pairs} '
            f'bash {shlex.quote(str(script))} > {shlex.quote(str(log_path))} '
            '2>&1 < /dev/null &')


def read_tail(path: Path, max_bytes: int = 200_000) -> str:
    try:
        with open(path, 'rb') as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode('utf-8', 'replace')
    except OSError:
        return ''


def detect_fatal(text: str) -> Optional[str]:
    for needle in FATAL_SUBSTRINGS:
        if needle in text:
            return needle
    match = RE_FATAL.search(text)
    if match:
        for line in text.splitlines():
            if line.strip().startswith(match.group(1)):
                return line.strip()[:200]
    return None


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog='al_launch.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Auto Learning 统一启动程序：缓存检查 → 并行补齐扫描 → 启动训练。')
    mode = ap.add_argument_group('模式')
    mode.add_argument('--no-cache', action='store_true',
                      help='忽略现有缓存：全量并行扫描全部任务后再启动训练（旧指纹目录会被'
                           '改名备份，不删除）')
    mode.add_argument('--dry-run', action='store_true',
                      help='只打印步骤计划与将用到的环境变量，不启动任何进程（无 GPU 也能跑）')
    mode.add_argument('--json', action='store_true',
                      help='机器可读输出：stdout 只给 JSON，人话日志走日志文件与 stderr')
    mode.add_argument('--selfcheck', action='store_true',
                      help='自检：与仓库 scout_cache.py 逐项对拍指纹实现（不需要 GPU）')

    core = ap.add_argument_group('核心')
    core.add_argument('--gpus', default=DEFAULT_GPUS,
                      help=f'可用卡（逗号分隔），默认 {DEFAULT_GPUS}；'
                           '会同时显式导出 CUDA_VISIBLE_DEVICES 与 HIP_VISIBLE_DEVICES')
    core.add_argument('--eval-config', default=DEFAULT_EVAL_CONFIG,
                      help=f'AL 配置（相对仓库根），默认 {DEFAULT_EVAL_CONFIG}')
    core.add_argument('--steps', type=int, default=200, help='训练 MAX_STEPS，默认 200')
    core.add_argument('--micro', type=int, default=12, help='micro batch，默认 12')
    core.add_argument('--gas', type=int, default=1,
                      help='梯度累积，默认 1（脚本自算 GBS = MICRO*GAS*N_GPU）')
    core.add_argument('--cache-root', default=DEFAULT_CACHE_ROOT,
                      help=f'scout 缓存根目录，默认 {DEFAULT_CACHE_ROOT}')
    core.add_argument('--fingerprint', default=None,
                      help='显式指定 64 位指纹（跳过计算；必须是完整小写 hex）')

    paths = ap.add_argument_group('路径')
    paths.add_argument('--repo', default=None, help='仓库根（默认取本脚本的上一级目录）')
    paths.add_argument('--python', default=DEFAULT_PYTHON,
                       help=f'训练环境解释器，默认 {DEFAULT_PYTHON}')
    paths.add_argument('--launch-script', default=DEFAULT_LAUNCH_SCRIPT,
                       help=f'训练启动脚本（相对仓库根），默认 {DEFAULT_LAUNCH_SCRIPT}')
    paths.add_argument('--train-config', default=DEFAULT_TRAIN_CONFIG,
                       help=f'训练配置（CONFIG 变量），默认 {DEFAULT_TRAIN_CONFIG}')
    paths.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                       help=f'初始权重（HF 目录），默认 {DEFAULT_CHECKPOINT}')
    paths.add_argument('--manifest', default=None,
                       help=f'任务划分 manifest，默认 <split-dir>/manifest.json')
    paths.add_argument('--baseline', default=None,
                       help='task baseline，默认 <split-dir>/task_baseline.json')
    paths.add_argument('--norm', default=None,
                       help=f'norm stats，默认 <repo>/{DEFAULT_NORM_REL}')
    paths.add_argument('--split-dir', default=DEFAULT_SPLIT_DIR,
                       help=f'任务划分目录，默认 {DEFAULT_SPLIT_DIR}')
    paths.add_argument('--phases', default=DEFAULT_PHASES,
                       help=f'PHASES 目录（datasets.txt），默认 {DEFAULT_PHASES}')
    paths.add_argument('--thresholds', default=None,
                       help='通过线阈值表（默认取 AL 配置里的 pass_thresholds_file）')
    paths.add_argument('--tasks', default=None,
                       help='显式任务全集（逗号分隔）；默认取 manifest 的 tasks')
    paths.add_argument('--task-source', choices=('manifest', 'lerobot'), default='manifest',
                       help='任务全集来源，默认 manifest（lerobot 用 *_joint_v30 目录名）')
    paths.add_argument('--lerobot-root', default=DEFAULT_LEROBOT_ROOT,
                       help=f'lerobot 数据根（--task-source lerobot 时用），默认 {DEFAULT_LEROBOT_ROOT}')
    paths.add_argument('--qwen3vl', default=DEFAULT_QWEN3VL,
                       help=f'QWEN3VL 权重目录，默认 {DEFAULT_QWEN3VL}')
    paths.add_argument('--train-out', default=None,
                       help='训练输出目录（默认 <worker-out-root>/train_<配置名>_<时间戳>）')
    paths.add_argument('--run-name', default=None, help='本次 run 名（用于输出目录与日志名）')
    paths.add_argument('--worker-out-root', default=DEFAULT_WORKER_OUT_ROOT,
                       help=f'扫描 worker 的输出根目录，默认 {DEFAULT_WORKER_OUT_ROOT}')
    paths.add_argument('--shard-config-dir', default=None,
                       help='已有的分片 AL 配置目录（用 al_shard<i>.yaml，不再自动生成）')

    runtime = ap.add_argument_group('运行环境')
    runtime.add_argument('--tmpdir', default=DEFAULT_TMPDIR,
                         help=f'TMPDIR（绝不能是 /tmp），默认 {DEFAULT_TMPDIR}')
    runtime.add_argument('--triton-cache', default=DEFAULT_TRITON_CACHE,
                         help=f'TRITON_CACHE_DIR，默认 {DEFAULT_TRITON_CACHE}')
    runtime.add_argument('--torchinductor-cache', default=DEFAULT_TORCHINDUCTOR_CACHE,
                         help=f'TORCHINDUCTOR_CACHE_DIR，默认 {DEFAULT_TORCHINDUCTOR_CACHE}')
    runtime.add_argument('--dtype', default='bfloat16',
                         help='AL_SCOUT_CACHE_DTYPE / 推理精度，默认 bfloat16（必须与权重一致）')
    runtime.add_argument('--scout-trajs', type=int, default=None,
                         help='每任务 scout 回合数；默认取 AL 配置的 global_scout_val_trajs')
    runtime.add_argument('--image-augment', action='store_true',
                         help='image_augment=true（默认 false，必须与启动脚本一致，否则指纹不同）')
    runtime.add_argument('--tb-port', type=int, default=6006, help='TensorBoard 端口，默认 6006')
    runtime.add_argument('--no-tb', action='store_true', help='训练不开 TensorBoard')
    runtime.add_argument('--workers', type=int, default=None,
                         help='扫描 worker 数（默认 = 可用卡数，且不超过待扫任务数）')
    runtime.add_argument('--worker-max-steps', type=int, default=1,
                         help='worker 的 MAX_STEPS，默认 1')
    runtime.add_argument('--worker-checkpoint', action='store_true',
                         help='允许 worker 写 checkpoint（默认关：worker 用 SMOKE_NO_CHECKPOINT=1）')
    runtime.add_argument('--min-free-gb', type=float, default=20.0,
                         help='TMPDIR/输出目录可用空间下限（GB），默认 20；不足则拒绝启动')

    timing = ap.add_argument_group('时序与安全')
    timing.add_argument('--ready-wait', type=float, default=30.0,
                        help='首轮就绪检查的时间点（秒），默认 30')
    timing.add_argument('--ready-timeout', type=float, default=900.0,
                        help='等待「已就绪：N 任务」的上限（秒），默认 900')
    timing.add_argument('--poll-interval', type=float, default=20.0,
                        help='进度轮询间隔（秒），默认 20')
    timing.add_argument('--scan-timeout', type=float, default=1800.0,
                        help='整轮扫描上限（秒），默认 1800')
    timing.add_argument('--exit-grace', type=float, default=90.0,
                        help='覆盖已完整后仍等待 worker 自然退出的宽限（秒），默认 90')
    timing.add_argument('--retry-rounds', type=int, default=0,
                        help='扫描后仍有缺失时的补扫轮数，默认 0（直接报缺失并非零退出）')
    timing.add_argument('--allow-busy', action='store_true',
                        help='即使检测到别的训练/启动器在跑也继续（默认拒绝，避免抢卡）')
    timing.add_argument('--log-file', default=None,
                        help='本工具日志文件（默认 <worker-out-root>/logs/al_launch_<时间戳>.log）')

    hidden = ap.add_argument_group('内部')
    hidden.add_argument('--_print-fingerprint', dest='_print_fingerprint', action='store_true',
                        help=argparse.SUPPRESS)
    hidden.add_argument('--al-config', dest='al_config_internal', default=None,
                        help=argparse.SUPPRESS)
    return ap


# --------------------------------------------------------------------------- #
# 环境变量
# --------------------------------------------------------------------------- #
def base_env(*, args: argparse.Namespace, repo: Path, python: Path, al_cfg: Path,
             train_config: Path, split_dir: Path, phases: Path, checkpoint: Path,
             manifest: Path, baseline: Path, norm: Path, dtype: str) -> Dict[str, str]:
    """扫描 worker 与训练共用的环境变量（顺序即日志里的打印顺序）。"""
    return {
        'PY': str(python),
        'PATH': f'{python.parent}:{os.environ.get("PATH", "/usr/bin:/bin")}',
        'PYTHONPATH': str(repo),
        'N_GPU': '1',
        'MICRO': str(args.micro),
        'GAS': str(args.gas),
        'AL_CFG': str(al_cfg),
        'CONFIG': str(train_config),
        'SPLIT_DIR': str(split_dir),
        'PHASES': str(phases),
        'TRAIN_IDS': '',
        'MODEL_PATH': str(checkpoint),
        'QWEN3VL': str(args.qwen3vl),
        'TMPDIR': str(args.tmpdir),
        'TRITON_CACHE_DIR': str(args.triton_cache),
        'TORCHINDUCTOR_CACHE_DIR': str(args.torchinductor_cache),
        'AL_SCOUT_CACHE_MODE': 'bootstrap',
        'AL_SCOUT_CACHE_ROOT': str(args.cache_root),
        'AL_SCOUT_CACHE_CHECKPOINT': str(checkpoint),
        'AL_SCOUT_CACHE_MANIFEST': str(manifest),
        'AL_SCOUT_CACHE_BASELINE': str(baseline),
        'AL_SCOUT_CACHE_NORM': str(norm),
        'AL_SCOUT_CACHE_DTYPE': str(dtype),
        'AL_EVAL_PREFLIGHT': '1',
        'AL_EVAL_WARMUP': '1',
        'AL_EVAL_PHASE_LOG': '1',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'TOKENIZERS_PARALLELISM': 'false',
    }


def worker_env(common: Dict[str, str], *, gpu: str, train_out: Path, max_steps: int,
               fingerprint: str, master_port: int, smoke_no_checkpoint: bool) -> Dict[str, str]:
    env = dict(common)
    env.update({
        'CUDA_VISIBLE_DEVICES': str(gpu),
        'HIP_VISIBLE_DEVICES': str(gpu),
        'N_GPU': '1',
        'MAX_STEPS': str(int(max_steps)),
        'TRAIN_OUT': str(train_out),
        'MASTER_PORT': str(int(master_port)),
        'TB': '0',                     # worker 不起 TensorBoard（端口会互撞）
        'AL_SCOUT_CACHE_FINGERPRINT': fingerprint,
    })
    if smoke_no_checkpoint:
        env['SMOKE_NO_CHECKPOINT'] = '1'
        env['SAVE_HF'] = '0'
        env['PRUNE'] = '0'
    return env


def train_env(common: Dict[str, str], *, gpus: str, n_gpu: int, train_out: Path,
              max_steps: int, fingerprint: str, master_port: int, tb_port: int,
              tb: bool) -> Dict[str, str]:
    env = dict(common)
    env.update({
        'CUDA_VISIBLE_DEVICES': str(gpus),
        'HIP_VISIBLE_DEVICES': str(gpus),
        'N_GPU': str(int(n_gpu)),
        'MAX_STEPS': str(int(max_steps)),
        'TRAIN_OUT': str(train_out),
        'MASTER_PORT': str(int(master_port)),
        'TB': '1' if tb else '0',
        'TB_PORT': str(int(tb_port)),
        'AL_SCOUT_CACHE_FINGERPRINT': fingerprint,
    })
    return env


# --------------------------------------------------------------------------- #
# 计划与前置检查
# --------------------------------------------------------------------------- #
class Plan:
    """一次运行的完整计划（dry-run 直接打印它）。"""

    def __init__(self) -> None:
        self.steps: List[Dict[str, str]] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def step(self, title: str, detail: str) -> None:
        self.steps.append({'step': len(self.steps) + 1, 'title': title, 'detail': detail})


def resolve_repo_path(value: str, repo: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidate = repo / path
    if candidate.exists():
        return candidate
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return candidate


def shard_tasks(tasks: Sequence[str], n: int) -> List[List[str]]:
    """`tasks[i::n]` 切片（n 必须 >= 1）。"""
    if n < 1:
        raise ValueError('分片数必须 >= 1')
    return [list(tasks[i::n]) for i in range(n)]


def check_union(tasks: Sequence[str], shards: Sequence[Sequence[str]]) -> None:
    """分片并集必须恰好等于任务全集（无重复、无遗漏）。"""
    flat = [t for shard in shards for t in shard]
    duplicates = sorted({t for t in flat if flat.count(t) > 1})
    missing = [t for t in tasks if t not in set(flat)]
    extra = [t for t in flat if t not in set(tasks)]
    if duplicates or missing or extra:
        raise LaunchError(
            '分片并集校验未通过 ⇒ 拒绝启动：\n'
            f'  重复: {duplicates[:10]}\n  遗漏: {missing[:10]}\n  多余: {extra[:10]}')


def check_disk(path: Path, min_free_gb: float) -> Tuple[Optional[float], str]:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(str(probe))
    except OSError as exc:
        return None, f'无法读取 {probe} 的磁盘信息（{exc}）'
    free_gb = usage.free / (1024 ** 3)
    return free_gb, str(probe)


def validate_gpus(value: str) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    parts = [p.strip() for p in str(value).split(',') if p.strip() != '']
    if not parts:
        raise LaunchError('--gpus 不能为空')
    seen: List[str] = []
    for part in parts:
        if not part.isdigit():
            raise LaunchError(f'--gpus 里有非法项 {part!r}（应为逗号分隔的非负整数）')
        if part not in seen:
            seen.append(part)
    if '4' in seen:
        warnings.append('--gpus 包含 4 号卡：该卡今天实测挂死（利用率恒 100%、温度 ~30 度、'
                        '任何 H2D 拷贝都挂住）；除非确认已换机器，否则请改用 '
                        f'{DEFAULT_GPUS}')
    return seen, warnings


def interpreter_version(python: Path) -> Optional[Tuple[int, int]]:
    try:
        proc = subprocess.run([str(python), '-c', 'import sys;print("%d.%d" % sys.version_info[:2])'],
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        major, minor = proc.stdout.strip().split('.')[:2]
        return int(major), int(minor)
    except (ValueError, IndexError):
        return None


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _print_fingerprint_mode(args: argparse.Namespace) -> int:
    """内部模式：用指定解释器算一次指纹并把 JSON 打到 stdout。"""
    repo = Path(args.repo or Path(__file__).resolve().parents[2]).resolve()
    try:
        fp, info = compute_fingerprint(
            repo_root=repo,
            checkpoint=Path(args.checkpoint).expanduser(),
            manifest=Path(args.manifest).expanduser(),
            norm=Path(args.norm).expanduser(),
            baseline=Path(args.baseline).expanduser(),
            thresholds=Path(args.thresholds).expanduser(),
            dtype=args.dtype, scout_trajs=int(args.scout_trajs or 2),
            image_augment=bool(args.image_augment))
    except (LaunchError, ValueError, OSError) as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({'ok': True, 'fingerprint': fp, 'info': info, 'python': sys.executable},
                     ensure_ascii=False))
    return 0


def _selfcheck(args: argparse.Namespace, log: Tee, report: Dict[str, Any]) -> int:
    """与仓库 `scout_cache.py` 对拍：semantic_sha256 / normalize_dtype / scout_key / provenance。"""
    import importlib.util
    import tempfile

    repo = Path(args.repo or Path(__file__).resolve().parents[2]).resolve()
    ref_path = repo / 'lingbotvla' / 'auto_learning' / 'scout_cache.py'
    if not ref_path.is_file():
        log(f'[selfcheck] 找不到参考实现：{ref_path}')
        return EXIT_PREFLIGHT
    spec = importlib.util.spec_from_file_location('scout_cache_ref', ref_path)
    if spec is None or spec.loader is None:
        log('[selfcheck] 无法加载参考实现')
        return EXIT_PREFLIGHT
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    failures: List[str] = []
    checks = 0

    # ① 语义 hash：逐文件对拍 EVAL_SOURCES 里真实存在的 .py
    for key, rel, _ in EVAL_SOURCES:
        path = repo / rel
        if not path.is_file():
            continue
        checks += 1
        mine, theirs = semantic_sha256(path), ref.semantic_sha256(path)
        if mine != theirs:
            failures.append(f'semantic_sha256({rel}) 不一致：{mine[:12]} != {theirs[:12]}')
        checks += 1
        if normalize_dtype('bf16') != ref.normalize_dtype('bf16'):
            failures.append('normalize_dtype 不一致')

    # ② 记录 slug
    for task, ids in (('click_bell', [3, 7]), ('adjust_bottle', [0]), ('x_y', list(range(10)))):
        checks += 1
        if scout_key(task, ids) != ref.scout_key(task, ids):
            failures.append(f'scout_key({task}, {ids}) 不一致')

    # ③ 端到端指纹：伪造 checkpoint + 数据/配置文件，比较两次 provenance 输出
    with tempfile.TemporaryDirectory(prefix='al_launch_selfcheck_') as tmp:
        tmpdir = Path(tmp)
        ckpt = tmpdir / 'ckpt'
        ckpt.mkdir()
        (ckpt / 'model-00001-of-00002.safetensors').write_bytes(b'al_launch-selfcheck-1')
        (ckpt / 'model-00002-of-00002.safetensors').write_bytes(b'al_launch-selfcheck-2')
        (ckpt / 'config.json').write_text('{"selfcheck": true}', encoding='utf-8')
        (ckpt / 'tokenizer.json').write_text('{"selfcheck": true}', encoding='utf-8')
        extra: Dict[str, Path] = {}
        for name in ('manifest', 'norm', 'baseline', 'thresholds'):
            path = tmpdir / f'{name}.json'
            path.write_text(json.dumps({'name': name}), encoding='utf-8')
            extra[name] = path
        extra['checkpoint_config'] = ckpt / 'config.json'
        extra['checkpoint_tokenizer'] = ckpt / 'tokenizer.json'

        options = {'inference_dtype': 'bfloat16', 'noise_seed': 1234, 'scout_trajs': 2,
                   'stride': 'per_episode', 'image_augment': False}
        mine_src, mine_missing = source_manifest(repo, extra)
        theirs_src, theirs_missing = ref.source_manifest(repo, dict(extra), strict=False)
        checks += 1
        if mine_src != theirs_src:
            failures.append('source_manifest 清单不一致：'
                            f'{sorted(mine_src)} != {sorted(theirs_src)}')
        checks += 1
        if sorted(mine_missing) != sorted(theirs_missing):
            failures.append(f'source_manifest 缺失清单不一致：{mine_missing} != {theirs_missing}')
        shards = sorted(ckpt.glob('*.safetensors'))
        checks += 1
        mine_fp = provenance(weight_files=shards, sources=mine_src, options=options)
        theirs_fp = ref.provenance(weight_files=shards, sources=theirs_src, options=options)
        if mine_fp != theirs_fp:
            failures.append(f'端到端指纹不一致：{mine_fp} != {theirs_fp}')

    ok = not failures
    log('[selfcheck] 参考实现：' + str(ref_path))
    log(f'[selfcheck] 对拍项：{checks}；结果：' + ('全部一致' if ok else f'{len(failures)} 项不一致'))
    for item in failures:
        log('  - ' + item)
    report['selfcheck'] = {'ok': ok, 'checks': checks, 'failures': failures,
                           'reference': str(ref_path)}
    return EXIT_OK if ok else EXIT_PREFLIGHT


def run(args: argparse.Namespace, log: Tee, report: Dict[str, Any]) -> int:
    repo = Path(args.repo or Path(__file__).resolve().parents[2]).resolve()
    dry = bool(args.dry_run)
    plan = Plan()
    report['repo'] = str(repo)
    report['dry_run'] = dry
    report['mode'] = None

    log('=' * 78)
    log('al_launch —— Auto Learning 统一启动程序')
    log('=' * 78)
    log(f'时间(UTC)   : {_now_utc()}')
    log(f'仓库根      : {repo}')
    log(f'模式        : {"预演（dry-run，不启动任何进程）" if dry else "实跑"}'
        f'{"；忽略现有缓存（--no-cache）" if args.no_cache else ""}')
    log(f'本脚本解释器: {sys.executable}（Python {sys.version_info.major}.{sys.version_info.minor}）')

    # ---------------- 步骤 1：静态校验 ----------------
    gpus, gpu_warnings = validate_gpus(args.gpus)
    for warning in gpu_warnings:
        plan.warnings.append(warning)
        log('[warn] ' + warning)
    if args.steps < 1:
        raise LaunchError(f'--steps 必须 >= 1（收到 {args.steps}）')
    if args.micro < 1 or args.gas < 1:
        raise LaunchError('--micro / --gas 必须 >= 1')
    dtype = normalize_dtype(args.dtype)
    if dtype not in ('bfloat16', 'float16', 'float32'):
        raise LaunchError(f'--dtype 只支持 bfloat16/float16/float32，收到 {args.dtype!r}')
    if args.fingerprint is not None:
        fp_given = args.fingerprint.strip().lower()
        if len(fp_given) != 64 or any(c not in '0123456789abcdef' for c in fp_given):
            raise LaunchError('--fingerprint 必须是 64 位小写 hex（传前缀会被 '
                              'BootstrapScoutCache 拒绝：invalid provenance fingerprint）')
        args.fingerprint = fp_given
    plan.step('静态校验',
              f'卡 {",".join(gpus)}（{len(gpus)} 张）；steps={args.steps}；'
              f'micro={args.micro} gas={args.gas} ⇒ GBS={args.micro * args.gas * len(gpus)}')

    # ---------------- 步骤 2：路径与配置 ----------------
    python = Path(args.python).expanduser()
    launch_script = resolve_repo_path(args.launch_script, repo)
    al_cfg = resolve_repo_path(args.eval_config, repo)
    train_config = resolve_repo_path(args.train_config, repo)
    split_dir = Path(args.split_dir).expanduser()
    manifest = Path(args.manifest).expanduser() if args.manifest else split_dir / 'manifest.json'
    baseline = Path(args.baseline).expanduser() if args.baseline else split_dir / 'task_baseline.json'
    norm = Path(args.norm).expanduser() if args.norm else repo / DEFAULT_NORM_REL
    checkpoint = Path(args.checkpoint).expanduser()
    phases = Path(args.phases).expanduser()
    cache_root = Path(args.cache_root).expanduser()

    missing_paths: List[str] = []
    for label, path, required in (
            ('仓库根', repo, True),
            ('训练启动脚本', launch_script, True),
            ('训练环境解释器', python, True),
            ('训练配置 CONFIG', train_config, True),
            ('AL 配置', al_cfg, True),
            ('任务划分 manifest', manifest, True),
            ('task baseline', baseline, True),
            ('norm stats', norm, True),
            ('初始权重 checkpoint', checkpoint, True),
            ('PHASES 目录', phases, True),
            ('QWEN3VL 目录', Path(args.qwen3vl), True),
    ):
        if not path.exists():
            missing_paths.append(f'{label} 不存在：{path}')
    if missing_paths:
        for item in missing_paths:
            log('[missing] ' + item)
        if dry:
            plan.warnings.extend(missing_paths)
            log('[dry-run] 以上路径缺失 ⇒ 预演继续；真正启动前必须先补齐。')
        else:
            raise LaunchError('前置路径检查未通过：\n  - ' + '\n  - '.join(missing_paths))
    plan.step('路径检查',
              f'共 {11 - len(missing_paths)}/11 项存在'
              + (f'；缺失 {len(missing_paths)} 项（dry-run 继续）' if missing_paths else ''))

    # ---------------- 步骤 3：AL 配置（指纹选项 / 分片复制 / fail-fast 项）----------------
    al_body: Dict[str, Any] = {}
    thresholds = Path(args.thresholds).expanduser() if args.thresholds else None
    scout_trajs = args.scout_trajs
    if al_cfg.is_file():
        al_body, al_warnings = read_al_config(al_cfg)
        for warning in al_warnings:
            plan.warnings.append(warning)
            log('[warn] ' + warning)
        enabled = bool(al_body.get('enabled', True))
        if not enabled:
            raise LaunchError(f'AL 配置 enabled=false ⇒ 没有 Auto Learning 可跑：{al_cfg}')
        pass_metric = str(al_body.get('pass_metric', 'nmse'))
        if pass_metric != 'gmean_mse':
            raise LaunchError(
                f'AL 配置 pass_metric={pass_metric!r}，但 scout 缓存只支持 gmean_mse：\n'
                '  build.py 会在启用缓存时直接 RuntimeError（Scout cache only supports GMean）\n'
                f'  请改 {al_cfg} 或换一份配置')
        if al_body.get('pass_thresholds_file') is None:
            raise LaunchError(
                f'AL 配置缺 pass_thresholds_file：启用 scout 缓存时 build.py 会执行 '
                f'Path(None) 抛 TypeError。请补上该字段：{al_cfg}')
        if thresholds is None:
            thresholds = Path(str(al_body['pass_thresholds_file'])).expanduser()
            if not thresholds.is_absolute():
                thresholds = repo / thresholds
        if scout_trajs is None:
            scout_trajs = int(al_body.get('global_scout_val_trajs', 2))
    else:
        if not dry:
            raise LaunchError(f'AL 配置不存在：{al_cfg}')
        available = sorted(p.name for p in (repo / 'configs' / 'auto_learning').glob('*.yaml'))
        log(f'[dry-run] AL 配置不存在：{al_cfg}')
        log('[dry-run] configs/auto_learning 下现成的配置：' + (', '.join(available) or '（无）'))
        plan.warnings.append(f'AL 配置不存在：{al_cfg}')
    scout_trajs = int(scout_trajs or 2)
    if thresholds is not None and not thresholds.is_file():
        if dry:
            plan.warnings.append(f'阈值表不存在：{thresholds}')
            log(f'[dry-run] 阈值表不存在：{thresholds}（真正启动前必须补齐）')
        else:
            raise LaunchError(f'阈值表不存在：{thresholds}（AL 配置的 pass_thresholds_file）')
    plan.step('AL 配置',
              f'{al_cfg}；enabled={al_body.get("enabled", True)}；'
              f'pass_metric={al_body.get("pass_metric", "（缺失）")}；'
              f'scout_trajs={scout_trajs}；阈值表={thresholds or "（无）"}')

    # ---------------- 步骤 4：任务全集 + val ids ----------------
    tasks, task_ids, task_warnings = load_task_ids(
        manifest=manifest, split_dir=split_dir, tasks_override=args.tasks,
        task_source=args.task_source, lerobot_root=Path(args.lerobot_root))
    for warning in task_warnings:
        plan.warnings.append(warning)
        log('[warn] ' + warning)
    if not tasks:
        if dry:
            plan.warnings.append('任务全集为空（manifest 缺失或没解析出任务）')
            log('[dry-run] 任务全集为空 ⇒ 预演继续，但真正启动前必须有任务')
        else:
            raise LaunchError(f'任务全集为空：manifest={manifest}（或 --tasks 未给）')
    scout_ids: Dict[str, List[int]] = {t: list(task_ids.get(t, []))[:scout_trajs] for t in tasks}
    short = [(t, len(task_ids.get(t, []))) for t in tasks if len(task_ids.get(t, [])) < scout_trajs]
    if short:
        message = ('以下任务的 val 回合数少于 scout_trajs=' + str(scout_trajs) + '：'
                   + ', '.join(f'{t}({n})' for t, n in short[:10])
                   + ' ⇒ AL registry 会 fail-fast（_require）')
        if dry:
            plan.warnings.append(message)
            log('[warn] ' + message)
        else:
            raise LaunchError(message)
    plan.step('任务全集',
              f'{len(tasks)} 个任务（来源 {args.task_source}）；'
              f'每任务 scout {scout_trajs} 条 val 回合；manifest={manifest}')

    # ---------------- 步骤 5：并发占用检查 ----------------
    busy = list_other_runs(['al_launch.py'])
    if busy:
        lines = [f'  pid={row["pid"]} ({row["kind"]}): {row["cmdline"]}' for row in busy[:10]]
        if args.allow_busy:
            plan.warnings.append(f'检测到 {len(busy)} 个在跑的进程（--allow-busy 已放行）')
            log('[warn] 检测到在跑的训练/启动器（--allow-busy 已放行）：')
            for line in lines:
                log(line)
        else:
            raise LaunchError(
                f'检测到 {len(busy)} 个在跑的训练/启动器进程 ⇒ 拒绝启动（并行扫描与训练不能同时跑，'
                '会抢卡）：\n' + '\n'.join(lines)
                + '\n  确认可以同跑时加 --allow-busy。本工具不会替你杀进程。')
    plan.step('并发占用检查',
              f'未发现其它训练/启动器进程' if not busy else f'发现 {len(busy)} 个（已放行）')

    # ---------------- 步骤 6：运行环境（TMPDIR / 缓存目录 / 空间）----------------
    tmpdir = Path(args.tmpdir).expanduser()
    tmpdir_bad = (str(tmpdir).rstrip('/') in ('/tmp', '/var/tmp')
                  or str(tmpdir).startswith('/tmp/')
                  or str(tmpdir.resolve() if tmpdir.exists() else tmpdir).startswith('/private/tmp'))
    if tmpdir_bad:
        message = (f'TMPDIR 不能是 {tmpdir}（该机器 /tmp 只有 4 GB tmpfs，训练会崩）'
                   f'；请用 --tmpdir {DEFAULT_TMPDIR}')
        if dry:
            plan.warnings.append(message)
            log('[warn] ' + message)
        else:
            raise LaunchError(message)
    warn_tmp: List[str] = []
    for label, path in (('TMPDIR', tmpdir), ('TRITON_CACHE_DIR', Path(args.triton_cache)),
                        ('TORCHINDUCTOR_CACHE_DIR', Path(args.torchinductor_cache)),
                        ('worker-out-root', Path(args.worker_out_root))):
        free_gb, probe = check_disk(path, args.min_free_gb)
        if free_gb is None:
            warn_tmp.append(f'{label} {path}：{probe}')
        elif free_gb < args.min_free_gb:
            warn_tmp.append(f'{label} {probe} 可用空间 {free_gb:.1f}G < {args.min_free_gb:.0f}G')
    if warn_tmp:
        message = '空间/目录检查有问题：\n  - ' + '\n  - '.join(warn_tmp)
        if dry:
            plan.warnings.append(message)
            log('[warn] ' + message.replace('\n', '\n[warn] '))
        else:
            raise LaunchError(message)
    plan.step('运行环境',
              f'TMPDIR={tmpdir}；TRITON_CACHE_DIR={args.triton_cache}；'
              f'TORCHINDUCTOR_CACHE_DIR={args.torchinductor_cache}')

    # ---------------- 步骤 7：指纹 ----------------
    fingerprint = args.fingerprint
    fp_info: Dict[str, Any] = {'source': 'explicit' if fingerprint else 'computed'}
    if fingerprint is None:
        can_compute = checkpoint.is_dir() and manifest.is_file() and baseline.is_file() \
            and norm.is_file() and thresholds is not None and thresholds.is_file()
        if not can_compute:
            if dry:
                plan.warnings.append('指纹跳过：checkpoint/数据文件不齐（dry-run 不阻塞）')
                log('[dry-run] 指纹跳过：checkpoint/数据文件不齐。'
                    '真正运行前用 --fingerprint <64位hex> 跳过计算，或补齐路径。')
                fingerprint = None
            else:
                raise LaunchError('无法计算指纹：checkpoint/manifest/baseline/norm/阈值表 有缺失；'
                                  '可用 --fingerprint <64位hex> 显式指定')
        else:
            target_py = interpreter_version(python)
            mine_py = (sys.version_info.major, sys.version_info.minor)
            if target_py is not None and target_py != mine_py:
                log(f'[fingerprint] 本脚本 Python {mine_py[0]}.{mine_py[1]} 与训练解释器 '
                    f'{target_py[0]}.{target_py[1]} 不一致 ⇒ 改用训练解释器子进程计算'
                    '（ast.dump 输出与版本相关，必须同版本）')
                inner = [str(python), str(Path(__file__).resolve()), '--_print-fingerprint',
                         '--repo', str(repo), '--checkpoint', str(checkpoint),
                         '--manifest', str(manifest), '--norm', str(norm),
                         '--baseline', str(baseline), '--thresholds', str(thresholds),
                         '--dtype', dtype, '--scout-trajs', str(scout_trajs)]
                if args.image_augment:
                    inner.append('--image-augment')
                proc = subprocess.run(inner, capture_output=True, text=True, timeout=3600)
                payload: Dict[str, Any] = {}
                for line in reversed(proc.stdout.strip().splitlines()):
                    try:
                        payload = json.loads(line)
                        break
                    except ValueError:
                        continue
                if proc.returncode != 0 or not payload.get('ok'):
                    raise LaunchError('用训练解释器计算指纹失败：'
                                      + (payload.get('error') or proc.stderr[-1500:]))
                fingerprint = str(payload['fingerprint'])
                fp_info.update(payload.get('info') or {})
                fp_info['source'] = f'subprocess({python})'
            else:
                shards = sorted(checkpoint.glob('*.safetensors'))
                total_bytes = sum(p.stat().st_size for p in shards)
                log(f'[fingerprint] 正在哈希 {len(shards)} 个权重分片（约 '
                    f'{total_bytes / 1024 ** 3:.1f} GiB，只读）+ 评测链源码/数据文件…')
                fingerprint, info = compute_fingerprint(
                    repo_root=repo, checkpoint=checkpoint, manifest=manifest, norm=norm,
                    baseline=baseline, thresholds=thresholds, dtype=dtype,
                    scout_trajs=scout_trajs, image_augment=bool(args.image_augment))
                fp_info.update(info)
                fp_info['source'] = f'inprocess(Python {mine_py[0]}.{mine_py[1]})'
    if fingerprint is not None:
        report['fingerprint'] = fingerprint
        report['fingerprint_info'] = fp_info
        included = fp_info.get('sources_included')
        log(f'[fingerprint] {fingerprint}')
        log(f'[fingerprint] 来源={fp_info.get("source")}'
            + (f'；纳入 source {included} 个' if included else '')
            + (f'；权重分片 {len(fp_info.get("weight_shards", []))} 个'
               if fp_info.get('weight_shards') else ''))
        for item in fp_info.get('sources_missing', []) or []:
            log(f'[fingerprint][warn] source 未纳入：{item.get("key")}（{item.get("reason")}）')
        plan.step('指纹',
                  f'{fingerprint[:16]}…（{fp_info.get("source")}；'
                  f'纳入 source {included if included is not None else "?"} 个）')
    else:
        plan.step('指纹', '跳过（dry-run 且路径不全）')

    cache_dir = cache_root / fingerprint if fingerprint else cache_root / '<指纹>'
    report['cache_dir'] = str(cache_dir)
    report['tasks'] = {'total': len(tasks), 'source': args.task_source,
                       'names': list(tasks) if len(tasks) <= 60 else list(tasks)[:60]}

    # ---------------- 步骤 8：覆盖检查 ⇒ 决定扫描范围 ----------------
    scan_target: List[str] = []
    mode = 'unknown'
    cov_before: Optional[Coverage] = None
    if fingerprint is None:
        mode = 'unknown'
        scan_target = []
        plan.step('缓存覆盖检查', '跳过（没有指纹，dry-run 模式）')
    else:
        cov_before = check_coverage(cache_dir, fingerprint, scout_ids)
        report['coverage_before'] = cov_before.as_dict()
        log(f'[cache] 目录 {cache_dir}')
        log(f'[cache] 该目录下 {cov_before.json_files()} 个条目文件；'
            f'覆盖 {cov_before.summary()} 个任务')
        if args.no_cache:
            mode = 'full-scan'
            scan_target = list(tasks)
            plan.step('缓存覆盖检查',
                      f'--no-cache：现有覆盖 {cov_before.summary()} 一律忽略，'
                      f'全量并行扫描 {len(tasks)} 个任务'
                      f'（旧目录会改名备份为 <指纹>.bak-<时间戳>，不删除）')
        elif not cov_before.missing:
            mode = 'reuse'
            plan.step('缓存覆盖检查',
                      f'覆盖完整 {cov_before.summary()} ⇒ 直接启动训练，不扫描')
        else:
            mode = 'incremental'
            scan_target = list(cov_before.missing)
            plan.step('缓存覆盖检查',
                      f'覆盖 {cov_before.summary()}，缺失 {len(scan_target)} 个 ⇒ '
                      '只并行扫描缺失部分：' + ', '.join(scan_target[:8])
                      + ('…' if len(scan_target) > 8 else ''))
    report['mode'] = mode
    log(f'[plan] 运行模式：{mode}')

    # ---------------- 步骤 9：分片规划 ----------------
    shards: List[List[str]] = []
    shard_gpus: List[str] = []
    if scan_target:
        max_workers = min(len(gpus), len(scan_target))
        workers = int(args.workers) if args.workers else max_workers
        if workers < 1:
            raise LaunchError(f'--workers 必须 >= 1（收到 {args.workers}）')
        if workers > max_workers:
            log(f'[plan] --workers={workers} 超过 min(卡数 {len(gpus)}, 待扫任务 '
                f'{len(scan_target)})={max_workers} ⇒ 收敛为 {max_workers}')
            workers = max_workers
        shards = shard_tasks(scan_target, workers)
        check_union(scan_target, shards)
        shard_gpus = gpus[:workers]
        sizes = ', '.join(f'片{i}={len(s)}' for i, s in enumerate(shards))
        log(f'[plan] 待扫 {len(scan_target)} 个任务 ⇒ {workers} 片：{sizes}')
        log('[plan] 分片并集校验：通过（无重复、无遗漏，合计 '
            f'{sum(len(s) for s in shards)} = 全集 {len(scan_target)}）')
        for i, shard in enumerate(shards):
            log(f'[plan]   片{i}（GPU {shard_gpus[i]}）：{", ".join(shard)}')
        plan.step('分片规划',
                  f'{workers} 片 / {len(scan_target)} 任务；并集校验通过；'
                  f'各片大小 {[len(s) for s in shards]}；卡 {",".join(shard_gpus)}')
    else:
        plan.step('分片规划', '无需扫描（复用缓存）')

    # ---------------- 步骤 10：环境变量清单 ----------------
    run_name = args.run_name or f'{Path(args.eval_config).stem}_{_stamp()}'
    worker_out_root = Path(args.worker_out_root).expanduser()
    run_root = worker_out_root / run_name
    train_out = Path(args.train_out).expanduser() if args.train_out else \
        worker_out_root / f'train_{run_name}'
    common = base_env(args=args, repo=repo, python=python, al_cfg=al_cfg,
                      train_config=train_config, split_dir=split_dir, phases=phases,
                      checkpoint=checkpoint, manifest=manifest, baseline=baseline, norm=norm,
                      dtype=dtype)
    fp_for_env = fingerprint or '<指纹：dry-run 未计算>'
    # 即使本次不需要扫描（缓存命中）或任务未知，也把 worker 的环境变量模板完整打印出来，
    # 让 --dry-run 的「环境变量清单」永远是完整的。
    example_worker_env = worker_env(
        common, gpu=(shard_gpus[0] if shard_gpus else gpus[0]),
        train_out=run_root / 'scan_shard0', max_steps=args.worker_max_steps,
        fingerprint=fp_for_env, master_port=free_port(62510),
        smoke_no_checkpoint=not args.worker_checkpoint)
    train_env_vars = train_env(common, gpus=','.join(gpus), n_gpu=len(gpus), train_out=train_out,
                              max_steps=args.steps, fingerprint=fp_for_env,
                              master_port=free_port(62500), tb_port=args.tb_port,
                              tb=not args.no_tb)
    report['env'] = {
        'common': common,
        'worker_example': example_worker_env,
        'worker_shard_config': {f'shard{i}': str((run_root / 'shards' / f'al_shard{i}.json'))
                                for i in range(len(shards))},
        'train': train_env_vars,
    }
    log('')
    log('---- 环境变量：扫描 worker（示意片0；每片只差 GPU / 分片配置 / TRAIN_OUT / 端口）----')
    if not scan_target:
        reason = '缓存命中，本次无需扫描' if mode == 'reuse' else \
            ('指纹未知（dry-run 且路径不全），下面是模板' if mode == 'unknown' else '本次无需扫描')
        log(f'  （{reason}：把 N_GPU=1、CUDA/HIP_VISIBLE_DEVICES、AL_CFG=<分片配置> 按片替换即可）')
    for key, value in example_worker_env.items():
        log(f'  {key}={value}')
    log('')
    log('---- 环境变量：训练 ----')
    for key, value in train_env_vars.items():
        log(f'  {key}={value}')
    log('')
    plan.step('环境变量', f'worker {len(example_worker_env)} 个变量；'
                         f'训练 {len(train_env_vars)} 个变量（上面已完整打印）')

    report['plan'] = plan.steps
    report['warnings'] = plan.warnings

    # ---------------- dry-run 到此为止 ----------------
    if dry:
        log('=' * 78)
        log('步骤计划（--dry-run，不启动任何进程）')
        log('=' * 78)
        for step_row in plan.steps:
            log(f'  {step_row["step"]:>2}. {step_row["title"]}：{step_row["detail"]}')
        log('')
        log('将会执行的等价命令（示意片0' + ('' if scan_target else '；模板') + '）：')
        cmd_worker = equivalent_command(launch_script, example_worker_env,
                                        run_root / 'scan_shard0' / 'worker_shard0.log', repo)
        log('  ' + cmd_worker)
        report['commands'] = {'worker_shard0': cmd_worker}
        cmd_train = equivalent_command(launch_script, train_env_vars,
                                       worker_out_root / 'logs' / f'train_{run_name}.log', repo)
        log('  ' + cmd_train)
        report.setdefault('commands', {})['train'] = cmd_train
        if plan.warnings:
            log('')
            log(f'预演警告 {len(plan.warnings)} 条：')
            for item in plan.warnings:
                log('  - ' + item.replace('\n', ' '))
        log('')
        log('[dry-run] 预演完成。真正启动请去掉 --dry-run。')
        report['ok'] = True
        return EXIT_OK

    # ---------------- 步骤 11：--no-cache 备份旧目录 / 生成分片配置 ----------------
    if fingerprint and args.no_cache and cache_dir.exists() and any(cache_dir.glob('*.json')):
        backup = cache_dir.with_name(cache_dir.name + '.bak-' + _stamp())
        if backup.exists():
            backup = cache_dir.with_name(cache_dir.name + f'.bak-{_stamp()}-{os.getpid()}')
        cache_dir.rename(backup)
        log(f'[cache] --no-cache：旧指纹目录已改名备份（不删除）⇒ {backup}')
    for label, path in (('TMPDIR', tmpdir), ('TRITON_CACHE_DIR', Path(args.triton_cache)),
                        ('TORCHINDUCTOR_CACHE_DIR', Path(args.torchinductor_cache)),
                        ('worker-out-root', worker_out_root), ('run 目录', run_root),
                        ('缓存目录', cache_dir)):
        path.mkdir(parents=True, exist_ok=True)

    shard_configs: List[Path] = []
    shard_dir = run_root / 'shards'
    shard_dir.mkdir(parents=True, exist_ok=True)
    for i, shard in enumerate(shards):
        if args.shard_config_dir:
            path = Path(args.shard_config_dir).expanduser() / f'al_shard{i}.yaml'
            if not path.is_file():
                raise LaunchError(f'--shard-config-dir 里缺 {path}')
        else:
            path = shard_dir / f'al_shard{i}.json'
            write_shard_config(path, index=i, total=len(shards), body=al_body,
                               tasks=shard, source_config=al_cfg)
        shard_configs.append(path)
    if shard_configs:
        log(f'[plan] 分片 AL 配置：{", ".join(p.name for p in shard_configs)}'
            f'（task_names 分片，其余标量原样复制自 {al_cfg}）')

    # ---------------- 步骤 12：启动 worker ----------------
    workers_rows: List[Dict[str, Any]] = []
    live_pids: List[int] = []
    log('=' * 78)
    log(f'启动 {len(shards)} 个扫描 worker（各 1 卡 + MAX_STEPS={args.worker_max_steps}'
        f'{" + SMOKE_NO_CHECKPOINT=1" if not args.worker_checkpoint else ""}）')
    log('=' * 78)
    for i, shard in enumerate(shards):
        gpu = shard_gpus[i]
        cfg = shard_configs[i]
        wdir = run_root / f'scan_shard{i}'
        wdir.mkdir(parents=True, exist_ok=True)
        wlog = wdir / f'worker_shard{i}_gpu{gpu}.log'
        env = worker_env(common, gpu=gpu, train_out=wdir, max_steps=args.worker_max_steps,
                         fingerprint=str(fingerprint), master_port=free_port(62510 + i * 10),
                         smoke_no_checkpoint=not args.worker_checkpoint)
        env['AL_CFG'] = str(cfg)
        pid = spawn_background(script=launch_script, env=env, log_path=wlog, cwd=repo,
                               shell_env=shell_env_for_launch(python), log=log)
        time.sleep(0.4)
        alive = proc_alive(pid)
        argv = proc_cmdline(pid)
        log(f'[scan] 片{i} GPU{gpu}：{len(shard)} 任务；pid={pid}；日志 {wlog}')
        log(f'[scan]   cmdline: {" ".join(argv)[:160]}')
        workers_rows.append({'index': i, 'gpu': gpu, 'tasks': list(shard), 'pid': pid,
                             'log': str(wlog), 'out': str(wdir), 'config': str(cfg),
                             'alive': alive})
        live_pids.append(pid)
        if not alive:
            raise LaunchError(f'片{i}（pid={pid}）启动后立刻退出 ⇒ 拒绝继续；'
                              f'日志尾部：\n{read_tail(wlog, 4000)}',
                              EXIT_WORKER_NOT_READY)
    report['shards'] = workers_rows

    # ---------------- 步骤 13：就绪检查 ----------------
    started = time.time()
    first_check_done = False
    ready_seen: Dict[int, int] = {}
    log(f'[ready] 等待就绪：首轮检查在 T+{args.ready_wait:.0f}s，'
        f'上限 T+{args.ready_timeout:.0f}s')
    while True:
        now = time.time()
        alive_count = 0
        ready_count = 0
        problems: List[str] = []
        for row in workers_rows:
            pid = int(row['pid'])
            alive = proc_alive(pid)
            row['alive'] = alive
            text = read_tail(Path(row['log']))
            match = RE_READY.search(text)
            expected = len(row['tasks'])
            row['expected'] = expected
            if match:
                got = int(match.group(1))
                row['ready_tasks'] = got
                if got != expected:
                    problems.append(f'片{row["index"]}：日志「已就绪：{got} 任务」'
                                    f'与分片任务数 {expected} 不符（分片配置没生效？）')
                else:
                    ready_seen[int(row['index'])] = got
            if alive:
                alive_count += 1
            elif not match:
                fatal = detect_fatal(text) or '（日志里没有「已就绪」，也没有明显异常）'
                problems.append(f'片{row["index"]}（pid={pid}）已退出且未就绪：{fatal}')
            fatal_now = detect_fatal(text)
            if fatal_now and not match:
                problems.append(f'片{row["index"]}：日志出现致命错误「{fatal_now}」')
        ready_count = len(ready_seen)
        elapsed = now - started
        if not first_check_done and elapsed >= args.ready_wait:
            first_check_done = True
            log(f'[ready] T+{elapsed:.0f}s：进程存活 {alive_count}/{len(workers_rows)}；'
                f'已就绪 {ready_count}/{len(workers_rows)}'
                + (f'（未就绪 {len(workers_rows) - ready_count} 个仍在加载模型）'
                   if ready_count < len(workers_rows) else ''))
        if problems:
            for row in workers_rows:
                if proc_alive(int(row['pid'])):
                    terminate_own(int(row['pid']), expect_tokens=('al_50task_bf16.sh',
                                                                  'train_lingbotvla.py'),
                                  role=f'scan shard {row["index"]}', log=log)
            detail = '\n  - '.join(problems)
            for row in workers_rows:
                log(f'[ready] 片{row["index"]} 日志尾部：\n'
                    + '\n'.join('    ' + ln for ln in read_tail(Path(row['log']), 3000)
                                .splitlines()[-25:]))
            raise LaunchError(f'worker 存活/就绪检查未通过：\n  - {detail}', EXIT_WORKER_NOT_READY)
        if ready_count == len(workers_rows) and alive_count == len(workers_rows):
            log(f'[ready] T+{elapsed:.0f}s：全部就绪 —— '
                + '；'.join(f'片{r["index"]}（GPU{r["gpu"]}）就绪 {len(r["tasks"])} 任务'
                            for r in workers_rows))
            break
        if alive_count == 0:
            break
        if elapsed > args.ready_timeout:
            for row in workers_rows:
                if proc_alive(int(row['pid'])):
                    terminate_own(int(row['pid']), expect_tokens=('al_50task_bf16.sh',
                                                                  'train_lingbotvla.py'),
                                  role=f'scan shard {row["index"]}', log=log)
            raise LaunchError(f'等待就绪超过 {args.ready_timeout:.0f}s（已就绪 {ready_count}'
                              f'/{len(workers_rows)}）⇒ 终止本次启动', EXIT_WORKER_NOT_READY)
        time.sleep(2.0)

    # ---------------- 步骤 14：逐分片进度（以缓存条目数为准）----------------
    log('=' * 78)
    log('扫描进行中（进度以缓存条目数为准；事件文件可能有残留数据，不用于判定）')
    log('=' * 78)
    deadline = time.time() + args.scan_timeout
    last_line = ''
    while True:
        cov_now = check_coverage(cache_dir, str(fingerprint), scout_ids)
        parts: List[str] = []
        for row in workers_rows:
            done = sum(1 for t in row['tasks'] if t in cov_now.covered)
            row['scan_done'] = done
            parts.append(f'片{row["index"]}(gpu{row["gpu"]}) {done}/{len(row["tasks"])}')
        line = (f'[progress] T+{_human(time.time() - started)} | ' + ' | '.join(parts)
                + f' | 合计 {len(cov_now.covered)}/{len(scout_ids)}（指纹目录内）')
        if line != last_line:
            log(line)
            last_line = line
        if not any(proc_alive(int(row['pid'])) for row in workers_rows):
            log('[progress] 所有 worker 已退出')
            break
        if time.time() > deadline:
            log(f'[progress] 超过 --scan-timeout={args.scan_timeout:.0f}s ⇒ 精确终止仍在跑的 worker')
            for row in workers_rows:
                if proc_alive(int(row['pid'])):
                    terminate_own(int(row['pid']), expect_tokens=('al_50task_bf16.sh',
                                                                  'train_lingbotvla.py'),
                                  role=f'scan shard {row["index"]}', log=log)
            break
        time.sleep(max(2.0, float(args.poll_interval)))

    # 收尾：覆盖已完整但 worker 还在跑（可能在做 1 个训练步）⇒ 给宽限
    grace_deadline = time.time() + max(0.0, float(args.exit_grace))
    while time.time() < grace_deadline:
        cov_now = check_coverage(cache_dir, str(fingerprint), scout_ids)
        if len(cov_now.missing) == 0 and not any(proc_alive(int(r['pid'])) for r in workers_rows):
            break
        if not any(proc_alive(int(r['pid'])) for r in workers_rows):
            break
        time.sleep(2.0)
    for row in workers_rows:
        if proc_alive(int(row['pid'])):
            log(f'[cleanup] 片{row["index"]} 覆盖已齐但进程仍在（超过宽限 '
                f'{args.exit_grace:.0f}s）⇒ 精确终止')
            terminate_own(int(row['pid']), expect_tokens=('al_50task_bf16.sh',
                                                          'train_lingbotvla.py'),
                          role=f'scan shard {row["index"]}', log=log)
    for row in workers_rows:
        row['exited'] = not proc_alive(int(row['pid']))

    # ---------------- 步骤 15：收尾统计（只看目标指纹目录）----------------
    final_cov = check_coverage(cache_dir, str(fingerprint), scout_ids)
    target_cov = Coverage(cache_dir, str(fingerprint))
    for task in scan_target:
        if task in final_cov.rows:
            target_cov.rows[task] = final_cov.rows[task]
    report['coverage_after'] = final_cov.as_dict()
    report['coverage_target'] = target_cov.as_dict()
    report['shards'] = workers_rows
    log('=' * 78)
    log('扫描收尾统计（只统计目标指纹目录）')
    log('=' * 78)
    log(f'目标指纹目录  : {cache_dir}')
    log(f'本次扫描范围  : {len(scan_target)} 个任务'
        + ('（--no-cache 全量）' if args.no_cache else '（缓存缺失部分）')
        if scan_target else '本次扫描范围  : 无（复用缓存）')
    log(f'范围覆盖      : {target_cov.summary()}')
    log(f'全集覆盖      : {final_cov.summary()}（该目录内 {final_cov.json_files()} 个条目文件）')
    if final_cov.missing:
        log(f'缺失任务（{len(final_cov.missing)} 个）：' + ', '.join(final_cov.missing))
        for task, reason in list(final_cov.reasons.items())[:10]:
            log(f'  - {task}: {reason}')
        shard_of = {t: i for i, row in enumerate(workers_rows) for t in row['tasks']}
        log('缺失任务所属分片：'
            + ', '.join(f'{t}→片{shard_of.get(t, "?")}' for t in final_cov.missing[:15]))
        log('结论：覆盖不完整 ⇒ 不启动训练，请按上面的缺失清单补齐后重跑（本工具不会伪造覆盖数）。')
        report['ok'] = False
        return EXIT_SCAN_INCOMPLETE
    log('缺失清单      : 无')
    log(f'结论：覆盖完整（{final_cov.summary()}）⇒ 启动训练。')
    report['coverage_complete'] = True

    # ---------------- 步骤 16：启动训练 ----------------
    train_log_dir = worker_out_root / 'logs'
    train_log = train_log_dir / f'train_{run_name}.log'
    train_out.mkdir(parents=True, exist_ok=True)
    train_log_dir.mkdir(parents=True, exist_ok=True)
    log('=' * 78)
    log('启动训练')
    log('=' * 78)
    port = free_port(62500)
    tenv = train_env(common, gpus=','.join(gpus), n_gpu=len(gpus), train_out=train_out,
                     max_steps=args.steps, fingerprint=str(fingerprint), master_port=port,
                     tb_port=args.tb_port, tb=not args.no_tb)
    train_pid = spawn_background(script=launch_script, env=tenv, log_path=train_log, cwd=repo,
                                 shell_env=shell_env_for_launch(python), log=log)
    time.sleep(1.0)
    alive = proc_alive(train_pid)
    argv = proc_cmdline(train_pid)
    report['training'] = {'launched': True, 'pid': train_pid, 'log': str(train_log),
                          'out': str(train_out), 'ranks': len(gpus), 'alive': alive,
                          'cmdline': ' '.join(argv)[:200], 'master_port': port}
    report.setdefault('commands', {})['train'] = equivalent_command(
        launch_script, tenv, train_log, repo)
    log(f'日志   : {train_log}')
    log(f'PID    : {train_pid}（会话/进程组 {proc_pgid(train_pid)}；启动 1s 后存活={alive}）')
    log(f'进程数 : {len(gpus)} 个 rank（N_GPU={len(gpus)}；'
        f'CUDA_VISIBLE_DEVICES={",".join(gpus)}）')
    log(f'输出   : {train_out}')
    log(f'配置   : AL_CFG={al_cfg}；MAX_STEPS={args.steps}；'
        f'GBS={args.micro * args.gas * len(gpus)}；MASTER_PORT={port}')
    log('查看   : tail -f ' + str(train_log))
    if not alive:
        log(f'[error] 训练进程启动后 1s 内已退出 ⇒ 请立刻看日志尾部：\n'
            + '\n'.join('  ' + ln for ln in read_tail(train_log, 4000).splitlines()[-30:]))
        report['ok'] = False
        report['training']['alive'] = False
        return EXIT_WORKER_NOT_READY
    report['ok'] = True
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args._print_fingerprint:
        return _print_fingerprint_mode(args)

    report: Dict[str, Any] = {
        'tool': 'experiment/robotwin/al_launch.py', 'utc': _now_utc(), 'ok': False,
        'exit_code': None, 'argv': list(sys.argv[1:] if argv is None else argv),
    }
    log_path: Optional[Path] = None
    if args.log_file:
        log_path = Path(args.log_file).expanduser()
    elif not args.dry_run and not args.selfcheck:
        log_path = Path(args.worker_out_root).expanduser() / 'logs' / f'al_launch_{_stamp()}.log'
    log = Tee(log_path, to_stream=True, json_mode=bool(args.json))
    report['log_file'] = str(log_path) if log_path else None
    if log.file_error:
        log(f'[warn] 日志文件不可写（{log.file_error}）⇒ 本次只写 stdout / stderr')
        report.setdefault('warnings', []).append(f'日志文件不可写：{log.file_error}')
    code = EXIT_OK
    try:
        if args.selfcheck:
            code = _selfcheck(args, log, report)
        else:
            code = run(args, log, report)
    except LaunchError as exc:
        code = int(exc.code)
        report['ok'] = False
        report.setdefault('errors', []).append(str(exc))
        log('')
        log('=' * 78)
        log(f'前置检查未通过（退出码 {code}）')
        log('=' * 78)
        for line in str(exc).splitlines():
            log('  ' + line)
    except KeyboardInterrupt:
        code = EXIT_INTERNAL
        report.setdefault('errors', []).append('用户中断（KeyboardInterrupt）')
        log('[error] 用户中断')
    except Exception as exc:  # noqa: BLE001 —— 顶层兜底：打印可读信息，不吐裸 traceback
        code = EXIT_INTERNAL
        import traceback
        report.setdefault('errors', []).append(f'{type(exc).__name__}: {exc}')
        log(f'[error] 内部错误：{type(exc).__name__}: {exc}')
        log(traceback.format_exc())
    report['exit_code'] = code
    log('')
    log(f'[done] 退出码 {code}')
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log.close()
    return code


if __name__ == '__main__':
    sys.exit(main())
