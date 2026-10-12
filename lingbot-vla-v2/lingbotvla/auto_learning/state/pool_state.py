"""池子状态的**独立**存档（`pool_state.json`）—— 2026-10-12 用户要求。

## 为什么需要它

Auto Learning 的"池子"（每任务 status / attempts / lp50 / gap / `sample_probs` …）原本
**只**随 DCP 的 `extra_state["auto_learning"]` 走。于是：

* **DCP 续训**：池子自动恢复（`load_extra_state()`），没问题；
* **从 HF 权重起新 run**（本项目常见做法：`--model.model_path=<上一次的 hf_ckpt>`）：
  `_al_resume_mode == "fresh"` ⇒ **池子从零**（日志原文"Auto Learning 状态从零开始"）
  ⇒ 已经 PASS 的任务会被重新训、刚 DEFER 的会被立刻重试，**白花算力**。

本模块提供"池子独立导出/导入"：每次存档时把整块 `auto_learning` 写成一个 JSON 放在
**该存档旁边**（`<global_step_N>/pool_state.json`），HF 目录里也有一份
（`<global_step_N>/hf_ckpt/pool_state.json`）⇒ 新 run 从哪份 HF 起，就在**同目录**读到对应池子。

## 设计要点（都是踩坑后定的）

1. **整块导出**：含 `registry`（其中 `sample_probs` 是逐样本采样权重，4548 项/任务）、
   `sampler`、`sampler_rng` ⇒ 权重采样不丢（用户明确要求保留）。
   实测单份 ≈ 9.4 MB（50 任务）。
2. **原子写**：先写 `.tmp` 再 `os.replace` ⇒ 不会留"半个 JSON"（存档被打断也安全）。
3. **不阻塞训练**：导出是"读内存里的 state + 写 9 MB"（毫秒级），且**调用方必须用
   `export_pool_state_safely()` 包住** ⇒ 任何异常只告警、**绝不中断训练**。
4. **`sampler_rng` 默认不恢复**：从 HF 起新 run 时，旧 run 的抽样骰子串**没有续训语义**
   （权重都换了一份）；池子与权重采样照常恢复。要复现旧骰子序列可设
   `AL_POOL_STATE_RESTORE_RNG=1`。
5. **不做一致性拦截**：导入时若"池子任务集合"与当前 manifest 对不上，**只摊开差异警告**
   （`AL_POOL_STATE_NO_IMPORT=1` 可跳过导入），不拦启动 —— 用户明确要求"读就是了"。

## 环境变量（都在**进程启动时**读取，改完要重启）

| 变量 | 默认 | 作用 |
|---|---|---|
| `AL_POOL_STATE_FILE` | 空 | 显式指定池子文件（最高优先，不做任何查找） |
| `AL_POOL_STATE_NO_IMPORT` | `0` | `1` ⇒ 不导入（干净起跑 / 对照实验） |
| `AL_POOL_STATE_EXPORT` | `1` | `0` ⇒ 不导出 |
| `AL_POOL_STATE_RESTORE_RNG` | `0` | `1` ⇒ 连 `sampler_rng` 一起恢复 |
| `AL_POOL_STATE_MIRROR` | `/workspace/al/pool_state` | 集中目录（另存一份带 step 的文件；设空串关闭） |
| `AL_POOL_STATE_PRUNE` | `2` | 集中目录保留最近几份（0 = 全留） |
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..cache_paths import model_slug

__all__ = [
    'POOL_FILENAME', 'POOL_ENV', 'NO_IMPORT_ENV', 'EXPORT_ENV', 'RESTORE_RNG_ENV',
    'MIRROR_ENV', 'PRUNE_ENV', 'DEFAULT_MIRROR',
    'export_enabled', 'import_disabled', 'restore_rng_enabled', 'mirror_root',
    'pool_filename', 'export_pool_state', 'export_pool_state_safely',
    'load_pool_state', 'pool_candidates', 'find_pool_state_file',
    'prune_mirror', 'summarize_pool', 'resolve_target',
]

POOL_FILENAME = 'pool_state.json'
POOL_ENV = 'AL_POOL_STATE_FILE'
NO_IMPORT_ENV = 'AL_POOL_STATE_NO_IMPORT'
EXPORT_ENV = 'AL_POOL_STATE_EXPORT'
RESTORE_RNG_ENV = 'AL_POOL_STATE_RESTORE_RNG'
MIRROR_ENV = 'AL_POOL_STATE_MIRROR'
PRUNE_ENV = 'AL_POOL_STATE_PRUNE'
DEFAULT_MIRROR = '/workspace/al/pool_state'

_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# 开关（进程启动时读一次，与 AL_EVAL_SHARD 等保持一致）
# --------------------------------------------------------------------------- #
def export_enabled() -> bool:
    """是否在存档时导出池子（`AL_POOL_STATE_EXPORT=0` 关闭）。"""
    return os.environ.get(EXPORT_ENV, '1') != '0'


def import_disabled() -> bool:
    """是否**禁止**导入池子（`AL_POOL_STATE_NO_IMPORT=1` ⇒ 干净起跑/对照）。"""
    return os.environ.get(NO_IMPORT_ENV, '0') == '1'


def restore_rng_enabled() -> bool:
    """是否连 `sampler_rng` 一起恢复（默认 False；见模块文档第 4 点）。"""
    return os.environ.get(RESTORE_RNG_ENV, '0') == '1'


def mirror_root() -> Optional[Path]:
    """集中镜像目录；设为空串 ⇒ 不写镜像。"""
    raw = os.environ.get(MIRROR_ENV, DEFAULT_MIRROR)
    return Path(raw) if raw else None


def pool_filename(ckpt_dir: Any, *, subdir: Optional[str] = None) -> Path:
    """某个存档旁的池子文件路径。

    `<ckpt_dir>/pool_state.json`，`subdir='hf_ckpt'` ⇒ `<ckpt_dir>/hf_ckpt/pool_state.json`
    （HF 目录里也放一份 ⇒ 新 run 用 `--model.model_path` 指向它时**同目录就能读到**）。
    """
    base = Path(ckpt_dir)
    if subdir:
        base = base / subdir
    return base / POOL_FILENAME


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def resolve_target(path: Any) -> Path:
    """把"目录或文件名"解析成**确定的文件路径**（写盘与日志上报都用它）。

    口径（**无歧义**）：**已存在的目录** ⇒ `<dir>/pool_state.json`；否则当成文件名。
    ⚠️ 不存在的路径一律按"文件名"处理（父目录自动建）—— 曾用"父目录不存在 ⇒ 当目录"做
    启发式，结果把镜像文件名 `pool_state.<key>.step2400.json` 误判成目录（单测抓到）。
    """
    p = Path(path)
    return (p / POOL_FILENAME) if p.is_dir() else p


def _atomic_write_json(path: Any, payload: Dict[str, Any]) -> int:
    """原子写：`<path>.tmp` → `os.replace`。返回字节数。路径口径见 `resolve_target()`。"""
    path = resolve_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, ensure_ascii=False)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.pool_state.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        # 🔴 `mkstemp` 建的文件是 **0600** ⇒ 若直接 replace，池子文件就变成"只有属主可读"。
        #    真机踩到过：手工脚本写的 644、模块写的 600 混在一起 ⇒ 换个用户跑续训就**静默读不到**
        #    （池子从零，正是本模块要避免的失败形态）。所以显式对齐常规文件权限（umask 语义）。
        try:
            os.chmod(tmp, 0o644)
        except OSError:
            pass
        os.replace(tmp, path)          # 同目录 rename ⇒ 原子
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return len(blob.encode('utf-8'))


def export_pool_state(
    ckpt_dir: Any,
    al_state: Optional[Dict[str, Any]],
    *,
    step: Optional[int],
    source_model_path: Optional[str] = None,
    source_checkpoint: Optional[str] = None,
    extra_dirs: Optional[List[Any]] = None,
    mirror_key: Optional[str] = None,
    prune_keep: Optional[int] = None,
) -> Dict[str, Any]:
    """把整块 `auto_learning` 写到 `<ckpt_dir>/pool_state.json`（+ 可选镜像目录）。

    Parameters
    ----------
    al_state
        `AutoLearnLoopHook.extra_state()` 的返回值（含 registry/sampler/sampler_rng）。
        `None`（例如非 AL run / hook 还没建）⇒ 直接返回 `{'skipped': 'no_al_state'}`。
    extra_dirs
        额外**目录**列表（例如同一个 `global_step_N` 下的 `hf_ckpt/`）⇒ 各写一份
        `<dir>/pool_state.json`。
    mirror_key
        镜像文件名里的 key（默认 `model_slug(source_model_path)`）。
        生成 `pool_state.<key>.step<N>.json`，便于集中目录里按"哪份权重"区分。

    Returns
    -------
    dict：`{'path':…, 'bytes':…, 'sha256':…, 'mirrors':[…], 'step':…}`；跳过时 `{'skipped': …}`。
    """
    if al_state is None:
        return {'skipped': 'no_al_state'}
    payload = {
        'version': _SCHEMA_VERSION,
        'exported_at': datetime.now().isoformat(timespec='seconds'),
        'global_step': step,
        'source_model_path': source_model_path,
        'source_checkpoint': source_checkpoint,
        'has_sampler_rng': bool(al_state.get('sampler_rng')),
        'auto_learning': al_state,
    }
    blob = json.dumps(payload, ensure_ascii=False)
    sha = hashlib.sha256(blob.encode('utf-8')).hexdigest()

    main = pool_filename(ckpt_dir)
    n_main = _atomic_write_json(main, payload)

    out: Dict[str, Any] = {
        'path': str(main), 'bytes': n_main, 'sha256': sha, 'step': step, 'mirrors': [],
    }
    for d in (extra_dirs or []):
        try:
            p = resolve_target(d)          # 目录 ⇒ <dir>/pool_state.json；文件名 ⇒ 原样
            _atomic_write_json(p, payload)
            out['mirrors'].append(str(p))
        except OSError as exc:
            out.setdefault('mirror_errors', []).append(f'{d}: {exc}')

    root = mirror_root()
    if root is not None:
        try:
            key = mirror_key or model_slug(source_model_path or '') or 'unknown'
            step_tag = f'.step{step}' if step is not None else ''
            mp = Path(root) / f'{Path(POOL_FILENAME).stem}.{key}{step_tag}.json'
            _atomic_write_json(mp, payload)
            out['mirrors'].append(str(mp))
            out['pruned'] = prune_mirror(root, key, keep=prune_keep)
        except OSError as exc:
            out.setdefault('mirror_errors', []).append(f'{root}: {exc}')
    return out


def export_pool_state_safely(logger: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """`export_pool_state` 的**绝不抛异常**版本 —— 存档路径上必须用它。

    为什么：导出只是"顺手记录池子"，任何失败（磁盘满、权限、序列化异常）
    **都不允许影响训练/存档**。失败 ⇒ rank0 打一行 warning 后返回 `{'error': …}`。
    """
    try:
        if not export_enabled():
            return {'skipped': EXPORT_ENV + '=0'}
        return export_pool_state(**kwargs)
    except Exception as exc:  # noqa: BLE001 —— 故意兜住一切
        msg = f'[pool_state] ⚠️ 导出失败（不影响训练）：{type(exc).__name__}: {str(exc)[:160]}'
        if logger is not None:
            try:
                logger.warning(msg)
            except Exception:  # noqa: BLE001
                pass
        return {'error': msg}


def prune_mirror(root: Any, key: str, *, keep: Optional[int] = None) -> List[str]:
    """集中目录里只保留最近 `keep` 份 `<key>` 的 step 文件（0/None ⇒ 用环境变量/不删）。

    返回被删除的文件名列表。⚠️ 只删 `pool_state.<key>.step*.json`，
    **绝不碰 `<ckpt_dir>/pool_state.json`**（那是随存档走的正本）。
    """
    if keep is None:
        try:
            keep = int(os.environ.get(PRUNE_ENV, '2') or 0)
        except ValueError:
            keep = 2
    if not keep or keep <= 0:
        return []
    files = sorted(Path(root).glob(f'{Path(POOL_FILENAME).stem}.{key}.step*.json'))
    gone: List[str] = []
    for p in files[:-keep]:
        try:
            p.unlink()
            gone.append(p.name)
        except OSError:
            pass
    return gone


# --------------------------------------------------------------------------- #
# 导入
# --------------------------------------------------------------------------- #
def load_pool_state(path: Optional[Any]) -> Optional[Dict[str, Any]]:
    """读池子 JSON；**文件不存在或损坏一律返回 None**（调用方据此"从零开始"）。

    不做任何校验/拦截（用户要求"读就是了"）；差异核对由调用方 `_check_task_set` 打日志。
    """
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with p.open(encoding='utf-8') as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get('auto_learning'), dict):
        return None
    return payload


def pool_candidates(model_path: Optional[str], *, run_dir: Optional[Any] = None) -> List[Path]:
    """按优先级给出"可能放着池子"的文件（越靠前越优先）。

    1. `AL_POOL_STATE_FILE` 显式指定；
    2. `<model_path>/pool_state.json` —— **首选**：池子跟在权重旁边（HF 目录里那份）；
    3. `<run_dir>/pool_state.json`（同 run 目录里那份）；
    4. `<MIRROR>/pool_state.<model_slug>.step*.json`（按 step 降序 ⇒ 最新优先）。
    """
    out: List[Path] = []
    env = os.environ.get(POOL_ENV) or ''
    if env:
        out.append(Path(env))
    if model_path:
        out.append(pool_filename(model_path))
    if run_dir:
        out.append(pool_filename(run_dir))
    root = mirror_root()
    if root is not None and model_path:
        try:
            key = model_slug(model_path)
            mirrors = sorted(
                Path(root).glob(f'{Path(POOL_FILENAME).stem}.{key}.step*.json'),
                key=lambda p: int(''.join(c for c in p.stem.rsplit('step', 1)[-1] if c.isdigit()) or 0),
                reverse=True,
            )
            out.extend(mirrors)
        except (OSError, ValueError):
            pass
    return out


def find_pool_state_file(model_path: Optional[str], *, run_dir: Optional[Any] = None) -> Optional[Path]:
    """返回**第一个真实存在**的候选文件；都没有 ⇒ None。"""
    for p in pool_candidates(model_path, run_dir=run_dir):
        if p.is_file():
            return p
    return None


# --------------------------------------------------------------------------- #
# 摘要 / 差异核对
# --------------------------------------------------------------------------- #
def summarize_pool(al_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """给日志用的池子摘要：状态计数 + 权重采样项数。"""
    reg = (al_state or {}).get('registry') or {}
    recs = reg.get('records') or {}
    counts: Dict[str, int] = {}
    n_probs = 0
    for rec in recs.values():
        st = str((rec or {}).get('status') or '?')
        counts[st] = counts.get(st, 0) + 1
        n_probs += len((rec or {}).get('sample_probs') or {})
    return {'n_tasks': len(recs), 'counts': counts, 'n_sample_probs': n_probs}


def describe_task_set_mismatch(pool_task_names: Any, manifest_task_names: Any) -> Optional[str]:
    """把"池子任务集合 vs 当前 manifest"的差异**摊开成可读文本**；一致 ⇒ None。

    为什么单独做：原先只打一句"切分不一致"，看不出**看什么、缺什么**（用户反馈读不懂）。
    """
    pool = set(pool_task_names or ())
    man = set(manifest_task_names or ())
    if pool == man:
        return None
    only_pool = sorted(pool - man)
    only_man = sorted(man - pool)
    both = len(pool & man)
    lines = ['[pool_state] ⚠️ 池子与当前 manifest 的任务集合对不上（照常继续，供你判断）：']
    lines.append('           池子里有、manifest 里没有：'
                 + (', '.join(only_pool[:6]) + (f'（共 {len(only_pool)} 个）' if len(only_pool) > 6
                                               else f'（{len(only_pool)} 个）') if only_pool else '（无）'))
    lines.append('           manifest 里有、池子里没有：'
                 + (', '.join(only_man[:6]) + (f'（共 {len(only_man)} 个）' if len(only_man) > 6
                                              else f'（{len(only_man)} 个）') if only_man else '（无）'))
    lines.append(f'           两侧都有：{both} 个；池子 {len(pool)} / manifest {len(man)}')
    lines.append(f'           ⇒ 若非预期请检查 manifest；要跳过导入设 {NO_IMPORT_ENV}=1')
    return '\n'.join(lines)
