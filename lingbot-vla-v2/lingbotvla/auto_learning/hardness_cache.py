"""Hardness 采样结果的**磁盘缓存**（2026-10-10）—— 跨 run 复用，避免每次重扫。

为什么需要
----------
`HardnessScan` 原本只在 **同一次 run 的 scheduler state** 里持久化
（`state/persistence.py` 的 `scans`），换 run（换 `TRAIN_OUT`、`enable_resume=false`）
就完全重算。真机实测：同一任务 `place_dual_shoes` 在 al_v15/v19/v21/v22/v23 里
被重扫了 5 次，每次 3–8 分钟，白烧 25–40 分钟 GPU。

语义（决定指纹怎么写）
----------------------
hardness 分数 = **模型权重固定**下、该样本的 flow-matching loss（`loss_type=L1_fm`），
噪声由 **per-sample-id RNG**（`seed ^ sid*const`）决定、flow time 固定。
⇒ 分数只依赖：**权重内容 + 噪声种子/语义 + 该任务的帧 + 评测链与打分代码**。
⇒ 而且逐样本 loss 是**批无关**的（分片改造时已强制绑定 RNG）⇒ 天然支持
**部分复用**：缺哪个样本只补哪个。

🔴 因此指纹**刻意不含** `hardness_probe_fraction` / batch：
不同 probe 设置扫的是**不同样本子集**，缓存按样本做并集，彼此互补而不是互相作废。

写者约定（与 scout 缓存一致）
----------------------------
* 只有 rank0 写（`write_enabled`）；全体只读。
* 原子替换：写 `.tmp` → `os.replace`，避免读到半截 JSON。
* 任何缓存异常（IO/JSON/schema）**只 warning，绝不打断训练**。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .scout_cache import EVAL_SOURCES, semantic_sha256, sha256_file

VERSION = 1
#: 逐样本噪声语义版本 —— 改了 `(seed, sid)` 的构造方式**必须**升这个数，
#: 否则旧缓存里的 loss 会被错误复用（分数已经不是同一口径）。
NOISE_SEMANTIC_VERSION = 1
MAX_JSON_BYTES = 8 * 1024 * 1024


def compute_hardness_fingerprint(
    *,
    checkpoint_dir: str | Path,
    repo_root: str | Path,
    noise_seed: int,
    flow_time: float,
    loss_type: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """算出"这批 hardness 分数还有效吗"的指纹。

    纳入：权重分片内容 + 评测链源码语义 hash + 噪声/损失口径 + 数据集身份（由调用方经 `extra` 传）。
    返回 ``(指纹, 明细)``；明细会印进日志，便于事后核对。
    """
    ckpt = Path(checkpoint_dir).expanduser()
    shards = sorted(ckpt.glob('*.safetensors'))
    if not shards:
        raise ValueError(f'hardness 指纹需要一个含 *.safetensors 的 checkpoint 目录：{ckpt}')

    sources: Dict[str, str] = {}
    for key, rel, _required in EVAL_SOURCES:
        path = Path(repo_root) / rel
        if path.is_file():
            sources[key] = (semantic_sha256(path) if path.suffix == '.py'
                            else sha256_file(path))

    payload = {
        'schema': VERSION,
        'noise_semantic_version': NOISE_SEMANTIC_VERSION,
        'weights': [(p.name, sha256_file(p)) for p in shards],
        'sources': dict(sorted(sources.items())),
        'options': {
            'noise_seed': int(noise_seed),
            'flow_time': float(flow_time),
            'loss_type': str(loss_type),
        },
        'extra': dict(sorted((extra or {}).items())),
    }
    blob = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest(), {
        'weight_shards': [p.name for p in shards],
        'sources_included': len(sources),
        'options': payload['options'],
        'extra': payload['extra'],
    }


class HardnessSampleCache:
    """按 **task** 一个 JSON，存逐样本 loss（键为 sample_id 的字符串形式）。"""

    def __init__(self, root: str | Path, fingerprint: str, *, enabled: bool = True,
                 write_enabled: bool = True, logger: Any = None,
                 node: Optional[str] = None):
        if len(str(fingerprint)) != 64 or any(c not in '0123456789abcdef' for c in str(fingerprint)):
            raise ValueError('invalid hardness fingerprint（需要 64 位小写 hex）')
        self.enabled = bool(enabled)
        self.write_enabled = bool(write_enabled) and self.enabled
        self.fingerprint = str(fingerprint)
        self.node = str(node) if node else f'{os.uname().nodename}-{os.getpid()}'
        #: 目录名加节点后缀：多机/多进程并行时同一指纹目录也能安全共存
        self.path = Path(root).expanduser() / f'{self.fingerprint}'
        self.logger = logger

    # -- 日志（失败绝不打断训练）------------------------------------------
    def _log(self, level: str, msg: str) -> None:
        if self.logger is None:
            return
        try:
            getattr(self.logger, level)(msg)
        except Exception:  # noqa: BLE001
            pass

    # -- 读 -----------------------------------------------------------------
    def _task_path(self, task: str) -> Path:
        safe = hashlib.sha256(str(task).encode()).hexdigest()[:16]
        return self.path / f'{safe}.json'

    def load(self, task: str, sample_ids: Sequence[int]) -> Tuple[Dict[int, float], List[int]]:
        """返回 ``(命中的逐样本 loss, 缺失的 sample_id 列表)``。"""
        want = [int(s) for s in sample_ids]
        if not self.enabled:
            return {}, list(want)
        p = self._task_path(task)
        if not p.is_file():
            return {}, list(want)
        try:
            if p.stat().st_size > MAX_JSON_BYTES:
                self._log('warning', f'[hardness_cache] {p} 超过上限 ⇒ 视为未命中')
                return {}, list(want)
            raw = json.loads(p.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            self._log('warning', f'[hardness_cache] 读取失败（{exc!r}）⇒ 视为未命中')
            return {}, list(want)
        if (raw.get('version') != VERSION or raw.get('fingerprint') != self.fingerprint
                or raw.get('task') != task or raw.get('noise_semantic_version') != NOISE_SEMANTIC_VERSION):
            return {}, list(want)
        stored = raw.get('losses') or {}
        hits: Dict[int, float] = {}
        missing: List[int] = []
        for sid in want:
            v = stored.get(str(sid))
            if v is None:
                missing.append(sid)
            else:
                hits[sid] = float(v)
        return hits, missing

    # -- 写 -----------------------------------------------------------------
    def store(self, task: str, losses: Mapping[int, float]) -> None:
        """把本次扫到的 loss 并入该任务的缓存（**按样本并集**，不覆盖别的样本）。"""
        if not self.write_enabled or not losses:
            return
        p = self._task_path(task)
        merged: Dict[str, float] = {}
        try:
            if p.is_file():
                raw = json.loads(p.read_text(encoding='utf-8'))
                if (raw.get('version') == VERSION and raw.get('fingerprint') == self.fingerprint
                        and raw.get('task') == task
                        and raw.get('noise_semantic_version') == NOISE_SEMANTIC_VERSION):
                    merged.update({str(k): float(v) for k, v in (raw.get('losses') or {}).items()})
        except (OSError, ValueError, TypeError):
            merged = {}          # 旧文件坏了就重写，不打断
        merged.update({str(int(k)): float(v) for k, v in losses.items()})
        payload = {
            'version': VERSION,
            'noise_semantic_version': NOISE_SEMANTIC_VERSION,
            'fingerprint': self.fingerprint,
            'task': str(task),
            'losses': merged,
            'n': len(merged),
        }
        try:
            blob = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(',', ':'))
            if len(blob.encode()) > MAX_JSON_BYTES:
                self._log('warning', f'[hardness_cache] {task} 缓存超过上限 ⇒ 不写')
                return
            self.path.mkdir(parents=True, exist_ok=True)
            tmp = self.path / f'.{p.name}.{self.node}.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
            self._log('info', f'[hardness_cache] 已写入 {task}：累计 {len(merged)} 个样本'
                              f'（本次新增 {len(losses)}）')
        except (OSError, ValueError, TypeError) as exc:
            self._log('warning', f'[hardness_cache] 写入失败（{exc!r}）⇒ 忽略，不影响训练')
        finally:
            try:
                (self.path / f'.{p.name}.{self.node}.tmp').unlink(missing_ok=True)
            except OSError:
                pass

    # -- 统计（日志/测试用）--------------------------------------------------
    def cached_count(self, task: str) -> int:
        if not self.enabled:
            return 0
        p = self._task_path(task)
        if not p.is_file():
            return 0
        try:
            raw = json.loads(p.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return 0
        if raw.get('fingerprint') != self.fingerprint:
            return 0
        return len(raw.get('losses') or {})


__all__ = ['HardnessSampleCache', 'compute_hardness_fingerprint',
           'NOISE_SEMANTIC_VERSION', 'VERSION']
