"""Hardness 采样结果的**磁盘缓存**（单文件 + 模型名，2026-10-10 重写）。

为什么重写（旧"指纹目录"方案被否）
----------------------------------
旧实现把缓存路径写成 `<root>/<指纹64位hex>/<任务>.json`，指纹 = 14 个源码文件语义 hash +
6 个权重分片 hash + 噪声口径。实证代价：只改了一行 `modeling_lingbot_vla_v2.py`（加 SDPA 分支）
⇒ 指纹从 `18c9762c…` 变成 `8054756539…` ⇒ 旧缓存整体"看不见" ⇒ **7 个任务 4833 个样本全量重扫，
实测 ETA 约 2 小时**。

现行语义（用户 2026-10-10 定）
-----------------------------
1. **缓存文件由入参显式指定**（`--hardness-cache-file`），路径不参与任何哈希推导 ⇒ 跨 run 稳定复用；
2. **缓存里记 `model`**（`--model-name`，默认取权重目录名）；读取时 **model 不一致即视为未命中**
   并明确打印原因，绝不静默复用 ⇒ "换了模型必须重扫"由显式模型名承担；
3. 逐样本 loss 存在同一文件的 `tasks[task]["losses"]` 里，**按样本做并集**（不同 probe 设置
   扫的是不同子集，互补而非互相作废）。

写者约定：只有 rank0 写；`store()` 会**先重读文件再合并**（避免用陈旧的 `self._data` 覆盖别处
已写入的任务），随后 `.tmp` + `os.replace` 原子替换。任何异常只 warning，绝不打断训练。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

VERSION = 2
#: 逐样本噪声语义版本 —— 改了 `(seed, sid)` 的构造方式**必须**升这个数，
#: 否则旧缓存里的 loss 会被错误复用（分数已经不是同一口径）。
NOISE_SEMANTIC_VERSION = 1
MAX_JSON_BYTES = 64 * 1024 * 1024


def resolve_model_name(explicit: Optional[str] = None, *,
                       checkpoint_dir: Optional[str | Path] = None,
                       fallback: str = 'unknown') -> str:
    """模型标识：优先显式入参，其次权重目录名，最后 `fallback`。"""
    if explicit:
        return str(explicit)
    if checkpoint_dir:
        name = Path(checkpoint_dir).expanduser().name
        if name:
            return name
    return fallback


class HardnessSampleCache:
    """**单文件**逐样本 loss 缓存：`{"model": ..., "tasks": {task: {"losses": {sid: loss}}}}`。"""

    def __init__(self, file: str | Path, *, model: str, enabled: bool = True,
                 write_enabled: bool = True, logger: Any = None,
                 noise_seed: Optional[int] = None, loss_type: Optional[str] = None,
                 node: Optional[str] = None):
        if not str(model).strip():
            raise ValueError('model 不能为空（缓存以模型名判定是否可用）')
        self.enabled = bool(enabled)
        self.write_enabled = bool(write_enabled) and self.enabled
        self.model = str(model)
        self.path = Path(file).expanduser()
        self.logger = logger
        self.noise_seed = None if noise_seed is None else int(noise_seed)
        self.loss_type = None if loss_type is None else str(loss_type)
        self.node = str(node) if node else f'{os.uname().nodename}-{os.getpid()}'
        self._data: Optional[dict] = None
        #: 上次 load 是否落空及原因（供日志/测试断言）
        self.last_miss_reason: Optional[str] = None

    # -- 日志（失败绝不打断训练）------------------------------------------
    def _log(self, level: str, msg: str) -> None:
        if self.logger is None:
            return
        try:
            getattr(self.logger, level)(msg)
        except Exception:  # noqa: BLE001
            pass

    # -- 读写 ---------------------------------------------------------------
    def _blank(self) -> dict:
        return {'version': VERSION, 'noise_semantic_version': NOISE_SEMANTIC_VERSION,
                'model': self.model, 'tasks': {}}

    def _read(self, *, reason: str) -> Optional[dict]:
        """读文件；不可用（不存在/坏/模型不符）⇒ 返回 None，并记录原因。"""
        if not self.enabled:
            self.last_miss_reason = 'disabled'
            return None
        if not self.path.is_file():
            self.last_miss_reason = 'missing'
            return None
        try:
            if self.path.stat().st_size > MAX_JSON_BYTES:
                self._log('warning', f'[hardness_cache] {self.path} 超过上限 ⇒ 视为未命中')
                self.last_miss_reason = 'too_large'
                return None
            raw = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            self._log('warning', f'[hardness_cache] 读取失败（{exc!r}）⇒ 视为未命中')
            self.last_miss_reason = 'unreadable'
            return None
        if not isinstance(raw, dict):
            self.last_miss_reason = 'corrupt'
            return None
        if raw.get('version') != VERSION:
            self._log('warning',
                      f'[hardness_cache] schema={raw.get("version")} 与 {VERSION} 不符 ⇒ 视为未命中')
            self.last_miss_reason = 'schema'
            return None
        if raw.get('noise_semantic_version') != NOISE_SEMANTIC_VERSION:
            self.last_miss_reason = 'noise_semantic'
            return None
        if str(raw.get('model') or '') != self.model:
            self._log('info',
                      f'[hardness_cache] 因模型不同忽略缓存'
                      f'（file={raw.get("model")!r} ≠ now={self.model!r}）⇒ {reason} 将重扫')
            self.last_miss_reason = 'model_mismatch'
            return None
        if not isinstance(raw.get('tasks'), dict):
            self.last_miss_reason = 'corrupt'
            return None
        self.last_miss_reason = None
        return raw

    def load(self, task: str, sample_ids: Sequence[int]) -> Tuple[Dict[int, float], List[int]]:
        """返回 ``(命中的逐样本 loss, 缺失的 sample_id 列表)``。"""
        want = [int(s) for s in sample_ids]
        raw = self._read(reason=f'task={task}')
        if raw is None:
            return {}, list(want)
        self._data = raw
        stored = ((raw.get('tasks') or {}).get(str(task)) or {}).get('losses') or {}
        hits: Dict[int, float] = {}
        missing: List[int] = []
        for sid in want:
            v = stored.get(str(sid))
            if v is None:
                missing.append(sid)
            else:
                hits[sid] = float(v)
        return hits, missing

    def store(self, task: str, losses: Mapping[int, float]) -> None:
        """把本次扫到的 loss 并入该任务（**按样本并集**）；写入前重读文件避免覆盖别处成果。"""
        if not self.write_enabled or not losses:
            return
        raw = None
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                raw = None
        if (not isinstance(raw, dict) or raw.get('version') != VERSION
                or raw.get('noise_semantic_version') != NOISE_SEMANTIC_VERSION
                or str(raw.get('model') or '') != self.model
                or not isinstance(raw.get('tasks'), dict)):
            raw = self._blank()
        slot = raw['tasks'].setdefault(str(task), {})
        merged = slot.setdefault('losses', {})
        before = len(merged)
        merged.update({str(int(k)): float(v) for k, v in losses.items()})
        slot['noise_seed'] = self.noise_seed
        slot['loss_type'] = self.loss_type
        try:
            blob = json.dumps(raw, sort_keys=True, allow_nan=False, separators=(',', ':'))
            if len(blob.encode()) > MAX_JSON_BYTES:
                self._log('warning', f'[hardness_cache] 缓存超过上限 ⇒ 不写（{task}）')
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f'.{self.path.name}.{self.node}.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            self._log('info', f'[hardness_cache] 已写入 {task}：累计 {len(merged)} 个样本'
                              f'（本次新增 {len(merged) - before}）')
        except (OSError, ValueError, TypeError) as exc:
            self._log('warning', f'[hardness_cache] 写入失败（{exc!r}）⇒ 忽略，不影响训练')
        finally:
            try:
                self.path.with_name(f'.{self.path.name}.{self.node}.tmp').unlink(missing_ok=True)
            except OSError:
                pass

    # -- 统计（日志/测试用）--------------------------------------------------
    def cached_count(self, task: str) -> int:
        raw = self._data if self._data is not None else self._read(reason='cached_count')
        if raw is None:
            return 0
        return len(((raw.get('tasks') or {}).get(str(task)) or {}).get('losses') or {})


__all__ = ['HardnessSampleCache', 'resolve_model_name', 'NOISE_SEMANTIC_VERSION', 'VERSION']
