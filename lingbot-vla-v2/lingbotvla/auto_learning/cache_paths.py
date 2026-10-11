"""缓存文件的**按模型分名**工具（2026-10-11 用户要求）。

问题
----
scout / hardness 缓存是**固定文件名 + 单 `model` 槽位**：
    /workspace/al/scout_cache/scout.json
换模型时 `store()` 会 `_blank()` 重建并 `os.replace()` **覆盖同一个文件**
⇒ 上一个模型的缓存（可能是几十分钟的扫描结果）被静默丢掉，且没有任何自动存档。

做法（用户选定）
----------------
把**模型标识编进文件名**：
    scout.json  ->  scout.<model-slug>.json
    model-slug = 模型标识，`/` 等非法字符替换为 `_`
                 （例如 `train_al_v37/checkpoints/global_step_500/hf_ckpt`
                  -> `train_al_v37_checkpoints_global_step_500_hf_ckpt`）
过长时截断并附 8 位 hash 防撞；slug 为空或 `AL_CACHE_SHARED=1` 时**保持原文件名**
（逃生开关，用于排障与向后兼容）。

⇒ 每个模型一份文件：互不覆盖、旧文件自然留档、不需要人工改名。

一次性迁移
----------
scoped 文件不存在、而**旧的无后缀文件存在**时，把旧文件**复制**过去（旧文件保留不动）。
这样既拿到"按模型分文件"的语义，又不会让已有缓存作废（只在允许写盘的 rank 上做）。
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

#: 逃生开关：`=1` ⇒ 回到"所有模型共用一个固定文件名"的旧行为（排障用）。
SHARED_ENV = 'AL_CACHE_SHARED'

#: model-slug 的最大长度（超出则截断 + 附 hash）。
MAX_SLUG_LEN = 80

#: 文件名里需要替换掉的字符（路径分隔符与常见控制字符）。
_REPLACE = {'/': '_', '\\': '_', ':': '_', '*': '_', '?': '_',
            '"': '_', '<': '_', '>': '_', '|': '_', ' ': '_'}


def cache_sharing_disabled() -> bool:
    """是否显式要求"共享一个缓存文件"（旧行为）。"""
    return str(os.environ.get(SHARED_ENV, '')).strip() in ('1', 'true', 'True', 'yes')


def model_slug(model: str) -> str:
    """把模型标识转成可安全放进**文件名**的 slug；无法转换时返回空串。"""
    raw = str(model or '').strip()
    if not raw:
        return ''
    slug = ''.join(_REPLACE.get(ch, ch) for ch in raw)
    # 去掉可能残留的路径穿越与首尾分隔符
    slug = slug.strip('._-')
    if not slug:
        return ''
    if len(slug) > MAX_SLUG_LEN:
        digest = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]
        slug = f'{slug[:MAX_SLUG_LEN - 9]}_{digest}'
    return slug


def scoped_cache_path(base: str | Path, model: str) -> Path:
    """返回该模型应使用的缓存文件路径（**纯函数，不碰磁盘**）。

    * `model` 为空 / 无法生成 slug / 打开 `AL_CACHE_SHARED=1` ⇒ 原样返回 `base`；
    * 若 `base` 的文件名**已经**以该 slug 结尾 ⇒ 原样返回（幂等，避免二次拼接）。
    """
    path = Path(str(base)).expanduser()
    if cache_sharing_disabled():
        return path
    slug = model_slug(model)
    if not slug:
        return path
    stem, suffix = path.stem, path.suffix or '.json'
    if stem.endswith(f'.{slug}') or stem == slug:
        return path                       # 幂等：已按该模型分名
    return path.with_name(f'{stem}.{slug}{suffix}')


def ensure_scoped_cache_file(base: str | Path, model: str, *,
                             write_enabled: bool = True) -> Path:
    """返回 scoped 路径，并在需要时把**旧的同名无后缀文件**复制过去（只复制，不删除）。

    `write_enabled=False`（非 rank0）时**不做迁移**，只返回路径；
    这样即使本 rank 还没看到旧文件，也不会误建空文件把别的 rank 带偏。
    """
    scoped = scoped_cache_path(base, model)
    src = Path(str(base)).expanduser()
    if scoped == src or not write_enabled:
        return scoped
    try:
        if not scoped.is_file() and src.is_file():
            scoped.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, scoped)
    except OSError:
        # 迁移失败不影响正确性：scoped 文件会按"未命中 ⇒ 重扫"正常走
        pass
    return scoped


__all__ = ['SHARED_ENV', 'MAX_SLUG_LEN', 'cache_sharing_disabled', 'model_slug',
           'scoped_cache_path', 'ensure_scoped_cache_file']
