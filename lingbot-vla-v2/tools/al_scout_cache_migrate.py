#!/usr/bin/env python3
"""把「旧算法指纹」下的存量 Bootstrap Scout 缓存过户到「新算法指纹」目录名下。

背景
----
``lingbotvla/auto_learning/scout_cache.py`` 的指纹算法在 2026-10-10 变了两次：

1. **source 清单补全**（新增数据集构造链 + AL 配置文件本身）；
2. **``.py`` 改用 AST 语义 hash**（注释/空行/docstring 改动不再让指纹失效）。

⇒ 真机上那个「50 个 JSON、刚扫了 ~5 分钟」的旧目录 ``$AL_SCOUT_CACHE_ROOT/<old_fp>/``
在新代码下读不到（缓存的正确性判据是严格相等），但**内容其实完全有效**。
本工具把它过户成 ``<new_fp>/``：用**硬链接**（同 inode，零拷贝、零额外磁盘），
**保留旧目录**当存档。

安全口径（宁可拒绝，不可猜）
--------------------------
过户的前提是「**旧扫描之后，没有任何一个进入指纹的依赖文件被改过**」。
工具把新旧算法依赖的**全部文件**（含权重分片、manifest/norm/thresholds/baseline、
checkpoint 的 config.json/tokenizer.json、以及全部评测链 .py）连同 **mtime** 列出来，
并与旧缓存目录的创建时间（目录 mtime）逐项比较：

* 任何依赖文件 ``mtime >= 旧缓存目录 mtime`` ⇒ **拒绝迁移**（并打印该文件清单）。
  用 ``>=`` 而不是 ``>``：文件与缓存同一秒内写下时无法区分先后，取保守的一侧。
* 依赖文件缺失 ⇒ **拒绝迁移**（文件被删/搬走同样是"变了"）。
* 新指纹目录已存在且非空 ⇒ **拒绝迁移**（不覆盖任何东西）。
* 目录里没有 ``*.json`` 记录 / 找不到旧目录 ⇒ **拒绝迁移**。
* 想覆盖 mtime 判定必须**显式** ``--allow-unchanged``（日志里会留下这条决定）。

用法（默认 ``--dry-run``，只打印不改盘）
--------------------------------------
    # 1) 先干跑：只看它与新指纹、依赖 mtime 表、以及是否会拒绝
    python tools/al_scout_cache_migrate.py --dry-run \\
        --cache-root /data/outputs/al_scout_cache \\
        --checkpoint /data/models/ckpt500 \\
        --manifest  .../task_split.json --norm .../norm_stats.json \\
        --thresholds .../pass_thresholds.json --baseline .../task_baseline.json \\
        --config configs/auto_learning/experiment_50task_gmean200.yaml --dtype bf16

    # 2) 人工核对 mtime 表 → 明确确认"这些文件在旧扫描之后没被改过" → 过户
    python tools/al_scout_cache_migrate.py --apply --yes ...（同一组参数）

依赖的**唯一事实来源**：``lingbotvla.auto_learning.scout_cache`` 的
``EVAL_SOURCES`` / ``source_manifest`` / ``provenance`` —— 与 ``real/build.py``
运行时算指纹走的是**同一段代码**（否则过户过去也读不到，等于白干）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lingbotvla.auto_learning.scout_cache import (  # noqa: E402
    EVAL_SOURCES, provenance, source_manifest,
)

# 供报告使用；实际取值以 scout_cache 为准（避免两处漂移）
try:  # pragma: no cover - 仅用于 argparse 帮助文本
    from lingbotvla.auto_learning.scout_cache import VERSION as _SCHEMA
except Exception:  # noqa: BLE001
    _SCHEMA = '?'

REJECT = 'REJECT'
APPLY_READY = 'APPLY_READY'
DRY_RUN_READY = 'DRY_RUN_READY'


def _fmt_ts(ns: Optional[int]) -> str:
    if ns is None:
        return 'n/a'
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ns / 1e9)) + f'.{int(ns % 1_000_000_000):09d}'


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value else default


def _resolve_path(value: Optional[str]) -> Optional[Path]:
    return Path(value).expanduser().resolve() if value else None


def load_thresholds_path(config_path: Optional[str]) -> Optional[str]:
    """从 AL 配置 yaml 里取 ``pass_thresholds_file``（与 build.py 的 cfg 同源）。"""
    if not config_path:
        return None
    import yaml  # 局部 import：本工具其余部分不需要 yaml

    with open(config_path, encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    body = raw.get('auto_learning', raw)
    value = body.get('pass_thresholds_file') if isinstance(body, dict) else None
    return str(value) if value else None


def load_scout_trajs(config_path: Optional[str]) -> Optional[int]:
    if not config_path:
        return None
    import yaml

    with open(config_path, encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    body = raw.get('auto_learning', raw)
    value = body.get('global_scout_val_trajs') if isinstance(body, dict) else None
    return int(value) if value is not None else None


def required_paths(args) -> Dict[str, Optional[Path]]:
    """运行时**必然**进入指纹的路径（缺任何一个 ⇒ 算出的"新指纹"都不是真的那个）。"""
    paths: Dict[str, Optional[Path]] = {
        'manifest': args.manifest, 'norm': args.norm,
        'thresholds': args.thresholds, 'baseline': args.baseline,
    }
    if args.checkpoint:
        paths['checkpoint_config'] = Path(args.checkpoint) / 'config.json'
        paths['checkpoint_tokenizer'] = Path(args.checkpoint) / 'tokenizer.json'
    return paths


def missing_required(args) -> List[str]:
    return [key for key, path in required_paths(args).items()
            if path is None or not Path(path).is_file()]


def build_sources(args, warnings: List[str]) -> Tuple[Dict[str, Path], List[Tuple[str, str]]]:
    extra: Dict[str, Any] = {
        'manifest': args.manifest, 'norm': args.norm,
        'thresholds': args.thresholds, 'baseline': args.baseline,
    }
    if args.checkpoint:
        extra['checkpoint_config'] = Path(args.checkpoint) / 'config.json'
        extra['checkpoint_tokenizer'] = Path(args.checkpoint) / 'tokenizer.json'
    extra = {k: v for k, v in extra.items() if v is not None}
    sources, missing = source_manifest(
        ROOT, extra, strict=False,
        on_missing=lambda key, reason: warnings.append(f'source 缺失: {key}（{reason}）'))
    return sources, missing


def compute_new_fingerprint(args, warnings: List[str]) -> Tuple[Optional[str], Dict[str, Path]]:
    sources, _missing = build_sources(args, warnings)
    if not args.checkpoint:
        warnings.append('未提供 --checkpoint（无法算权重/checkpoint 配置的哈希）⇒ 无法计算新指纹')
        return None, sources
    shards = sorted(Path(args.checkpoint).glob('*.safetensors'))
    if not shards:
        warnings.append(f'{args.checkpoint} 下没有 *.safetensors ⇒ 无法计算新指纹')
        return None, sources
    options = {
        'inference_dtype': args.dtype,
        'noise_seed': int(args.noise_seed),
        'scout_trajs': int(args.scout_trajs),
        'stride': 'per_episode',
        'image_augment': False,
    }
    try:
        return provenance(weight_files=shards, sources=sources, options=options), sources
    except (OSError, ValueError) as exc:
        warnings.append(f'provenance 计算失败: {type(exc).__name__}: {exc}')
        return None, sources


def list_cache_candidates(cache_root: Path, exclude: Sequence[str] = ()) -> List[Tuple[Path, int]]:
    """``$CACHE_ROOT/<64hex>/`` 里含 ``*.json`` 的目录 ⇒ ``[(路径, json 数)]``，按 mtime 新→旧。"""
    out: List[Tuple[Path, int]] = []
    if not cache_root.is_dir():
        return out
    for child in sorted(cache_root.iterdir()):
        if not child.is_dir() or child.name in exclude:
            continue
        n = len(list(child.glob('*.json')))
        if n:
            out.append((child, n))
    out.sort(key=lambda item: item[0].stat().st_mtime_ns, reverse=True)
    return out


def pick_old_dir(args, new_fp: Optional[str], warnings: List[str]) -> Tuple[Optional[Path], str]:
    cache_root = Path(args.cache_root)
    if args.old_fingerprint:
        cand = cache_root / args.old_fingerprint
        return (cand if cand.is_dir() else None), ('--old-fingerprint 指定' if cand.is_dir()
                                                   else f'--old-fingerprint={args.old_fingerprint} 目录不存在')
    exclude = [new_fp] if new_fp else []
    found = list_cache_candidates(cache_root, exclude=exclude)
    if not found:
        return None, f'{cache_root} 下没有任何含 *.json 的指纹目录'
    if len(found) > 1:
        # 不猜：多个候选必须人工指定（猜错就把别的 run 的扫描结果过户过来了）
        listing = ', '.join(f'{p.name}({n} 个 json, mtime={_fmt_ts(p.stat().st_mtime_ns)})'
                            for p, n in found)
        return None, (f'{cache_root} 下有 {len(found)} 个候选旧目录 ⇒ 请用 --old-fingerprint 明确指定：'
                      f'{listing}')
    return found[0][0], f'自动选中唯一候选（{found[0][1]} 个 json）'


def inspect_dependencies(sources: Dict[str, Path], weight_files: Sequence[Path],
                         old_dir: Path, tolerance_ns: int) -> List[Dict[str, Any]]:
    """逐项列出依赖文件的 mtime，并判定它是否"比旧缓存目录新"。"""
    cutoff = old_dir.stat().st_mtime_ns - int(tolerance_ns)
    rows: List[Dict[str, Any]] = []
    for key, path in sorted(sources.items()):
        rows.append(_dep_row(key, path, cutoff))
    for path in sorted(weight_files, key=str):
        rows.append(_dep_row(f'weights:{path.name}', path, cutoff))
    return rows


def _dep_row(key: str, path: Path, cutoff_ns: int) -> Dict[str, Any]:
    try:
        st = path.stat()
    except OSError as exc:
        return {'key': key, 'path': str(path), 'mtime': None, 'mtime_ns': None,
                'exists': False, 'newer': True, 'problem': f'缺失/不可读: {exc}'}
    newer = st.st_mtime_ns >= cutoff_ns
    return {'key': key, 'path': str(path), 'mtime': _fmt_ts(st.st_mtime_ns),
            'mtime_ns': st.st_mtime_ns, 'exists': True, 'newer': bool(newer),
            'problem': ('依赖文件 mtime 不早于旧缓存目录 ⇒ 无法证明"旧扫描之后没被改过"'
                        if newer else None)}


def _link_or_copy(src: Path, dst: Path) -> str:
    try:
        os.link(src, dst)
        return 'hardlink'
    except OSError:
        shutil.copy2(src, dst)
        return 'copy'


def migrate(old_dir: Path, new_dir: Path) -> Dict[str, Any]:
    """把 ``old_dir`` 的内容过户到 ``new_dir``（硬链接优先，跨设备回退 copy2）。"""
    new_dir.mkdir(parents=True, exist_ok=True)
    linked = copied = 0
    for src in sorted(old_dir.iterdir()):
        if not src.is_file():
            continue
        dst = new_dir / src.name
        if dst.exists():
            continue
        method = _link_or_copy(src, dst)
        if method == 'hardlink':
            linked += 1
        else:
            copied += 1
    old_files = [p for p in old_dir.iterdir() if p.is_file()]
    new_files = [p for p in new_dir.iterdir() if p.is_file()]
    return {
        'old_dir': str(old_dir), 'new_dir': str(new_dir),
        'old_files': len(old_files), 'new_files': len(new_files),
        'old_json': len(list(old_dir.glob('*.json'))), 'new_json': len(list(new_dir.glob('*.json'))),
        'hardlinked': linked, 'copied': copied,
        'old_dir_kept': old_dir.is_dir(),
    }


def run(args) -> Tuple[int, Dict[str, Any]]:
    warnings: List[str] = []
    report: Dict[str, Any] = {
        'mode': 'apply' if args.apply else 'dry-run',
        'schema': _SCHEMA,
        'eval_source_entries': len(EVAL_SOURCES),
        'dtype': args.dtype, 'noise_seed': int(args.noise_seed), 'scout_trajs': int(args.scout_trajs),
    }
    if args.cache_root is None:
        report.update(status=REJECT, reason='缺少缓存根目录（--cache-root 或 $AL_SCOUT_CACHE_ROOT）')
        report['warnings'] = warnings
        return 2, report
    cache_root = Path(args.cache_root).expanduser()
    report['cache_root'] = str(cache_root)

    # 🔴 先查依赖完整性：缺文件时算出的"新指纹"与运行时算的**不是同一个**
    #    （source_manifest 会跳过缺失项）⇒ 过户过去也读不到，必须先拒绝。
    absent = missing_required(args)
    if absent:
        report.update(status=REJECT,
                      reason=('这些必填依赖不存在 ⇒ 无法算出与运行时一致的指纹，拒绝迁移: '
                              + ', '.join(absent)
                              + '（请补 --manifest/--norm/--thresholds/--baseline/--config 或修好文件路径）'))
        report['missing_required'] = absent
        report['warnings'] = warnings
        return 2, report

    new_fp, sources = compute_new_fingerprint(args, warnings)
    report['new_fingerprint'] = new_fp
    report['n_sources'] = len(sources)
    if new_fp is None:
        report.update(status=REJECT, reason='无法计算新指纹（见 warnings）')
        report['warnings'] = warnings
        return 2, report
    new_dir = cache_root / new_fp
    report['new_dir'] = str(new_dir)

    old_dir, why = pick_old_dir(args, new_fp, warnings)
    report['old_dir_reason'] = why
    if old_dir is None:
        report.update(status=REJECT, reason=f'找不到旧缓存目录：{why}')
        report['warnings'] = warnings
        return 2, report
    report['old_dir'] = str(old_dir)
    old_json = sorted(old_dir.glob('*.json'))
    report['old_json_count'] = len(old_json)
    report['old_dir_mtime'] = _fmt_ts(old_dir.stat().st_mtime_ns)
    if len(old_json) < args.min_files:
        report.update(status=REJECT,
                      reason=f'旧目录只有 {len(old_json)} 个 json < --min-files={args.min_files}'
                             '（可能不是那份完整扫描结果）')
        report['warnings'] = warnings
        return 2, report
    if new_dir.exists() and any(new_dir.iterdir()):
        report.update(status=REJECT,
                      reason=f'目标目录已存在且非空：{new_dir}（本工具绝不覆盖已有缓存）')
        report['warnings'] = warnings
        return 2, report
    if old_dir.resolve() == new_dir.resolve():
        report.update(status=REJECT, reason='新旧指纹目录是同一个 ⇒ 无事可做')
        report['warnings'] = warnings
        return 2, report

    shards = sorted(Path(args.checkpoint).glob('*.safetensors')) if args.checkpoint else []
    rows = inspect_dependencies(sources, shards, old_dir, args.tolerance_s * 1e9)
    report['dependencies'] = rows
    bad = [r for r in rows if r['newer']]
    report['dependencies_newer'] = [r['key'] for r in bad]
    if bad and not args.allow_unchanged:
        report.update(status=REJECT,
                      reason=(f'{len(bad)} 个依赖文件不早于旧缓存目录 ⇒ 拒绝过户。'
                              '请逐个确认它们在旧扫描之后**没有被修改**（看下面 dependencies 的 mtime），'
                              '确认无误后再加 --allow-unchanged 重跑。'))
        report['warnings'] = warnings
        return 2, report
    if bad:
        warnings.append(f'⚠️ --allow-unchanged：{len(bad)} 个依赖文件比旧缓存新，'
                        '已由人工显式承担"旧扫描结果仍然有效"的责任')

    report['dependencies_unchecked'] = not bool(bad)
    if not args.apply:
        report.update(status=DRY_RUN_READY,
                      reason='dry-run：未改动文件系统。核对 dependencies 的 mtime 后，用 --apply 过户。')
        report['warnings'] = warnings
        return 0, report

    result = migrate(old_dir, new_dir)
    report['result'] = result
    ok = (result['old_json'] == result['new_json'] and result['new_files'] >= result['old_files']
          and result['old_dir_kept'])
    report.update(status=APPLY_READY if ok else REJECT,
                  reason=('过户完成（旧目录保留为存档）' if ok else '过户后两侧文件数不一致 ⇒ 请人工检查'))
    report['warnings'] = warnings
    return (0 if ok else 2), report


def print_report(report: Dict[str, Any]) -> None:
    print('=' * 78)
    print(f"AL Scout 缓存过户 | 模式={report.get('mode')} | 状态={report.get('status')}")
    print('=' * 78)
    print(f"缓存根目录     : {report.get('cache_root')}")
    print(f"旧指纹目录     : {report.get('old_dir')}  ({report.get('old_json_count')} 个 json, "
          f"mtime={report.get('old_dir_mtime')})")
    print(f"  └ 选择依据   : {report.get('old_dir_reason')}")
    print(f"新指纹         : {report.get('new_fingerprint')}")
    print(f"新指纹目录     : {report.get('new_dir')}")
    print(f"指纹选项       : dtype={report.get('dtype')} seed={report.get('noise_seed')} "
          f"scout_trajs={report.get('scout_trajs')}")
    print(f"评测链 source  : {report.get('eval_source_entries')} 个（清单定义在 scout_cache.EVAL_SOURCES）")
    print('-' * 78)
    deps = report.get('dependencies') or []
    if deps:
        print(f"依赖文件 mtime 表（{len(deps)} 项）—— 请人工确认「旧扫描之后没有被修改过」：")
        for row in deps:
            flag = '✗ 比缓存新' if row['newer'] else '✓ 早于缓存'
            print(f"  [{flag}] {row['mtime']}  {row['key']:<28} {row['path']}")
            if row.get('problem') and not row['newer']:
                print(f"            ⚠️ {row['problem']}")
    if report.get('dependencies_newer'):
        print(f"🔴 比旧缓存新的依赖: {report['dependencies_newer']}")
    print('-' * 78)
    if report.get('result'):
        r = report['result']
        print(f"过户结果: 旧目录 {r['old_files']} 个文件（{r['old_json']} json，**保留**） → "
              f"新目录 {r['new_files']} 个文件（{r['new_json']} json）；"
              f"硬链接 {r['hardlinked']} / 复制 {r['copied']}")
    print(f"结论: {report.get('reason')}")
    for w in report.get('warnings') or []:
        print(f"  ⚠️ {w}")
    print('=' * 78)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', default=True,
                      help='只核对并打印，不改动文件系统（默认）')
    mode.add_argument('--apply', action='store_true',
                      help='真正过户（硬链接优先；旧目录保留）')
    ap.add_argument('--cache-root', default=_env('AL_SCOUT_CACHE_ROOT'),
                    help='缓存根（默认 $AL_SCOUT_CACHE_ROOT）')
    ap.add_argument('--checkpoint', default=_env('AL_SCOUT_CACHE_CHECKPOINT'),
                    help='HF checkpoint 目录（默认 $AL_SCOUT_CACHE_CHECKPOINT）')
    ap.add_argument('--manifest', default=_env('AL_SCOUT_CACHE_MANIFEST'), help='task split manifest')
    ap.add_argument('--norm', default=_env('AL_SCOUT_CACHE_NORM'), help='norm stats json')
    ap.add_argument('--baseline', default=_env('AL_SCOUT_CACHE_BASELINE'), help='task baseline json')
    ap.add_argument('--thresholds', default=_env('AL_SCOUT_CACHE_THRESHOLDS'),
                    help='pass thresholds json（缺省时从 --config 的 pass_thresholds_file 取）')
    ap.add_argument('--config', default=_env('AL_AUTO_LEARNING_CONFIG') or _env('AL_SCOUT_CACHE_AL_CFG'),
                    help='AL 配置 yaml（即 --train.auto_learning 指向的文件）')
    ap.add_argument('--dtype', default=_env('AL_SCOUT_CACHE_DTYPE', 'bfloat16'),
                    help='推理精度（默认 $AL_SCOUT_CACHE_DTYPE 或 bfloat16）')
    ap.add_argument('--noise-seed', type=int, default=1234)
    ap.add_argument('--scout-trajs', type=int, default=None,
                    help='cfg.global_scout_val_trajs（缺省时从 --config 取，再缺省 2）')
    ap.add_argument('--old-fingerprint', default=None,
                    help='明确指定旧指纹目录名（默认自动探测）')
    ap.add_argument('--min-files', type=int, default=1,
                    help='旧目录至少要有多少个 .json 才认（默认 1；真机 50-task 扫描可设 50）')
    ap.add_argument('--tolerance-s', type=float, default=0.0,
                    help='mtime 比对容差秒数（默认 0 = 最严：>= 旧目录 mtime 即拒绝）')
    ap.add_argument('--allow-unchanged', action='store_true',
                    help='🔴 人工显式承担"依赖文件没变"的责任（mtime 判定不算数）')
    ap.add_argument('--yes', action='store_true',
                    help='--apply 时跳过交互式确认（CI/脚本用；日志仍会记录）')
    ap.add_argument('--json-report', action='store_true', help='额外打印机器可读 JSON')
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tolerance_s < 0:
        raise SystemExit('--tolerance-s 不能为负')
    if args.min_files < 1:
        raise SystemExit('--min-files 至少为 1（空目录不算"存量缓存"）')
    if args.scout_trajs is None:
        args.scout_trajs = load_scout_trajs(args.config) or 2
    if args.thresholds is None:
        args.thresholds = load_thresholds_path(args.config)
    if args.thresholds is None and args.config:
        print(f'⚠️ {args.config} 里没有 pass_thresholds_file，且未给 --thresholds '
              '⇒ 指纹会与运行时的不同（阈值表是评测判据的一部分）', file=sys.stderr)
    for name in ('cache_root', 'checkpoint', 'manifest', 'norm', 'baseline', 'thresholds', 'config'):
        value = getattr(args, name)
        if value:
            setattr(args, name, _resolve_path(value))
    if args.apply and not args.yes:
        print('即将把旧缓存目录**硬链接**过户到新指纹目录（旧目录保留）。')
        try:
            reply = input('确认这些依赖文件在旧扫描之后没有被修改过？输入 yes 继续: ').strip().lower()
        except EOFError:
            reply = ''
        if reply not in ('y', 'yes'):
            print('已取消（未改动任何文件）。')
            return 3
    code, report = run(args)
    print_report(report)
    if args.json_report:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    sys.exit(main())
