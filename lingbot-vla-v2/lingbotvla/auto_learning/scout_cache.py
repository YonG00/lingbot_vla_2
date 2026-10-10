"""Bootstrap-only GMean Scout cache with strict provenance and atomic writes.

Explicitly **not** a Rescan or learned-weight cache. Fingerprint is constructed
from hashes of real weight files and of every file that can move the eval number
(eval code, dataset/normalisation code, model code, norm stats, split manifest,
threshold table, baseline, checkpoint config/tokenizer).

Hashing rules（2026-10-10 起）
-----------------------------
* ``.py`` → **语义 hash**（``semantic_sha256``）：AST 归一化后哈希。注释 / 空行 /
  缩进风格 / docstring 改动**不**改变指纹；逻辑改动（常数、分支、运算符…）**必**改变指纹。
  ``ast.parse`` 失败 ⇒ 回退内容 hash + warning（**绝不静默**）。
* 其它（``.json`` / ``.yaml`` / ``.safetensors`` …）→ 内容 hash（``sha256_file``）。
* 逃生舱：文件里写 ``# scout-fingerprint: no-semantic-hash`` ⇒ 该文件按内容 hash
  （用于「语义没变但必须失效」的极少数场景）。

⚠️ 算法本身的变更会让**所有**旧指纹失效 ⇒ 存量缓存请用
``tools/al_scout_cache_migrate.py`` 过户（见该工具的安全核对）。

``VERSION`` 保持 1：它进 manifest 的是**记录 schema**（``BootstrapScoutCache`` 读盘时
校验），哈希算法变更不需要动它 —— 指纹本身已经把算法变更体现出来了。
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

VERSION = 1
MAX_JSON_BYTES = 1_000_000

# 语义 hash 的逃生舱：文件里出现这一行 ⇒ 该 .py 退回内容 hash
NO_SEMANTIC_HASH_MARKER = "scout-fingerprint: no-semantic-hash"

_log = logging.getLogger(__name__)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def _is_docstring_stmt(node: ast.stmt) -> bool:
    """``Expr(Constant(str))`` —— 精确判断「这一条语句是 docstring」。

    ⚠️ ``Python 3.8+`` 的 ``ast.Str`` 是 ``Constant`` 的别名 ⇒ 只需认 ``Constant``；
    只有**模块 / 类 / 函数体里的第一条**语句算 docstring（调用方保证）。
    """
    return (isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


class _DocstringStripper(ast.NodeTransformer):
    """删除模块/类/函数 docstring（只删每个 body 的第一条 ``Expr(Constant(str))``）。

    保留 Attribute docstring（第二条起的字符串表达式）—— 那是语句，不是 docstring。
    """

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


def semantic_sha256(path: str | Path) -> str:
    """``.py`` 的语义 hash：``ast.parse`` → 去 docstring → ``ast.dump`` → sha256。

    * 注释 / 空行 / 缩进风格 ⇒ **不变**（``ast`` 根本不保留它们；
      ``include_attributes=False`` 保证行号列号也不进哈希）
    * docstring（模块/类/函数体的第一条字符串表达式）**整体去掉** ⇒ 加 / 删 / 改
      docstring 文字都**不**改变指纹（精确语义见 ``tests/test_al_scout_cache_semantic_fp.py``；
      想让 docstring 参与哈希请用逃生舱 marker）
    * 逻辑改动（改常数、加删分支、换运算符、改**非 docstring** 的字符串字面量）⇒ **必变**
    * ``SyntaxError`` 等解析失败 ⇒ **回退内容 hash** 并 warning（绝不静默）
    * 文件含 ``NO_SEMANTIC_HASH_MARKER`` ⇒ 主动退回内容 hash（warning 级日志）
    """
    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        _log.warning('[scout_cache] %s 不是 UTF-8（%s）⇒ 回退内容 hash', path, exc)
        return hashlib.sha256(raw).hexdigest()
    if NO_SEMANTIC_HASH_MARKER in text:
        _log.warning('[scout_cache] %s 含 %s ⇒ 按内容 hash（语义 hash 逃生舱）',
                     path, NO_SEMANTIC_HASH_MARKER)
        return hashlib.sha256(raw).hexdigest()
    try:
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, ValueError) as exc:
        _log.warning('[scout_cache] ⚠️ %s ast.parse 失败（%s: %s）⇒ 回退内容 hash',
                     path, type(exc).__name__, exc)
        return hashlib.sha256(raw).hexdigest()
    tree = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    canonical = ast.dump(tree, include_attributes=False, annotate_fields=True)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def source_sha256(path: str | Path) -> str:
    """按扩展名分派：``.py`` → 语义 hash；其它 → 内容 hash。"""
    path = Path(path)
    if path.suffix == '.py':
        return semantic_sha256(path)
    return sha256_file(path)


def normalize_dtype(value: Any) -> str:
    """把 dtype 写法归一化（``bf16`` / ``bfloat16`` / ``torch.bfloat16`` → ``bfloat16``）。

    🔴 必要性：``tools/scan_accel_preflight.py --dtype bfloat16`` 与
    ``AL_SCOUT_CACHE_DTYPE=bf16`` 指的是**同一个**推理精度，若原样进 options
    就会算出两个不同指纹 ⇒ 预检认证过的缓存运行时读不到（静默重扫）。
    """
    text = str(value).strip().lower()
    if text.startswith('torch.'):
        text = text[len('torch.'):]
    return {'bf16': 'bfloat16', 'fp16': 'float16', 'half': 'float16',
            'fp32': 'float32', 'float': 'float32', 'double': 'float64'}.get(text, text)


def provenance(*, weight_files: Sequence[str | Path], sources: Mapping[str, str | Path],
               options: Mapping[str, Any]) -> str:
    """Hash actual checkpoint shards (not only names, mtimes or 'step500')."""
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


# ---------------------------------------------------------------------------
# 评测链路的 source 清单（唯一事实来源）
# ---------------------------------------------------------------------------
# 清单原则：**宁可多列，绝不少列** —— 一个能改变评测数字的文件漏了，缓存就会
# 拿着过期的扫描结果判 PASS/FAIL（比"多失效一次"危险得多）。每一项都必须在报告里
# 给出「它在评测路径上」的证据（调用链 / grep）。
#
# 相对仓库根 `REPO_ROOT` 的路径；`required=True` 的缺失 ⇒ 直接报错（缺了它评测链
# 本身就跑不起来）；`required=False` ⇒ 跳过 + warning（真机/精简检出上可能不存在）。
EVAL_SOURCES: Tuple[Tuple[str, str, bool], ...] = (
    # ---- 评测执行本体 ----
    ('eval', 'lingbotvla/utils/open_loop_validation.py', True),
    # ---- 模型 ----
    ('model', 'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py', True),
    # ---- 数据集构造链（决定「取到哪些帧/回合」与「喂进模型的张量」）----
    ('transform', 'lingbotvla/data/vla_data/transform.py', True),
    ('dataset_builder', 'lingbotvla/data/dataset.py', True),
    ('multi_vla_dataset', 'lingbotvla/data/vla_data/multi_vla_dataset.py', True),
    ('base_dataset', 'lingbotvla/data/vla_data/base_dataset.py', True),
    ('data_utils', 'lingbotvla/data/vla_data/utils.py', True),
    ('ee_pose_transform', 'lingbotvla/data/vla_data/ee_pose_transform.py', False),
    ('video_utils', 'lingbotvla/data/vla_data/video_utils.py', False),
    # ---- 精度 / 指标 ----
    ('eval_precision', 'lingbotvla/utils/eval_precision.py', True),
    ('evaluator', 'lingbotvla/auto_learning/evaluator.py', True),
    ('gmean', 'lingbotvla/auto_learning/decision/gmean.py', True),
    # 诊断/加速路径用（serial 生产路径只 import 它做形状守卫）；列上以防万一
    ('scan_accel', 'lingbotvla/auto_learning/scan_accel.py', True),
    ('eval_batch_policy', 'lingbotvla/auto_learning/eval_batch_policy.py', True),
)


def eval_source_paths(repo_root: str | Path) -> Dict[str, Path]:
    """``EVAL_SOURCES`` → ``{key: 绝对路径}``（**不做**存在性检查）。"""
    root = Path(repo_root)
    return {key: root / rel for key, rel, _ in EVAL_SOURCES}


def source_manifest(
    repo_root: str | Path,
    extra_sources: Mapping[str, str | Path] | None = None,
    *,
    strict: bool = False,
    on_missing: Any = None,
) -> Tuple[Dict[str, Path], List[Tuple[str, str]]]:
    """构造完整的评测 source 清单：``EVAL_SOURCES`` + ``extra_sources``（权重/数据文件）。

    返回 ``(sources, missing)``；``missing`` 是 ``(key, reason)`` 列表。

    * ``strict=True`` ⇒ 任何文件缺失都 ``FileNotFoundError``（**评测链缺文件就不该算指纹**，
      否则会静默算出一个"看起来合法"的指纹）。
    * ``strict=False`` ⇒ 缺失项**跳过**并把原因交给 ``on_missing(key, reason)``
      （build.py 用 ``logger.warning``；工具用 print）—— 绝不抛 ``KeyError``。
    """
    sources: Dict[str, Path] = {}
    missing: List[Tuple[str, str]] = []
    for key, rel, required in EVAL_SOURCES:
        path = Path(repo_root) / rel
        if path.is_file():
            sources[key] = path
        else:
            missing.append((key, f'{rel} 不存在（required={required}）'))
    for key, value in dict(extra_sources or {}).items():
        path = Path(value).expanduser()
        if path.is_file():
            sources[key] = path
        else:
            missing.append((key, f'{path} 不存在'))
        if key in {k for k, _, _ in EVAL_SOURCES}:
            raise ValueError(f'extra_sources 的 key 与评测链清单冲突: {key}')
    if missing:
        if strict:
            raise FileNotFoundError(
                'Scout cache 的评测 source 缺失 ⇒ 拒绝计算指纹（缺失文件会让指纹"看起来合法"）：\n  - '
                + '\n  - '.join(f'{k}: {r}' for k, r in missing))
        if on_missing is not None:
            for key, reason in missing:
                on_missing(key, reason)
    return sources, missing


def scout_key(task: str, episode_ids: Sequence[int]) -> str:
    ids = list(map(int, episode_ids))
    if not task or not ids or len(set(ids)) != len(ids):
        raise ValueError('invalid task/episode ID set')
    return hashlib.sha256(json.dumps([task, ids], separators=(',', ':')).encode()).hexdigest()


def _is_rank0() -> bool:
    """本进程是否 rank0（无 dist / 未初始化 ⇒ 视为 rank0）。"""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()) == 0
    except Exception:  # noqa: BLE001
        pass
    return True



class BootstrapScoutCache:
    def __init__(self, root: str | Path, *, fingerprint: str,
                 write_enabled: bool | None = None):
        if len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
            raise ValueError('invalid provenance fingerprint')
        self.path = Path(root) / fingerprint
        self.fingerprint = fingerprint
        # 🔴 多卡：**只有 rank0 写缓存**。所有 rank 都跑调度器，若都写同一文件，
        #    会有并发写（撕裂）与重复 I/O；读不受影响（各 rank 都可 load）。
        if write_enabled is None:
            write_enabled = _is_rank0()
        self.write_enabled = bool(write_enabled)

    def load(self, task: str, ids: Sequence[int]) -> dict | None:
        key = scout_key(task, ids)
        p = self.path / f'{key}.json'
        if not p.is_file() or p.stat().st_size > MAX_JSON_BYTES:
            return None
        try:
            raw = json.loads(p.read_text(encoding='utf-8'))
            if (raw.get('version') != VERSION or raw.get('fingerprint') != self.fingerprint
                    or raw.get('task') != task or raw.get('episode_ids') != list(map(int, ids))):
                return None
            m = raw['metrics']
            if not isinstance(m, dict) or m.get('task') != task or m.get('episode_ids') != list(map(int, ids)):
                return None
            per = m.get('per_traj_mse', {})
            if not isinstance(per, dict) or set(per) != set(map(str, ids)):
                return None
            vals = [float(v) for v in per.values()]
            if any(not math.isfinite(v) or v < 0 for v in vals):
                return None
            nmse, mse, base = (float(m[k]) for k in ('nmse','mse','baseline_mse'))
            if (m.get('n_trajs') != len(ids) or m.get('metric_valid') is not True
                    or not all(math.isfinite(v) for v in (nmse,mse,base))
                    or min(nmse,mse) < 0 or base <= 0
                    or not math.isclose(nmse, mse/base, rel_tol=1e-5, abs_tol=1e-8)
                    or not math.isclose(mse, sum(vals)/len(vals), rel_tol=1e-4, abs_tol=1e-7)):
                return None
            return m
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def store(self, task: str, ids: Sequence[int], metrics: Mapping[str, Any]) -> None:
        if not self.write_enabled:
            return                      # 非 rank0：只读不写（多卡下避免并发写同一文件）
        key = scout_key(task, ids)
        if (metrics.get('task') != task or list(metrics.get('episode_ids', [])) != list(map(int, ids))):
            raise ValueError('cannot cache mismatched task or IDs')
        payload = {'version': VERSION, 'fingerprint': self.fingerprint, 'task': task,
                   'episode_ids': list(map(int, ids)), 'metrics': dict(metrics)}
        blob = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(',', ':'))
        if len(blob.encode()) > MAX_JSON_BYTES:
            raise ValueError('scout cache record too large')
        self.path.mkdir(parents=True, exist_ok=True)
        target = self.path / f'{key}.json'
        tmp = self.path / f'.{key}.{os.getpid()}.tmp'
        try:
            with open(tmp, 'x', encoding='utf-8') as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
