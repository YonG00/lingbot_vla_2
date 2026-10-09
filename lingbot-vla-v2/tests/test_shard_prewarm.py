"""并行分片预读的 CPU 测试（AST 编译：`module_utils` 本地无法导入 —— transformers 版本问题）。

为什么要测：预读是"加速手段"，**失败绝不能挡住加载**；同时它的开关与线程数必须可控。
"""
from __future__ import annotations

import ast
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "lingbotvla/models/module_utils.py"

NAMES = {
    "SAFE_WEIGHTS_NAME": "model.safetensors",
    "SAFE_WEIGHTS_INDEX_NAME": "model.safetensors.index.json",
    "DIFFUSERS_SAFETENSORS_WEIGHTS_NAME": "diffusion_pytorch_model.safetensors",
    "DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME": "diffusion_pytorch_model.safetensors.index.json",
    "WEIGHTS_NAME": "pytorch_model.bin",
    "WEIGHTS_INDEX_NAME": "pytorch_model.bin.index.json",
}


def _load_functions(*, index_shards=None):
    """把 `_resolve_weight_files` / `_parallel_prewarm_shards` / `_log_info` 按源码编译。"""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    wanted = ("_resolve_weight_files", "_parallel_prewarm_shards", "_log_info")
    fns = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in wanted]
    assert len(fns) == len(wanted), [f.name for f in fns]
    ast.fix_missing_locations(tree)
    ns = {"os": os, "time": time, "ThreadPoolExecutor": ThreadPoolExecutor,
          "Optional": object, "Dict": dict, "Any": object, "List": list,
          # 模块级常量（AST 只编译了函数体，这里补上它们引用的名字）
          "SHARD_PREWARM_ENV": "AL_SHARD_PREWARM",
          "SHARD_PREWARM_THREADS_ENV": "AL_SHARD_PREWARM_THREADS",
          **NAMES}

    def _cached_file(path, name, **kw):
        return os.path.join(path, name) if os.path.isfile(os.path.join(path, name)) else None

    def _get_shard_files(path, index, **kw):
        return list(index_shards or []), None

    ns["cached_file"] = _cached_file
    ns["get_checkpoint_shard_files"] = _get_shard_files
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(SRC), "exec"), ns)
    return ns


# ---------------------------------------------------------------------------
# _resolve_weight_files：查找顺序必须与原实现一致
# ---------------------------------------------------------------------------
def test_resolve_prefers_safetensors_then_index(tmp_path):
    ns = _load_functions()
    resolve = ns["_resolve_weight_files"]
    (tmp_path / NAMES["SAFE_WEIGHTS_NAME"]).write_bytes(b"a")
    assert resolve(str(tmp_path)) == [str(tmp_path / NAMES["SAFE_WEIGHTS_NAME"])]

    (tmp_path / NAMES["SAFE_WEIGHTS_NAME"]).unlink()
    (tmp_path / NAMES["SAFE_WEIGHTS_INDEX_NAME"]).write_text("{}", encoding="utf-8")
    shards = [str(tmp_path / "s1.safetensors"), str(tmp_path / "s2.safetensors")]
    for s in shards:
        Path(s).write_bytes(b"a")
    ns2 = _load_functions(index_shards=shards)
    assert ns2["_resolve_weight_files"](str(tmp_path)) == shards


def test_resolve_falls_back_to_bin_and_raises_when_empty(tmp_path):
    ns = _load_functions()
    (tmp_path / NAMES["WEIGHTS_NAME"]).write_bytes(b"a")
    assert ns["_resolve_weight_files"](str(tmp_path)) == [str(tmp_path / NAMES["WEIGHTS_NAME"])]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        ns["_resolve_weight_files"](str(empty))


# ---------------------------------------------------------------------------
# _parallel_prewarm_shards
# ---------------------------------------------------------------------------
def test_prewarm_reads_all_files_in_parallel(tmp_path):
    ns = _load_functions()
    files, total = [], 0
    for i in range(3):
        p = tmp_path / f"s{i}.safetensors"
        p.write_bytes(b"x" * (1024 * (i + 1)))
        files.append(str(p))
        total += p.stat().st_size
    ns["_resolve_weight_files"] = lambda *a, **k: files
    info = ns["_parallel_prewarm_shards"](str(tmp_path))
    assert info["enabled"] is True and info["error"] is None
    assert info["files"] == 3 and info["bytes"] == total
    assert info["seconds"] is not None and info["seconds"] >= 0
    assert 1 <= info["threads"] <= 3           # 线程数不超过文件数


def test_prewarm_env_gate_and_thread_override(tmp_path, monkeypatch):
    ns = _load_functions()
    p = tmp_path / "s0.safetensors"
    p.write_bytes(b"x" * 16)
    ns["_resolve_weight_files"] = lambda *a, **k: [str(p)]

    monkeypatch.setenv("AL_SHARD_PREWARM", "0")
    assert ns["_parallel_prewarm_shards"](str(tmp_path))["enabled"] is False

    monkeypatch.setenv("AL_SHARD_PREWARM", "1")
    monkeypatch.setenv("AL_SHARD_PREWARM_THREADS", "2")
    assert ns["_parallel_prewarm_shards"](str(tmp_path))["threads"] == 1   # 受文件数限制
    assert ns["_parallel_prewarm_shards"](str(tmp_path))["enabled"] is True


def test_prewarm_failure_never_blocks_loading(tmp_path):
    """解析失败 ⇒ 只记 error，**不抛异常**（加载照常继续）。"""
    ns = _load_functions()

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    ns["_resolve_weight_files"] = boom
    info = ns["_parallel_prewarm_shards"](str(tmp_path))
    assert info["enabled"] is False and "disk on fire" in (info["error"] or "")


def test_prewarm_is_wired_before_state_dict_load():
    """接线检查：预读必须发生在 `_load_state_dict` **之前**（否则等于没做）。"""
    src = SRC.read_text(encoding="utf-8")
    assert "_parallel_prewarm_shards(weights_path" in src
    assert src.index("_parallel_prewarm_shards(weights_path") < src.index(
        "state_dict_iterators = _load_state_dict(weights_path)")
    assert "_resolve_weight_files" in src and "ThreadPoolExecutor" in src
