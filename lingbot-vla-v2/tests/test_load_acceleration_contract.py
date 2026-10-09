"""加载加速的**契约与边界体检**：用边界用例撞隐藏 bug。

三组：
A. `_resolve_weight_files` 与**旧实现 `_load_state_dict` 的差分对比**（最重要 —— 重构最容易改掉行为）；
B. BF16 工具的边界：0 维/空张量、非连续、共享存储、单文件 ckpt、分片缺失、采样校验；
C. 并行预读：单文件线程数、重复调用幂等。
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[1]
MU = REPO / "lingbotvla/models/module_utils.py"
TOOL = REPO / "tools/make_bf16_ckpt.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools import make_bf16_ckpt as mk  # noqa: E402

NAMES = {
    "SAFE_WEIGHTS_NAME": "model.safetensors",
    "SAFE_WEIGHTS_INDEX_NAME": "model.safetensors.index.json",
    "DIFFUSERS_SAFETENSORS_WEIGHTS_NAME": "diffusion_pytorch_model.safetensors",
    "DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME": "diffusion_pytorch_model.safetensors.index.json",
    "WEIGHTS_NAME": "pytorch_model.bin",
    "WEIGHTS_INDEX_NAME": "pytorch_model.bin.index.json",
}


def _ns_with(source: str, funcs, *, shards=None):
    tree = ast.parse(source)
    wanted = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in funcs]
    assert len(wanted) == len(funcs), [f.name for f in wanted]
    ast.fix_missing_locations(tree)
    ns = {"os": os, "time": time, "ThreadPoolExecutor": ThreadPoolExecutor,
          "Optional": object, "Dict": dict, "Any": object, "List": list,
          "SHARD_PREWARM_ENV": "AL_SHARD_PREWARM",
          "SHARD_PREWARM_THREADS_ENV": "AL_SHARD_PREWARM_THREADS",
          "StateDictIterator": lambda f: SimpleNamespace(filename=f),
          **NAMES}

    def _cached_file(path, name, **kw):
        p = os.path.join(path, name)
        return p if os.path.isfile(p) else None

    ns["cached_file"] = _cached_file
    ns["get_checkpoint_shard_files"] = lambda path, index, **kw: (list(shards or []), None)
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "<contract>", "exec"), ns)
    return ns


def _old_impl_source() -> str:
    """从上一个提交取旧实现（含 `_load_state_dict` 的原查找顺序）。"""
    # `git show <rev>:./<path>` 中的 `./` 相对 **cwd** ⇒ cwd 必须是内层仓库目录
    out = subprocess.run(["git", "show", "df6eb95^:./lingbotvla/models/module_utils.py"],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[:200]
    return out.stdout


# ---------------------------------------------------------------------------
# A. 差分对比（新旧解析必须逐条一致）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("layout", [
    "safe_single", "safe_index", "diff_single", "diff_index", "bin_single", "bin_index",
    "safe_single+index",          # 同时存在：必须选中**单文件**（顺序优先）
    "safe_index+diff_single",     # index 先于 diffusers 单文件
    "empty",
])
def test_resolve_matches_old_implementation(tmp_path, layout):
    def touch(name):
        (tmp_path / name).write_bytes(b"x")

    if layout == "safe_single":
        touch(NAMES["SAFE_WEIGHTS_NAME"])
    elif layout == "safe_index":
        touch(NAMES["SAFE_WEIGHTS_INDEX_NAME"])
    elif layout == "diff_single":
        touch(NAMES["DIFFUSERS_SAFETENSORS_WEIGHTS_NAME"])
    elif layout == "diff_index":
        touch(NAMES["DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME"])
    elif layout == "bin_single":
        touch(NAMES["WEIGHTS_NAME"])
    elif layout == "bin_index":
        touch(NAMES["WEIGHTS_INDEX_NAME"])
    elif layout == "safe_single+index":
        touch(NAMES["SAFE_WEIGHTS_NAME"]); touch(NAMES["SAFE_WEIGHTS_INDEX_NAME"])
    elif layout == "safe_index+diff_single":
        touch(NAMES["SAFE_WEIGHTS_INDEX_NAME"]); touch(NAMES["DIFFUSERS_SAFETENSORS_WEIGHTS_NAME"])

    shards = [str(tmp_path / "s1.safetensors"), str(tmp_path / "s2.safetensors")]
    for s in shards:
        Path(s).write_bytes(b"x")

    new = _ns_with(MU.read_text(encoding="utf-8"), ["_resolve_weight_files"], shards=shards)
    old = _ns_with(_old_impl_source(), ["_load_state_dict"], shards=shards)

    if layout == "empty":
        with pytest.raises(ValueError):
            new["_resolve_weight_files"](str(tmp_path))
        with pytest.raises(ValueError):
            old["_load_state_dict"](str(tmp_path))
        return
    got_new = new["_resolve_weight_files"](str(tmp_path))
    got_old = [it.filename for it in old["_load_state_dict"](str(tmp_path))]
    assert got_new == got_old, f"{layout}: new={got_new} old={got_old}"


# ---------------------------------------------------------------------------
# B. BF16 工具边界
# ---------------------------------------------------------------------------
def _ckpt(root: Path, tensors=None, *, index=True):
    root.mkdir(parents=True, exist_ok=True)
    tensors = tensors or {"w": torch.randn(3, 2), "b": torch.randn(3),
                          "n": torch.arange(3), "flag": torch.tensor([True, False])}
    name = "model-00001-of-00001.safetensors"
    save_file(tensors, str(root / name), metadata={"format": "pt"})
    if index:
        (root / mk.INDEX_NAME).write_text(json.dumps(
            {"metadata": {"total_size": (root / name).stat().st_size},
             "weight_map": {k: name for k in tensors}}), encoding="utf-8")
    (root / "config.json").write_text("{}", encoding="utf-8")
    return root


def test_handles_scalar_empty_and_indexless_ckpt(tmp_path):
    src = _ckpt(tmp_path / "src", {
        "scalar": torch.tensor(1.5), "empty": torch.zeros(0), "w": torch.randn(2, 2),
        "i": torch.arange(2, dtype=torch.int32)}, index=False)
    dst = tmp_path / "dst"
    assert mk.main(["--src", str(src), "--dst", str(dst), "--verify", "full"]) == 0
    out = load_file(str(dst / "model-00001-of-00001.safetensors"))
    assert tuple(out["scalar"].shape) == () and out["scalar"].dtype == torch.bfloat16
    assert tuple(out["empty"].shape) == (0,)
    assert out["i"].dtype == torch.int32                       # 非浮点原样
    assert not (dst / mk.INDEX_NAME).exists()                  # 无索引则不造索引
    assert (dst / "config.json").is_file()


def test_sanitize_fixes_noncontiguous_and_shared_storage(tmp_path):
    """两条 safetensors 硬限制：非连续 / 共享存储 ⇒ `_sanitize_tensors` 必须修好。"""
    base = torch.randn(4, 4)
    tensors = {"contig": torch.randn(2, 2).t(),          # 非连续
               "a": base, "b": base.view(16)}            # 共享存储
    stats = mk._sanitize_tensors(tensors)
    assert stats["contiguous_fixed"] >= 1 and stats["shared_cloned"] >= 1
    assert tensors["contig"].is_contiguous()
    ptrs = [t.untyped_storage().data_ptr() for t in tensors.values()]
    assert len(ptrs) == len(set(ptrs)), "去重后不应再有共享存储"
    save_file(tensors, str(tmp_path / "ok.safetensors"))   # 能存下去才算修好


def test_verify_missing_shard_fails_without_crashing(tmp_path):
    src = _ckpt(tmp_path / "src")
    dst = tmp_path / "dst"
    assert mk.main(["--src", str(src), "--dst", str(dst)]) == 0
    (dst / "model-00001-of-00001.safetensors").unlink()
    res = mk.verify(src, dst, ["model-00001-of-00001.safetensors"], torch.bfloat16)
    assert res["ok"] is False and res["problems"], res


def test_verify_sample_mode_is_deterministic(tmp_path):
    tensors = {f"k{i}": torch.randn(3) for i in range(10)}
    src = _ckpt(tmp_path / "src", tensors)
    dst = tmp_path / "dst"
    assert mk.main(["--src", str(src), "--dst", str(dst), "--verify", "none"]) == 0
    r1 = mk.verify(src, dst, ["model-00001-of-00001.safetensors"], torch.bfloat16, sample=3)
    r2 = mk.verify(src, dst, ["model-00001-of-00001.safetensors"], torch.bfloat16, sample=3)
    assert r1["ok"] and r1["checked"] == 3 and r1["n_tensors"] == 10
    assert r1["checked"] == r2["checked"]


# ---------------------------------------------------------------------------
# C. 预读
# ---------------------------------------------------------------------------
def test_prewarm_single_file_uses_one_thread_and_is_idempotent(tmp_path, monkeypatch):
    ns = _ns_with(MU.read_text(encoding="utf-8"),
                  ["_resolve_weight_files", "_parallel_prewarm_shards", "_log_info"])
    p = tmp_path / NAMES["SAFE_WEIGHTS_NAME"]      # 必须是 HF 已知文件名，解析函数才认
    p.write_bytes(b"x" * 32)
    for k in ("AL_SHARD_PREWARM", "AL_SHARD_PREWARM_THREADS"):
        monkeypatch.delenv(k, raising=False)
    a = ns["_parallel_prewarm_shards"](str(tmp_path))
    b = ns["_parallel_prewarm_shards"](str(tmp_path))
    assert a["enabled"] and b["enabled"]
    assert a["threads"] == b["threads"] == 1
    assert a["bytes"] == b["bytes"] == 32


def test_tool_refuses_duplicate_keys_across_shards(tmp_path):
    """跨分片重复 key ⇒ 必须拒绝（HF 索引会歧义）。"""
    src = tmp_path / "src"; src.mkdir()
    save_file({"same": torch.randn(2)}, str(src / "model-00001-of-00002.safetensors"))
    save_file({"same": torch.randn(2)}, str(src / "model-00002-of-00002.safetensors"))
    with pytest.raises(SystemExit):
        mk.main(["--src", str(src), "--dst", str(tmp_path / "dst")])


def test_prewarm_invalid_threads_env_is_swallowed(tmp_path, monkeypatch):
    """非法线程数（如 `abc`）⇒ 预读整体跳过、**不抛异常**（加速手段不能挡住加载）。"""
    ns = _ns_with(MU.read_text(encoding="utf-8"),
                  ["_resolve_weight_files", "_parallel_prewarm_shards", "_log_info"])
    (tmp_path / NAMES["SAFE_WEIGHTS_NAME"]).write_bytes(b"x" * 8)
    monkeypatch.setenv("AL_SHARD_PREWARM", "1")
    monkeypatch.setenv("AL_SHARD_PREWARM_THREADS", "abc")
    info = ns["_parallel_prewarm_shards"](str(tmp_path))
    assert info["enabled"] is False and "ValueError" in (info["error"] or "")


# ---------------------------------------------------------------------------
# D. 体检脚本自身的两个 bug（2026-10-09 训练机实测撞到）
# ---------------------------------------------------------------------------
def _smoke():
    from tools import load_accel_smoke as sm
    return sm


def test_smoke_prefers_package_import(monkeypatch):
    """🔴 `module_utils` 内有相对导入 ⇒ **必须优先按包导入**，按文件加载只是兜底。

    实测：训练机上按文件加载报 `ImportError: attempted relative import with no known parent package`。
    """
    import sys
    import types

    sm = _smoke()
    fake = types.ModuleType("lingbotvla.models.module_utils")
    fake.MARKER = "package"
    pkg = types.ModuleType("lingbotvla.models")
    pkg.module_utils = fake
    monkeypatch.setitem(sys.modules, "lingbotvla.models", pkg)
    monkeypatch.setitem(sys.modules, "lingbotvla.models.module_utils", fake)
    assert sm._import_loader().MARKER == "package"
    src = (REPO / "tools/load_accel_smoke.py").read_text(encoding="utf-8")
    # 用**代码级**锚点（docstring 里也会提到 spec_from_file_location，不能用裸子串）
    assert src.index("from lingbotvla.models import module_utils as mu") < src.index(
        'spec_from_file_location("loader_under_test"')


def test_smoke_runs_mode_b_even_when_mode_a_is_partial(monkeypatch, tmp_path):
    """A 返回 3（部分完成）时**不能提前 return** —— 否则真机实测/副本生成被静默跳过。"""
    sm = _smoke()
    calls = {"b": 0}
    monkeypatch.setattr(sm, "mode_a", lambda: 3)
    monkeypatch.setattr(sm, "mode_b", lambda *a, **k: (calls.__setitem__("b", calls["b"] + 1), 0)[1])
    assert sm.main(["--ckpt", str(tmp_path)]) == 3      # 部分完成
    assert calls["b"] == 1, "A 部分完成时仍必须跑 B"

    calls["b"] = 0
    monkeypatch.setattr(sm, "mode_a", lambda: 2)        # 真失败 ⇒ 立即停
    assert sm.main(["--ckpt", str(tmp_path)]) == 2
    assert calls["b"] == 0

    calls["b"] = 0
    monkeypatch.setattr(sm, "mode_a", lambda: 0)
    monkeypatch.setattr(sm, "mode_b", lambda *a, **k: (calls.__setitem__("b", calls["b"] + 1), 0)[1])
    assert sm.main(["--ckpt", str(tmp_path)]) == 0
    assert calls["b"] == 1
