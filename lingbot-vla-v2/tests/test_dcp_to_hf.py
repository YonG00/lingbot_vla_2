"""`tools/dcp_to_hf.py` 的 CPU 测试（**不需要 GPU / 分布式**）。

为什么需要这个工具：50 任务那轮训练只落了 DCP，而闭环评测要求 HF 目录 ⇒ 之前**没有**转换脚本。
本测试用**合成 DCP** + **替身 save_model_weights**（真实的那个本地导不进来：transformers 版本差异）
跑通整条链路，钉住：
* DCP → state_dict（真实 `dcp_to_torch_state_dict`）；
* 产出布局必须是 `<checkpoint-root>/global_step_<N>/hf_ckpt`（闭环脚本硬要求）；
* 浮点按目标 dtype 导出、非浮点原样保留；
* key 集合缺一即失败；
* 资源目录按**目录**拷贝（不是把路径当 ModelAssets 传进去）；
* 重复导出会加 `_retry_N` 后缀而不是覆盖。
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import dcp_to_hf  # noqa: E402


def _fake_save_model_weights(output_dir, state_dict, global_rank=None, save_dtype="bfloat16",
                             shard_size=5_000_000_000, safe_serialization=True,
                             model_assets=None):
    """替身：写一个真实分片 + 索引（真实实现本地导不进来）。"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    name = "model-00001-of-00001.safetensors"
    save_file(state_dict, os.path.join(output_dir, name), metadata={"format": "pt"})
    total = sum(int(t.numel()) * int(t.element_size()) for t in state_dict.values())
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total},
                   "weight_map": {k: name for k in state_dict}}, f)


@pytest.fixture()
def stub_models(monkeypatch):
    """把 `lingbotvla.models.save_model_weights` 换成替身（`export_model_hf_direct` 内部才 import）。"""
    fake = types.ModuleType("lingbotvla.models")
    fake.save_model_weights = _fake_save_model_weights
    monkeypatch.setitem(sys.modules, "lingbotvla.models", fake)
    return fake


def _make_dcp(root: Path, *, step: int = 750):
    """造一个与训练器同构的 DCP：`<root>/global_step_750/model/...`。"""
    import torch.distributed.checkpoint as dcp

    ckpt = root / "checkpoints" / f"global_step_{step}"
    sd = {"model": {"w": torch.randn(8, 4), "b": torch.randn(8),
                    "counter": torch.arange(8, dtype=torch.int64),
                    "flag": torch.tensor([True, False])}}
    dcp.save(sd, checkpoint_id=str(ckpt))
    return ckpt, sd["model"]


def test_convert_end_to_end_bf16(tmp_path, stub_models):
    ckpt, ref = _make_dcp(tmp_path)
    assets = tmp_path / "model_assets"
    assets.mkdir()
    (assets / "config.json").write_text('{"model_type": "t"}', encoding="utf-8")
    (assets / "tokenizer.json").write_text("{}", encoding="utf-8")

    out = dcp_to_hf.convert(dcp=ckpt, step=750, checkpoint_root=ckpt.parent,
                            dtype="bf16", model_assets=str(assets), log=lambda *_: None)
    # 闭环脚本硬要求的布局
    assert out == ckpt / "hf_ckpt" and (out / "model.safetensors.index.json").is_file()
    idx = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert set(idx["weight_map"]) == set(ref)
    # 资源按目录拷进来
    assert (out / "config.json").is_file() and (out / "tokenizer.json").is_file()
    # 浮点 → bf16 且逐位一致；非浮点原样
    from safetensors.torch import load_file
    got = load_file(str(out / idx["weight_map"]["w"]))
    assert got["w"].dtype == torch.bfloat16 and torch.equal(got["w"], ref["w"].to(torch.bfloat16))
    assert got["counter"].dtype == torch.int64 and torch.equal(got["counter"], ref["counter"])
    assert got["flag"].dtype == torch.bool


def test_second_export_goes_to_retry_dir(tmp_path, stub_models):
    """重复导出**不覆盖**已有产物（`_retry_001`），与训练器里程碑语义一致。"""
    ckpt, _ = _make_dcp(tmp_path)
    a = dcp_to_hf.convert(dcp=ckpt, step=750, checkpoint_root=ckpt.parent, dtype="bf16",
                          model_assets=None, log=lambda *_: None)
    b = dcp_to_hf.convert(dcp=ckpt, step=750, checkpoint_root=ckpt.parent, dtype="bf16",
                          model_assets=None, log=lambda *_: None)
    assert a == ckpt / "hf_ckpt"
    assert b == ckpt.parent / "global_step_750_retry_001" / "hf_ckpt"
    assert b.is_dir() and a.is_dir()


def test_key_missing_in_shard_but_present_in_index_fails(tmp_path, stub_models):
    """分片里少了张量但 index 仍列着 ⇒ **必须失败**（只查索引会漏掉这种损坏）。"""
    ckpt, ref = _make_dcp(tmp_path)
    out = dcp_to_hf.convert(dcp=ckpt, step=750, checkpoint_root=ckpt.parent, dtype="bf16",
                            model_assets=None, log=lambda *_: None)
    shard = next(out.glob("*.safetensors"))
    save_file({k: v for k, v in ref.items() if k != "b"}, str(shard))
    with pytest.raises(RuntimeError, match="分片里没有"):
        dcp_to_hf._verify(out, ref, "bf16", sample=0, log=lambda *_: None)


def test_key_set_mismatch_fails(tmp_path, stub_models):
    """索引与分片都缺同一个 key ⇒ key 集合不一致必须失败（fail-closed）。"""
    ckpt, ref = _make_dcp(tmp_path)
    out = dcp_to_hf.convert(dcp=ckpt, step=750, checkpoint_root=ckpt.parent, dtype="bf16",
                            model_assets=None, log=lambda *_: None)
    kept = {k: v for k, v in ref.items() if k != "b"}
    save_file(kept, str(next(out.glob("*.safetensors"))))
    idx_path = out / "model.safetensors.index.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    idx["weight_map"] = {k: v for k, v in idx["weight_map"].items() if k != "b"}
    idx_path.write_text(json.dumps(idx), encoding="utf-8")
    with pytest.raises(RuntimeError, match="key 集合不一致"):
        dcp_to_hf._verify(out, ref, "bf16", sample=0, log=lambda *_: None)


def test_dry_run_reads_nothing(tmp_path, stub_models, monkeypatch):
    ckpt, _ = _make_dcp(tmp_path)
    called = {"n": 0}
    monkeypatch.setattr(dcp_to_hf, "convert", dcp_to_hf.convert)          # 保持真实实现
    import lingbotvla.checkpoint.format_utils as fu
    monkeypatch.setattr(fu, "dcp_to_torch_state_dict",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or {})
    assert dcp_to_hf.convert(dcp=ckpt, step=750, checkpoint_root=ckpt.parent, dtype="bf16",
                             model_assets=None, dry_run=True, log=lambda *_: None) is None
    assert called["n"] == 0, "dry-run 不得读 DCP"


@pytest.mark.parametrize("path,step", [
    ("/x/checkpoints/global_step_750", 750),
    ("/x/global_step_1", 1),
])
def test_parse_step_from_path(path, step):
    assert dcp_to_hf._parse_step(Path(path), None) == step


def test_parse_step_requires_number():
    with pytest.raises(SystemExit):
        dcp_to_hf._parse_step(Path("/x/checkpoints/latest"), None)


def test_default_assets_prefers_run_root(tmp_path):
    ckpt, _ = _make_dcp(tmp_path)
    (tmp_path / "model_assets").mkdir()
    assert dcp_to_hf._default_assets(ckpt, None) == str(tmp_path / "model_assets")
    assert dcp_to_hf._default_assets(ckpt, "/explicit") == "/explicit"
