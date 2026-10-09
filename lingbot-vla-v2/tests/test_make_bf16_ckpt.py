"""`tools/make_bf16_ckpt.py` 的 CPU 测试：转换正确性 + **校验必须能抓出篡改** + 拒绝覆盖。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import make_bf16_ckpt as mk  # noqa: E402


def _make_ckpt(root: Path, *, shards: int = 2) -> Path:
    """造一个两分片的迷你 HF ckpt：含 F32 浮点 + int64 + bool。"""
    root.mkdir(parents=True, exist_ok=True)
    weight_map, total = {}, 0
    for i in range(shards):
        name = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        tensors = {
            f"layer{i}.weight": torch.randn(4, 3, dtype=torch.float32),
            f"layer{i}.bias": torch.randn(4, dtype=torch.float32),
            f"layer{i}.counter": torch.arange(4, dtype=torch.int64),
            f"layer{i}.mask": torch.tensor([True, False, True, False]),
        }
        save_file(tensors, str(root / name), metadata={"format": "pt"})
        total += (root / name).stat().st_size
        for k in tensors:
            weight_map[k] = name
    (root / mk.INDEX_NAME).write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}), encoding="utf-8")
    (root / "config.json").write_text('{"model_type": "mini"}', encoding="utf-8")
    return root


def test_convert_casts_float_keeps_int_and_index(tmp_path):
    src = _make_ckpt(tmp_path / "src")
    dst = tmp_path / "dst"
    assert mk.main(["--src", str(src), "--dst", str(dst), "--verify", "full"]) == 0

    for p in sorted(src.glob("*.safetensors")):
        a, b = load_file(str(p)), load_file(str(dst / p.name))
        assert set(a) == set(b)
        for k in a:
            if a[k].is_floating_point():
                assert b[k].dtype == torch.bfloat16
                assert torch.equal(b[k], a[k].to(torch.bfloat16)), k      # 逐位相等
            else:
                assert b[k].dtype == a[k].dtype and torch.equal(b[k], a[k]), k
    # 非权重文件与索引
    assert (dst / "config.json").read_text() == (src / "config.json").read_text()
    si = json.loads((src / mk.INDEX_NAME).read_text())
    di = json.loads((dst / mk.INDEX_NAME).read_text())
    assert si["weight_map"] == di["weight_map"]                          # 映射不变
    assert di["metadata"]["total_size"] == sum(f.stat().st_size for f in dst.glob("*.safetensors"))
    assert di["metadata"]["total_size"] < si["metadata"]["total_size"]    # 体积确实变小


def test_verify_catches_tampering(tmp_path):
    """**校验必须能抓出篡改**（否则"逐张量校验"就是摆设）。"""
    src = _make_ckpt(tmp_path / "src")
    dst = tmp_path / "dst"
    assert mk.main(["--src", str(src), "--dst", str(dst), "--verify", "none"]) == 0
    shard = sorted(dst.glob("*.safetensors"))[0]
    tensors = load_file(str(shard))
    first_float = next(k for k in sorted(tensors) if tensors[k].is_floating_point())
    tensors[first_float] = tensors[first_float] + 1.0                    # 篡改
    save_file(tensors, str(shard), metadata={"format": "pt"})

    res = mk.verify(src, dst, [p.name for p in sorted(dst.glob("*.safetensors"))],
                    torch.bfloat16)
    assert res["ok"] is False and res["problems"], res


def test_cli_returns_nonzero_when_verify_fails(tmp_path, monkeypatch):
    """校验失败 ⇒ CLI 必须**非零退出**（否则"校验"只是打印）。"""
    src = _make_ckpt(tmp_path / "src")
    dst = tmp_path / "dst"
    monkeypatch.setattr(mk, "verify", lambda *a, **k: {
        "ok": False, "checked": 1, "n_tensors": 1, "key_sets_equal": True,
        "problems": [{"shard": "s", "key": "x", "detail": "boom"}]})
    assert mk.main(["--src", str(src), "--dst", str(dst), "--verify", "full"]) == 2


def test_refuses_nonempty_dst_and_src_inside_dst(tmp_path):
    src = _make_ckpt(tmp_path / "src")
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("y", encoding="utf-8")
    with pytest.raises(SystemExit):
        mk.main(["--src", str(src), "--dst", str(busy)])
    with pytest.raises(SystemExit):
        mk.main(["--src", str(src), "--dst", str(src)])                  # 目标=源
    with pytest.raises(SystemExit):
        mk.main(["--src", str(src), "--dst", str(src / "sub")])          # 目标在源内部


def test_dry_run_writes_nothing(tmp_path):
    src = _make_ckpt(tmp_path / "src")
    dst = tmp_path / "dst"
    assert mk.main(["--src", str(src), "--dst", str(dst), "--dry-run"]) == 0
    assert not dst.exists()


def test_bin_source_is_rejected(tmp_path):
    src = _make_ckpt(tmp_path / "src")
    (src / "pytorch_model.bin").write_bytes(b"x")
    with pytest.raises(SystemExit):
        mk.main(["--src", str(src), "--dst", str(tmp_path / "d2")])
