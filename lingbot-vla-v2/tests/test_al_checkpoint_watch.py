"""只用临时目录验证清理阈值、完整性与保护范围；不加载 torch。"""

import importlib.util
import json
from pathlib import Path
import struct

import pytest

spec = importlib.util.spec_from_file_location(
    "al_checkpoint_watch",
    Path(__file__).resolve().parents[1] / "tools/auto_learning_pipeline/al_checkpoint_watch.py",
)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def checkpoint(run, step):
    path = run / "checkpoints" / f"global_step_{step}"
    hf = path / "hf_ckpt"
    hf.mkdir(parents=True)
    (hf / "config.json").write_text('{}')
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (hf / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
    (hf / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"w": "model-00001-of-00001.safetensors"},
    }))
    (hf / "pool_state.json").write_text(json.dumps({
        "global_step": step, "auto_learning": {"registry": {"records": {"A": {"status": "PASS"}}}},
    }))
    return path


def test_four_complete_delete_oldest_keep_three(tmp_path):
    for step in (220, 440, 660, 880):
        checkpoint(tmp_path, step)
    result = watch.prune_once(tmp_path, apply=True)
    assert result["removed"] == ["global_step_220"]
    assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir()) == [
        "global_step_440", "global_step_660", "global_step_880",
    ]
    assert watch.prune_once(tmp_path, apply=True)["removed"] == []


def test_preview_never_deletes_and_catches_up_after_restart(tmp_path):
    paths = [checkpoint(tmp_path, n) for n in (1, 2, 3, 4, 5)]
    assert watch.prune_once(tmp_path)["would_remove"] == ["global_step_1", "global_step_2"]
    assert all(p.exists() for p in paths)
    assert watch.prune_once(tmp_path, apply=True)["removed"] == ["global_step_1", "global_step_2"]


@pytest.mark.parametrize("damage", ["shard", "index", "config", "pool", "step", "temporary", "dcp"])
def test_incomplete_or_dcp_never_counts_or_gets_deleted(tmp_path, damage):
    oldest = checkpoint(tmp_path, 1)
    hf = oldest / "hf_ckpt"
    if damage == "shard":
        (hf / "model-00001-of-00001.safetensors").write_bytes(b"partial")
    elif damage == "index":
        (hf / "model.safetensors.index.json").write_text("{")
    elif damage == "config":
        (hf / "config.json").unlink()
    elif damage == "pool":
        (hf / "pool_state.json").unlink()
    elif damage == "step":
        data = json.loads((hf / "pool_state.json").read_text())
        data["global_step"] = 99
        (hf / "pool_state.json").write_text(json.dumps(data))
    elif damage == "temporary":
        (oldest / ".hf_ckpt.tmp.1").mkdir()
    else:
        (oldest / "model").mkdir()
    for n in (2, 3, 4):
        checkpoint(tmp_path, n)
    assert watch.prune_once(tmp_path, apply=True)["removed"] == []
    assert oldest.exists()


def test_protected_and_external_input_are_untouched(tmp_path):
    run = tmp_path / "run"
    original = checkpoint(tmp_path / "original", 4300)
    protected = checkpoint(run, 1)
    for n in (2, 3, 4, 5):
        checkpoint(run, n)
    result = watch.prune_once(run, apply=True, protected=[protected / "hf_ckpt", original])
    assert result["removed"] == ["global_step_2"]
    assert protected.exists() and original.exists()


def test_symlink_cannot_redirect_cleanup(tmp_path):
    outside = checkpoint(tmp_path / "outside", 1)
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints/global_step_1").symlink_to(outside, target_is_directory=True)
    for n in (2, 3, 4):
        checkpoint(run, n)
    assert watch.prune_once(run, apply=True)["removed"] == []
    assert outside.exists()


def test_directory_disappearing_during_inspection_is_skipped(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, 1)
    original = Path.iterdir

    def race(p):
        if p == path:
            raise FileNotFoundError(str(p))
        return original(p)

    monkeypatch.setattr(Path, "iterdir", race)
    assert not watch.complete_periodic_hf(path)
