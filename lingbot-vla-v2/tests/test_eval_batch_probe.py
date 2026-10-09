"""Eval Batch 诊断证据模块的离线单测（纯 numpy/json，**不需要 torch / 数据集依赖**）。

覆盖用户 2026-10-09 的硬要求：

1. 三类证据的文件名/记录**必须带完整样本身份**（task/episode_id/chunk_start/
   dataset_index/inference_path/batch_position/repeat_index），不靠时间戳推断；
2. 判定是 **fail-closed**：文件缺失 / 钩子未执行 / 样本 ID 不匹配 / 非有限值 /
   应逐位一致的项不一致 ⇒ BLOCKED；
3. **扰动反例**证明检测器能识别：错误输入、错误 noise、错误样本对应、
   仅输出偏差、非有限值、身份重复、钩子未执行、证据文件缺失。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lingbotvla.utils import eval_batch_probe as ebp  # noqa: E402


EXPECTED = [
    {"task": "click_bell", "episode_id": 51, "chunk_start": 0, "dataset_index": 0},
    {"task": "click_bell", "episode_id": 51, "chunk_start": 50, "dataset_index": 50},
]


def _identity(base, path, position=0, repeat=0):
    return ebp.make_identity(task=base.get("task"), episode_id=base.get("episode_id"),
                             chunk_start=base.get("chunk_start"),
                             dataset_index=base["dataset_index"], inference_path=path,
                             batch_position=(position if path == "batch" else 0),
                             repeat_index=repeat)


def _inputs(seed):
    rng = np.random.default_rng(seed)
    return {"images": rng.standard_normal((3, 8, 8)).astype(np.float32),
            "img_masks": np.ones(1, dtype=bool),
            "lang_tokens": np.arange(5, dtype=np.int64) + seed,
            "lang_masks": np.ones(5, dtype=bool),
            "state": rng.standard_normal(7).astype(np.float32),
            "image_grid_thw": np.asarray([[1, 8, 8]], dtype=np.int64)}


def build_recorder(*, swap=False, wrong_input=False, wrong_noise=False, output_bias=0.0,
                   nan=False, duplicate_identity=False, skip_serial=False,
                   repeat=True):
    """构造一份"应当逐位一致"的录制；各开关按需注入**上游**故障。"""
    recorder = ebp.ProbeRecorder()
    for position, base in enumerate(EXPECTED):
        rng = np.random.default_rng(1000 + position)
        noise = rng.standard_normal((4, 3)).astype(np.float32)
        inputs = _inputs(position)
        output = noise.copy() + 0.01 * position
        if nan and position == 0:
            output = output.copy()
            output[0, 0] = np.nan
        declared = EXPECTED[0] if (duplicate_identity and position == 1) else base
        if not skip_serial:
            recorder.add(identity=_identity(declared, "serial"), inputs=inputs,
                         noise=noise, output=output)
            if repeat:
                recorder.add(identity=_identity(declared, "serial_repeat", repeat=1),
                             inputs=inputs, noise=noise, output=output)
        # 批量路径：swap ⇒ 位置 i 拿到的是"另一个样本"的内容
        source = EXPECTED[(position + 1) % 2] if swap else base
        source_position = EXPECTED.index(source)
        batch_inputs = _inputs(source_position)
        batch_noise = np.random.default_rng(1000 + source_position).standard_normal(
            (4, 3)).astype(np.float32)
        batch_output = batch_noise.copy() + 0.01 * source_position
        if wrong_input and position == 1:
            batch_inputs = dict(batch_inputs, state=batch_inputs["state"] + 1.0)
        if wrong_noise and position == 1:
            batch_noise = batch_noise + np.float32(1e-3)
        if output_bias and position == 1:
            batch_output = batch_output + np.float32(output_bias)
        recorder.add(identity=_identity(declared, "batch", position), inputs=batch_inputs,
                     noise=batch_noise, output=batch_output)
    return recorder


def codes(problems):
    return [p.split(":")[0] for p in problems]


# ---------------------------------------------------------------------------
# 样本身份
# ---------------------------------------------------------------------------
def test_identity_key_is_stable_and_timestamp_free():
    ident = ebp.make_identity(task="click_bell", episode_id=51, chunk_start=50,
                              dataset_index=50, inference_path="batch", batch_position=1)
    key = ebp.identity_key(ident)
    assert key == "ds50:click_bell:ep51:c50:batch:b1:r0"
    assert ebp.identity_key(dict(ident)) == key
    assert not any(ch.isdigit() and len(ch) > 10 for ch in [key])   # 不含时间戳


def test_identity_all_fields_present_and_explicit_none():
    ident = ebp.make_identity(task=None, episode_id=None, chunk_start=None,
                              dataset_index=3, inference_path="serial")
    assert set(ebp.IDENTITY_FIELDS) <= set(ident)
    assert ident["task"] is None and ident["episode_id"] is None
    assert "task?" in ebp.identity_key(ident)      # 缺失显式可见，不臆造


@pytest.mark.parametrize("kwargs", [
    {"inference_path": "nope"},
    {"batch_position": -1},
    {"dataset_index": True},
    {"episode_id": "51"},
])
def test_identity_rejects_bad_values(kwargs):
    base = dict(task="t", episode_id=51, chunk_start=0, dataset_index=0,
                inference_path="serial")
    base.update(kwargs)
    with pytest.raises(ValueError):
        ebp.make_identity(**base)


# ---------------------------------------------------------------------------
# 逐位比较
# ---------------------------------------------------------------------------
def test_bitwise_identical_is_byte_exact():
    a = np.asarray([1.0, 2.0], dtype=np.float32)
    assert ebp.bitwise_identical(a, a.copy())
    assert not ebp.bitwise_identical(a, a.astype(np.float64))          # dtype 不同
    nxt = np.nextafter(np.float32(2.0), np.float32(3.0))               # 只差 1 ulp
    assert nxt != np.float32(2.0)
    assert not ebp.bitwise_identical(a, np.asarray([1.0, nxt], dtype=np.float32))
    assert not ebp.bitwise_identical(a, a.reshape(1, 2))               # shape 不同
    nan = np.asarray([np.nan], dtype=np.float32)
    assert ebp.bitwise_identical(nan, nan.copy())                      # 同 bit 的 NaN 也算一致
    assert not ebp.all_finite(nan)


# ---------------------------------------------------------------------------
# 落盘 / 读回
# ---------------------------------------------------------------------------
def test_evidence_files_carry_full_identity(tmp_path):
    recorder = build_recorder()
    files = ebp.write_group_evidence(recorder, str(tmp_path), EXPECTED)
    assert set(files) == set(ebp.EVIDENCE_STEMS)
    for stem in ebp.EVIDENCE_STEMS:
        assert os.path.getsize(files[stem]["npz"]) > 0
        assert os.path.getsize(files[stem]["json"]) > 0
    doc = json.loads(Path(files["batch_actions"]["json"]).read_text(encoding="utf-8"))
    # batch_actions = 每条样本的**两条路径**（serial + batch）的实际入参 + 输出
    assert doc["count"] == len(EXPECTED) * 2
    assert {r["identity"]["inference_path"] for r in doc["records"]} == {"serial", "batch"}
    for record in doc["records"]:
        assert set(ebp.IDENTITY_FIELDS) <= set(record["identity"])
        assert record["identity"]["inference_path"] in ebp.INFERENCE_PATHS
    # npz 里也要能自证身份（内嵌清单 + 键带身份）
    import numpy as np
    with np.load(files["noise"]["npz"], allow_pickle=False) as bundle:
        assert "__manifest__" in bundle.files
        embedded = json.loads(str(bundle["__manifest__"]))
        assert embedded["count"] == len(EXPECTED) * 2      # serial + batch 两条路径
        assert any(key.startswith("ds0_click_bell_ep51_c0_") for key in bundle.files)


def test_read_evidence_fails_loudly_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ebp.read_evidence(str(tmp_path), "noise")


# ---------------------------------------------------------------------------
# 正常组：应当 PASS
# ---------------------------------------------------------------------------
def test_clean_group_passes(tmp_path):
    recorder = build_recorder()
    ebp.write_group_evidence(recorder, str(tmp_path), EXPECTED)
    verdict = ebp.detect_problems(recorder, EXPECTED, str(tmp_path))
    assert verdict["problems"] == []
    assert verdict["status"] == "PASS"
    assert verdict["n_records"] == len(EXPECTED) * 3
    assert all(entry["output_bitwise"] for entry in verdict["per_sample"])
    assert all(all(entry["inputs_bitwise"].values()) for entry in verdict["per_sample"])


# ---------------------------------------------------------------------------
# 扰动反例：检测器必须命中
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs,expected_code", [
    ({"wrong_input": True}, "batch_input_not_bitwise"),
    ({"wrong_noise": True}, "batch_noise_not_bitwise"),
    ({"swap": True}, "sample_correspondence_mismatch"),
    ({"output_bias": 1e-3}, "batch_output_not_bitwise"),
    ({"nan": True}, "nonfinite"),
    ({"duplicate_identity": True}, "duplicate_"),
    ({"skip_serial": True}, "hooks_not_executed"),
])
def test_negative_controls_are_detected(tmp_path, kwargs, expected_code):
    recorder = build_recorder(**kwargs)
    ebp.write_group_evidence(recorder, str(tmp_path), EXPECTED)
    verdict = ebp.detect_problems(recorder, EXPECTED, str(tmp_path))
    assert verdict["status"] == "BLOCKED"
    assert any(code.startswith(expected_code) for code in verdict["problems"]), verdict["problems"]


def test_missing_evidence_file_is_blocked(tmp_path):
    recorder = build_recorder()
    ebp.write_group_evidence(recorder, str(tmp_path), EXPECTED)
    os.remove(tmp_path / "serial_repeat.npz")
    verdict = ebp.detect_problems(recorder, EXPECTED, str(tmp_path))
    assert verdict["status"] == "BLOCKED"
    assert "missing_or_empty:serial_repeat.npz" in verdict["problems"]


def test_repeat_noise_must_be_bitwise_identical(tmp_path):
    """Serial Repeat 的核心断言：两次喂进去的 noise 必须逐位相同。"""
    recorder = build_recorder()
    key = ebp.identity_key(_identity(EXPECTED[0], "serial_repeat", repeat=1))
    record = recorder.records[key]
    record["noise"] = record["noise"] + np.float32(1e-7)
    ebp.write_group_evidence(recorder, str(tmp_path), EXPECTED)
    verdict = ebp.detect_problems(recorder, EXPECTED, str(tmp_path))
    assert "repeat_noise_not_bitwise:pos0" in verdict["problems"]


def test_expected_position_identity_must_match_declaration(tmp_path):
    """位置声明的身份与期望不符（张冠李戴）⇒ sample_id_mismatch。"""
    recorder = build_recorder()
    shuffled = [dict(EXPECTED[1]), dict(EXPECTED[0])]
    verdict = ebp.detect_problems(recorder, shuffled, None, require_evidence_files=False)
    assert any(code.startswith("sample_id_mismatch") for code in verdict["problems"]), \
        verdict["problems"]
