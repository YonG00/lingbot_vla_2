"""GPU 四腿对照探针的**离线**测试：跨腿对拍 / 单腿自检 / 判定表 / PLAN ONLY 无副作用。

真实模型与 GPU 部分不在这里测（那必须有卡）；这里只保证：
1. 跨腿比较按**样本身份**配对（不是顺序、不是时间戳），逐位结论与**真实误差**都落进证据；
2. 单腿自检对"文件缺失 / 钩子未执行 / 身份不匹配 / 非有限"一律 BLOCKED；
3. 判定表把三种交叉结果翻成正确结论（缓存 / varlen / serial 不稳定）；
4. `--probe-layer0` 的比较在形状不同时**不猜**语义（可判就给逐位，不可判就明说）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import types

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lingbotvla.utils import eval_batch_probe as ebp  # noqa: E402
from tools import eval_batch_gpu_probe as probe  # noqa: E402


EXPECTED = [
    {"task": "click_bell", "episode_id": 51, "chunk_start": 0, "dataset_index": 0},
    {"task": "click_bell", "episode_id": 51, "chunk_start": 50, "dataset_index": 50},
]


def _mk_leg(tmp_path, name, *, cleared=False, batch=False, bias=0.0, drop_file=None,
            skip_batch=False):
    """铺一条腿的证据（同噪声起点的多条 forward 之一）。"""
    out = tmp_path / f"leg_{name}"
    out.mkdir(parents=True, exist_ok=True)
    recorder = ebp.ProbeRecorder()
    for position, base in enumerate(EXPECTED):
        rng = np.random.default_rng(1000 + position)
        noise = rng.standard_normal((4, 3)).astype(np.float32)      # 所有腿噪声一致
        inputs = {"images": np.full((3, 4, 4), 0.5 + position, np.float32),
                  "img_masks": np.ones(1, np.bool_), "lang_tokens": np.arange(2, dtype=np.int64),
                  "lang_masks": np.ones(2, np.bool_),
                  "state": np.full(3, 0.25 * position, np.float32),
                  "image_grid_thw": np.asarray([[1, 4, 4]], dtype=np.int64)}
        output = noise.copy() + np.float32(bias)
        recorder.add(identity=ebp.make_identity(
            task=base["task"], episode_id=base["episode_id"], chunk_start=base["chunk_start"],
            dataset_index=base["dataset_index"],
            inference_path=("batch" if (batch and not skip_batch) else "serial"),
            batch_position=(position if batch else 0)), inputs=inputs, noise=noise, output=output)
    ebp.write_group_evidence(recorder, str(out), EXPECTED,
                             extra={"leg": name, "fresh_visual_grid": cleared})
    verdict = ebp.detect_leg_problems(recorder, EXPECTED, str(out))
    ebp.write_verdict(str(out), verdict)
    if drop_file:
        (out / drop_file).unlink()
    return out


# ---------------------------------------------------------------------------
# 跨腿对拍
# ---------------------------------------------------------------------------
def test_identical_legs_compare_pass(tmp_path):
    a = _mk_leg(tmp_path, "b1_normal")
    b = _mk_leg(tmp_path, "b1_cleared", cleared=True)
    cmp = ebp.compare_evidence(str(a), str(b))
    assert cmp["status"] == "PASS" and cmp["problems"] == []
    for entry in cmp["per_sample"]:
        assert entry["noise_bitwise"] and entry["output_bitwise"]
        assert entry["output_max_abs_diff"] == 0.0
        assert entry["output_ref_absmax"] > 0      # 保留真实误差的参考量级


def test_compare_reports_real_error_not_only_bitwise(tmp_path):
    a = _mk_leg(tmp_path, "b1_cleared", cleared=True)
    b = _mk_leg(tmp_path, "b2_cleared", batch=True, bias=1e-3)
    cmp = ebp.compare_evidence(str(a), str(b))
    assert cmp["status"] == "BLOCKED"
    assert any(p.startswith("output_not_bitwise") for p in cmp["problems"]), cmp["problems"]
    assert all(not e["output_bitwise"] for e in cmp["per_sample"])
    for entry in cmp["per_sample"]:
        assert entry["output_max_abs_diff"] == pytest.approx(1e-3, abs=1e-6)
        assert entry["output_mean_abs_diff"] == pytest.approx(1e-3, abs=1e-6)


def test_compare_pairs_by_identity_not_order(tmp_path):
    a = _mk_leg(tmp_path, "b1_cleared", cleared=True)
    # 同内容但**顺序反了**的腿：按身份配对 ⇒ 仍然 PASS（顺序不该造成假失败）
    out = tmp_path / "leg_reordered"
    out.mkdir()
    recorder = ebp.ProbeRecorder()
    for position, base in enumerate(EXPECTED):
        rng = np.random.default_rng(1000 + position)
        noise = rng.standard_normal((4, 3)).astype(np.float32)
        recorder.add(identity=ebp.make_identity(
            task=base["task"], episode_id=base["episode_id"], chunk_start=base["chunk_start"],
            dataset_index=base["dataset_index"], inference_path="serial"),
            inputs={"images": np.full((3, 4, 4), 0.5 + position, np.float32),
                    "img_masks": np.ones(1, np.bool_),
                    "lang_tokens": np.arange(2, dtype=np.int64),
                    "lang_masks": np.ones(2, np.bool_),
                    "state": np.full(3, 0.25 * position, np.float32),
                    "image_grid_thw": np.asarray([[1, 4, 4]], dtype=np.int64)},
            noise=noise, output=noise.copy())
    ebp.write_group_evidence(recorder, str(out), EXPECTED)
    cmp = ebp.compare_evidence(str(a), str(out))
    assert cmp["status"] == "PASS", cmp["problems"]


def test_compare_flags_sample_set_mismatch(tmp_path):
    a = _mk_leg(tmp_path, "b1_normal")
    empty = tmp_path / "leg_empty"
    empty.mkdir()
    cmp = ebp.compare_evidence(str(a), str(empty))
    assert cmp["status"] == "BLOCKED"
    assert any(p.startswith("empty_leg") for p in cmp["problems"])


# ---------------------------------------------------------------------------
# 单腿自检
# ---------------------------------------------------------------------------
def test_leg_selfcheck_blocks_missing_file(tmp_path):
    out = _mk_leg(tmp_path, "b1_normal", drop_file="noise.npz")
    verdict = ebp.detect_leg_problems(_recorder_from(out), EXPECTED, str(out))
    assert verdict["status"] == "BLOCKED"
    assert "missing_or_empty:noise.npz" in verdict["problems"]


def test_leg_selfcheck_blocks_hook_not_executed(tmp_path):
    out = tmp_path / "leg_hookless"
    out.mkdir()
    recorder = ebp.ProbeRecorder()
    ebp.write_group_evidence(recorder, str(out), EXPECTED)
    verdict = ebp.detect_leg_problems(recorder, EXPECTED, str(out))
    assert verdict["status"] == "BLOCKED"
    assert any(p.startswith("hooks_not_executed") for p in verdict["problems"])
    assert any(p.startswith("missing_record") for p in verdict["problems"])


def test_leg_selfcheck_blocks_nonfinite(tmp_path):
    out = tmp_path / "leg_nan"
    out.mkdir()
    recorder = ebp.ProbeRecorder()
    base = EXPECTED[0]
    bad = np.asarray([[np.nan]], dtype=np.float32)
    recorder.add(identity=ebp.make_identity(
        task=base["task"], episode_id=base["episode_id"], chunk_start=base["chunk_start"],
        dataset_index=base["dataset_index"], inference_path="serial"),
        inputs={"images": bad}, noise=bad, output=bad)
    ebp.write_group_evidence(recorder, str(out), EXPECTED)
    verdict = ebp.detect_leg_problems(recorder, EXPECTED, str(out))
    assert verdict["status"] == "BLOCKED"
    assert any(p.startswith("nonfinite:") for p in verdict["problems"])


def _recorder_from(dir_path):
    """把已落盘的一条腿读回成 ProbeRecorder（仅供自检复用）。"""
    recorder = ebp.ProbeRecorder()
    for key, entry in ebp.load_leg(str(dir_path)).items():
        ident = dict(entry["identity"])
        recorder.add(identity=ident, inputs=entry["inputs"], noise=entry["noise"],
                     output=entry["output"])
    return recorder


# ---------------------------------------------------------------------------
# 判定表
# ---------------------------------------------------------------------------
def _result(cache_same, varlen_same, repeat_same, layer0=None):
    def cmp(status):
        return {"status": status}
    return {"compare": {"b1_normal_vs_b1_cleared": cmp("PASS" if cache_same else "BLOCKED"),
                        "b1_cleared_vs_b2_cleared": cmp("PASS" if varlen_same else "BLOCKED"),
                        "b1_normal_vs_b1_repeat": cmp("PASS" if repeat_same else "BLOCKED")},
            "layer0": ({} if layer0 is None else
                       {"b1_cleared_vs_b2_cleared": layer0})}


def test_decision_cache_staleness():
    d = probe.decide(_result(cache_same=False, varlen_same=False, repeat_same=True))
    assert d["conclusion"] == "visual_grid_cache_staleness"
    assert "清缓存" in d["advice"]


def test_decision_batch_equivalent_after_cache_fix():
    d = probe.decide(_result(cache_same=True, varlen_same=True, repeat_same=True))
    assert d["conclusion"] == "batch_numerically_equivalent_after_cache_fix"


def test_decision_serial_unstable_wins():
    d = probe.decide(_result(cache_same=True, varlen_same=True, repeat_same=False))
    assert d["conclusion"] == "serial_itself_unstable"


def test_decision_varlen_path_with_layer0_hints():
    d = probe.decide(_result(cache_same=True, varlen_same=False, repeat_same=True,
                             layer0={"verdict": "layer0_identical", "flags": []}))
    assert d["conclusion"] == "divergence_inside_batched_path"
    assert "第 0 层" in d["advice"] and "之后" in d["advice"]
    d2 = probe.decide(_result(cache_same=True, varlen_same=False, repeat_same=True,
                              layer0={"verdict": "layer0_differs_before_or_at_layer0",
                                      "flags": ["layer0_input_differs"]}))
    assert "之前" in d2["advice"]


# ---------------------------------------------------------------------------
# 第 0 层比较
# ---------------------------------------------------------------------------
def test_layer0_compare_bitwise_and_shape_mismatch():
    same = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    out = probe.compare_layer0({"input": same.copy(), "output": same.copy()},
                               {"input": same.copy(), "output": same.copy()},
                               label_a="b1", label_b="b2")
    assert out["verdict"] == "layer0_identical" and out["input"]["bitwise"] is True

    diff = same + np.float32(1e-3)
    out2 = probe.compare_layer0({"input": same, "output": same},
                                {"input": diff, "output": diff}, label_a="b1", label_b="b2")
    assert out2["verdict"] == "layer0_differs_before_or_at_layer0"
    assert out2["input"]["max_abs_diff"] == pytest.approx(1e-3, abs=1e-6)

    # 形状不同但元素数相同 ⇒ 按扁平比；元素数也不同 ⇒ 明说不可判（不猜）
    flat = same.reshape(-1)
    out3 = probe.compare_layer0({"input": same}, {"input": flat}, label_a="b1", label_b="b2")
    assert out3["input"]["comparable"] is True and out3["input"]["bitwise"] is True
    out4 = probe.compare_layer0({"input": same}, {"input": flat[:5]}, label_a="b1", label_b="b2")
    assert out4["input"]["comparable"] is False and "不可逐元素比" in out4["input"]["note"]


# ---------------------------------------------------------------------------
# PLAN ONLY
# ---------------------------------------------------------------------------
def test_plan_only_is_side_effect_free(tmp_path, capsys):
    out = tmp_path / "probe_out"
    rc = probe.main(["--out-dir", str(out), "--ckpt", str(tmp_path / "nope"),
                     "--val-ids", str(tmp_path / "nope.json"), "--dataset", str(tmp_path / "nope")])
    assert rc == 2                      # 路径不齐 ⇒ 明确阻塞
    assert not out.exists()             # 且**不创建任何目录**
    text = capsys.readouterr().out
    assert "PLAN ONLY" in text and "b2_cleared" in text


def test_execute_refuses_non_empty_out_dir(tmp_path):
    out = tmp_path / "busy"
    out.mkdir()
    (out / "x").write_text("y", encoding="utf-8")
    assert probe.main(["--out-dir", str(out), "--execute"]) == 2


# ---------------------------------------------------------------------------
# 证据必须在**调用前**快照（真实模型 `sample_actions` 会原地改 noise）
# ---------------------------------------------------------------------------
def _production_core():
    """按 AST 编译生产 `_infer_core`（与交付补丁测试同一手法）。"""
    import ast
    src = REPO / "lingbotvla/utils/open_loop_validation.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    cls = next(x for x in tree.body
               if isinstance(x, ast.ClassDef) and x.name == "OpenLoopValidator")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
                ("_infer_core", "_probe_capture", "_probe_identity_for", "_noise_generator")]
    ast.fix_missing_locations(cls)
    ns = {"torch": torch, "np": np, "os": __import__("os"), "Dict": dict,
          "Any": object, "List": list, "Sequence": list, "EVAL_SEED": 1234,
          "_visual_grid_cache_clear": lambda m: None,
          "_visual_grid_cache_restore": lambda m, s: None}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(src), "exec"), ns)
    return ns["OpenLoopValidator"]


class _InPlaceMutatingPolicy(torch.nn.Module):
    """忠实复刻真实模型的契约：`x_t = noise; x_t += ...` —— **原地**改调用方的张量。"""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, *,
                       noise=None, image_grid_thw=None):
        self.calls.append(noise.clone())          # 调用时的真值（我们期望证据 == 这个）
        x_t = noise                               # 别名（和真实实现一致）
        x_t += 1.0                                # 原地修改
        return x_t


def _core_validator(policy, recorder, group_ids):
    cls = _production_core()
    v = object.__new__(cls)
    v.model = policy
    v.device = torch.device("cpu")
    v.dtype = torch.float32
    v.logger = types.SimpleNamespace(info_rank0=lambda *a, **k: None, warning=lambda *a, **k: None)
    v._model_config = types.SimpleNamespace(n_action_steps=2, max_action_dim=3, action_fp32=False)
    v.args = types.SimpleNamespace(train=types.SimpleNamespace(
        eval_inference_dtype="auto", use_bf16=False))
    v._noise_gen = None
    v._precision_logged = True
    v.dump_dir = None
    v._dump_prefix = None
    v._probe_recorder = recorder
    v._probe_group_ids = group_ids
    v._probe_pass = "serial"
    v._probe_position = 0
    return v


def test_evidence_records_input_noise_not_the_mutated_output():
    """证据里的 noise 必须是**喂进去的那份**，不是被原地改写后的输出。

    2026-10-09 GPU 实测踩到：录到的 noise 与 output 逐位相同 ⇒ 误报"跨腿噪声不一致"。
    """
    items = [{"images": torch.zeros(1, 3, 4, 4), "img_masks": torch.ones(1),
              "lang_tokens": torch.tensor([1]), "lang_masks": torch.ones(1),
              "state": torch.zeros(1), "image_grid_thw": torch.tensor([[1, 4, 4]]),
              "actions": torch.zeros(2, 3)}]
    group_ids = [{"dataset_index": 0, "episode_id": 51, "chunk_start": 0, "task": "click_bell"}]
    recorder = ebp.ProbeRecorder()
    policy = _InPlaceMutatingPolicy()
    v = _core_validator(policy, recorder, group_ids)
    ft = types.SimpleNamespace(unapply=lambda item: dict(item))
    with torch.inference_mode():
        v._infer_core((items[0],), ft, fresh_visual_grid=False)

    record = next(iter(recorder.records.values()))
    fed_at_call = policy.calls[0][0].numpy()
    assert ebp.bitwise_identical(record["noise"], fed_at_call), \
        "证据里的 noise 不是调用时的输入（说明快照点在调用之后）"
    assert not ebp.bitwise_identical(record["noise"], record["output"]), \
        "证据里的 noise 与 output 相同 ⇒ 录到的是原地改写后的结果"
