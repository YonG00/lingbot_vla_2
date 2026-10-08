"""Regressions found in review of user snapshot adb2d75.

No GPU needed; avoid the expensive full VLA model and training loop.
"""
from __future__ import annotations

import json
import os
import sys
from types import ModuleType

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from lingbotvla.auto_learning.decision.thresholds import (
    PassThresholds, ThresholdsError, verify_threshold_stat_compatible,
)
from lingbotvla.utils.direct_hf_checkpoint import (
    export_model_hf_direct, resolve_export_dtype, verify_hf_weight_files,
)
from lingbotvla.utils.eval_precision import fp32_error_fp64_aggregation


def test_gmean_reference_threshold_must_not_compare_candidate_arithmetic_mean():
    table = PassThresholds(config_fingerprint="f", tasks={"t": 1.0},
                           stat="geomean")
    with pytest.raises(ThresholdsError, match="算术平均 MSE"):
        verify_threshold_stat_compatible(table)
    verify_threshold_stat_compatible(PassThresholds(stat="", tasks={"t": 1.0}))


def test_metric_subtraction_rounds_operands_to_fp32_before_fp64_aggregation():
    gt = np.array([1.0], dtype=np.float64)
    pred = np.array([1.0 + 1e-8], dtype=np.float64)
    err, baseline_input = fp32_error_fp64_aggregation(pred, gt)
    assert err.dtype == np.float64
    assert baseline_input.dtype == np.float64
    assert err[0] == 0.0  # would be 1e-8 if subtraction happened in FP64


def test_invalid_legacy_dtype_fails_loudly():
    with pytest.raises(ValueError, match="legacy save_dtype"):
        resolve_export_dtype("bf16", save_dtype="typo")


def test_hf_verification_valid_unsharded_and_sharded(tmp_path):
    w = {"a": torch.ones(2), "b": torch.ones(3)}
    save_file(w, str(tmp_path / "model.safetensors"))
    verify_hf_weight_files(str(tmp_path), w.keys())
    (tmp_path / "model.safetensors").unlink()
    save_file({"a": w["a"]}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"b": w["b"]}, str(tmp_path / "model-00002-of-00002.safetensors"))
    index = {"weight_map": {"a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    verify_hf_weight_files(str(tmp_path), w.keys())
    index["weight_map"]["b"] = "model-00001-of-00002.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(OSError, match="index"):
        verify_hf_weight_files(str(tmp_path), w.keys())


@pytest.mark.parametrize("failure", ["config_only", "truncated", "missing_tensor"])
def test_failed_hf_never_publishes_or_leaves_tmp(tmp_path, monkeypatch, failure):
    mod = ModuleType("lingbotvla.models")
    def broken_save(path, state_dict, **kwargs):
        os.makedirs(path, exist_ok=True)
        if failure == "config_only":
            (tmp_path / "ignored").write_text("x")
            with open(os.path.join(path, "config.json"), "w") as f:
                f.write("{}")
        elif failure == "truncated":
            with open(os.path.join(path, "model.safetensors"), "wb") as f:
                f.write(b"bad")
        elif failure == "missing_tensor":
            save_file({"weight": torch.ones(2,2)}, os.path.join(path, "model.safetensors"))
    mod.save_model_weights = broken_save
    monkeypatch.setitem(sys.modules, "lingbotvla.models", mod)
    model = torch.nn.Linear(2,2)
    with pytest.raises(RuntimeError, match="Direct HF export"):
        export_model_hf_direct(model, global_step=42, checkpoint_root=str(tmp_path))
    assert not list(tmp_path.glob("global_step_42*"))
