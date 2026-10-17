"""FK unit tests; validation methods and evidence live beside this file."""

import math
from types import SimpleNamespace

import torch
import pytest

from lingbotvla.losses.ee_fk import (
    DifferentiableEEFKSolver,
    build_action_layout,
    build_action_norm_spec,
    compute_ee_fk_losses,
)


def _chain(tip_x=1.0):
    return {
        "root": {"translation": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        "joints": [
            {"name": f"j{i}", "origin_xyz": [0.0, 0.0, 0.0], "origin_rpy": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0]}
            for i in range(6)
        ],
        "tip_xyz": [tip_x, 0.0, 0.0],
        "tip_rpy": [0.0, 0.0, 0.0],
    }


def _solver():
    return DifferentiableEEFKSolver({"left": _chain(), "right": _chain()})


def test_fk_known_geometry_and_batch_shape():
    solver = _solver()
    joints = torch.zeros(2, 4, 6)
    joints[0, 0, 0] = math.pi / 2
    position, rotation = solver.left(joints)
    assert position.shape == (2, 4, 3)
    assert rotation.shape == (2, 4, 3, 3)
    torch.testing.assert_close(position[0, 0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=1e-6)


def test_fk_loss_has_gradient_and_separate_weights():
    solver = _solver()
    predicted = torch.zeros(1, 2, 30, requires_grad=True)
    target = torch.zeros(1, 2, 30)
    target[..., 0] = 0.2
    target[..., 28:30] = 1.0
    layout = {"left_arm": [0, 6], "right_arm": [6, 12], "gripper": [28, 30]}
    total, metrics = compute_ee_fk_losses(
        solver,
        predicted,
        target,
        action_layout=layout,
        action_norm_spec=[],
        position_weight=1.0,
        orientation_weight=0.5,
        gripper_weight=0.25,
        valid_time=torch.tensor([[True, False]]),
    )
    assert set(metrics) == {
        "ee_fk/position_loss",
        "ee_fk/orientation_loss",
        "ee_fk/gripper_loss",
        "ee_fk/weighted_loss",
    }
    assert total.item() > 0
    total.backward()
    assert predicted.grad is not None
    assert torch.count_nonzero(predicted.grad).item() > 0


def test_layout_uses_named_packed_features_and_real_dims():
    stats = {
        "action.arm.position": {"mean": torch.zeros(12).numpy()},
        "action.effector.position": {"mean": torch.zeros(2).numpy()},
    }
    transform = SimpleNamespace(
        feature_config=SimpleNamespace(
            joints=["arm.position", "end.position", "effector.position"],
            joints_max_dim={"arm.position": 14, "end.position": 14, "effector.position": 2},
        ),
        actions={"action.arm.position", "action.effector.position"},
        normalizer=SimpleNamespace(norm_stats=stats, norm_type={
            "action.arm.position": "identity",
            "action.effector.position": "identity",
        }),
    )
    assert build_action_layout(transform) == {
        "left_arm": [0, 6],
        "right_arm": [6, 12],
        "gripper": [28, 30],
    }
    norm = build_action_norm_spec(transform)
    assert [(x["key"], x["start"], x["end"]) for x in norm] == [
        ("action.arm.position", 0, 12),
        ("action.effector.position", 28, 30),
    ]


def test_action_hat_recovers_clean_action_for_exact_flow_field():
    clean = torch.randn(2, 3, 4)
    noise = torch.randn_like(clean)
    time = torch.tensor([0.1, 0.2])[:, None, None]
    x_t = time * noise + (1 - time) * clean
    v_t = noise - clean
    torch.testing.assert_close(x_t - time * v_t, clean)


def test_fk_rejects_invalid_geometry_before_training():
    bad = _chain()
    bad["joints"][2]["axis"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="axes must be non-zero"):
        DifferentiableEEFKSolver({"left": bad, "right": _chain()})

    bad = _chain()
    bad["tip_xyz"] = [0.0, 0.0]
    with pytest.raises(ValueError, match="tip_xyz must have shape"):
        DifferentiableEEFKSolver({"left": bad, "right": _chain()})
