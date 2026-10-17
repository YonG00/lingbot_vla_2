"""Differentiable end-effector FK supervision for flow-matching actions.

The solver intentionally does not import RoboTwin or a simulator.  A verified
robot chain is supplied as a small JSON-compatible specification, so training
and CPU tests use the same pure PyTorch implementation.  The training action
is first restored from the dataset normalization and then interpreted using
the explicit packed-action layout produced by the dataset feature transform.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _as_float_tensor(value: Any) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if not torch.isfinite(tensor).all():
        raise ValueError("ee_fk chain geometry must contain only finite values")
    return tensor


def _require_vec(value: Any, *, size: int, label: str) -> Any:
    tensor = _as_float_tensor(value)
    if tuple(tensor.shape) != (size,):
        raise ValueError(f"{label} must have shape [{size}], got {tuple(tensor.shape)}")
    return tensor.tolist()


def _rpy_matrix(rpy: Tensor) -> Tensor:
    """Return an XYZ-RPY rotation matrix using the conventional Rz Ry Rx order."""
    r, p, y = rpy.unbind(-1)
    zr = torch.zeros_like(r)
    o = torch.ones_like(r)
    rx = torch.stack(
        [o, zr, zr, zr, torch.cos(r), -torch.sin(r), zr, torch.sin(r), torch.cos(r)], -1
    ).reshape(r.shape + (3, 3))
    ry = torch.stack(
        [torch.cos(p), zr, torch.sin(p), zr, o, zr, -torch.sin(p), zr, torch.cos(p)], -1
    ).reshape(r.shape + (3, 3))
    rz = torch.stack(
        [torch.cos(y), -torch.sin(y), zr, torch.sin(y), torch.cos(y), zr, zr, zr, o], -1
    ).reshape(r.shape + (3, 3))
    return rz @ ry @ rx


def _axis_angle_matrix(axis: Tensor, angle: Tensor) -> Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(axis.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand(axis.shape[:-1] + (3, 3))
    outer = axis.unsqueeze(-1) @ axis.unsqueeze(-2)
    return torch.cos(angle)[..., None, None] * eye + (1 - torch.cos(angle))[..., None, None] * outer + torch.sin(angle)[..., None, None] * k


def _transform(rotation: Tensor, translation: Tensor) -> Tensor:
    out = torch.eye(4, dtype=rotation.dtype, device=rotation.device).expand(rotation.shape[:-2] + (4, 4)).clone()
    out[..., :3, :3] = rotation
    out[..., :3, 3] = translation
    return out


class _SerialChain(nn.Module):
    def __init__(self, spec: Mapping[str, Any], label: str):
        super().__init__()
        joints = spec.get("joints")
        if not isinstance(joints, list) or len(joints) != 6:
            raise ValueError(f"ee_fk.{label}.joints must contain exactly six named joints")
        if any(not isinstance(joint, Mapping) for joint in joints):
            raise ValueError(f"ee_fk.{label}.joints entries must be mappings")
        names = [str(j.get("name", "")) for j in joints]
        if any(not n for n in names) or len(set(names)) != len(names):
            raise ValueError(f"ee_fk.{label}.joints names must be non-empty and unique")
        self.joint_names = tuple(names)
        self.label = label
        origins_rpy = []
        origins_xyz = []
        axes = []
        for joint in joints:
            origins_xyz.append(joint.get("origin_xyz", [0.0, 0.0, 0.0]))
            origins_rpy.append(joint.get("origin_rpy", [0.0, 0.0, 0.0]))
            axes.append(joint.get("axis"))
        if any(axis is None for axis in axes):
            raise ValueError(f"ee_fk.{label}.joints entries must define axis")
        origins_xyz = [
            _require_vec(value, size=3, label=f"ee_fk.{label}.joints[{idx}].origin_xyz")
            for idx, value in enumerate(origins_xyz)
        ]
        origins_rpy = [
            _require_vec(value, size=3, label=f"ee_fk.{label}.joints[{idx}].origin_rpy")
            for idx, value in enumerate(origins_rpy)
        ]
        axes = [
            _require_vec(value, size=3, label=f"ee_fk.{label}.joints[{idx}].axis")
            for idx, value in enumerate(axes)
        ]
        if any(sum(component * component for component in axis) <= 1e-12 for axis in axes):
            raise ValueError(f"ee_fk.{label}.joints axes must be non-zero")
        self.register_buffer("origin_xyz", _as_float_tensor(origins_xyz))
        self.register_buffer("origin_rpy", _as_float_tensor(origins_rpy))
        self.register_buffer("axes", _as_float_tensor(axes))
        self.register_buffer(
            "tip_xyz",
            _as_float_tensor(_require_vec(spec.get("tip_xyz", [0.0, 0.0, 0.0]), size=3, label=f"ee_fk.{label}.tip_xyz")),
        )
        self.register_buffer(
            "tip_rpy",
            _as_float_tensor(_require_vec(spec.get("tip_rpy", [0.0, 0.0, 0.0]), size=3, label=f"ee_fk.{label}.tip_rpy")),
        )
        root = spec.get("root", {})
        if not isinstance(root, Mapping):
            raise ValueError(f"ee_fk.{label}.root must be a mapping")
        self.register_buffer(
            "root_xyz",
            _as_float_tensor(_require_vec(root.get("translation", [0.0, 0.0, 0.0]), size=3, label=f"ee_fk.{label}.root.translation")),
        )
        self.register_buffer(
            "root_rpy",
            _as_float_tensor(_require_vec(root.get("rpy", [0.0, 0.0, 0.0]), size=3, label=f"ee_fk.{label}.root.rpy")),
        )

    def forward(self, joints: Tensor) -> Tuple[Tensor, Tensor]:
        if joints.shape[-1] != 6:
            raise ValueError(f"{self.label} FK expects six joint values, got {joints.shape[-1]}")
        shape = joints.shape[:-1]
        root_r = _rpy_matrix(self.root_rpy).expand(shape + (3, 3))
        root_t = self.root_xyz.expand(shape + (3,))
        current = _transform(root_r, root_t)
        for idx in range(6):
            origin_r = _rpy_matrix(self.origin_rpy[idx]).expand(shape + (3, 3))
            origin_t = self.origin_xyz[idx].expand(shape + (3,))
            joint_r = _axis_angle_matrix(self.axes[idx].expand(shape + (3,)), joints[..., idx])
            current = current @ _transform(origin_r, origin_t) @ _transform(joint_r, torch.zeros_like(origin_t))
        tip_r = _rpy_matrix(self.tip_rpy).expand(shape + (3, 3))
        tip_t = self.tip_xyz.expand(shape + (3,))
        current = current @ _transform(tip_r, tip_t)
        return current[..., :3, 3], current[..., :3, :3]


class DifferentiableEEFKSolver(nn.Module):
    """Two-arm differentiable FK solver built from a verified chain spec."""

    def __init__(self, spec: Mapping[str, Any]):
        super().__init__()
        if not isinstance(spec, Mapping) or "left" not in spec or "right" not in spec:
            raise ValueError("ee_fk spec must contain left and right chains")
        self.left = _SerialChain(spec["left"], "left")
        self.right = _SerialChain(spec["right"], "right")

    @classmethod
    def from_path(cls, path: str | Path) -> "DifferentiableEEFKSolver":
        with Path(path).open() as handle:
            return cls(json.load(handle))

    def forward(self, left_joints: Tensor, right_joints: Tensor) -> Dict[str, Tuple[Tensor, Tensor]]:
        return {"left": self.left(left_joints), "right": self.right(right_joints)}


class EEFKRuntime:
    """Training-only geometry, deliberately outside the model module tree.

    A plain object prevents model.to(bf16), FSDP and HF weight restoration from
    rounding fixed robot geometry. Rebuild from the JSON config on a new run;
    cache only FP32 solvers by the actual action device. No parameters or RNG.
    """

    def __init__(self, spec: Mapping[str, Any]):
        self.spec = deepcopy(spec)
        self._solvers = {}

    def solver(self, device) -> DifferentiableEEFKSolver:
        key = str(torch.device(device))
        if key not in self._solvers:
            with torch.device("cpu"):
                solver = DifferentiableEEFKSolver(self.spec)
            self._solvers[key] = solver.to(device=device, dtype=torch.float32)
        return self._solvers[key]


def recover_clean_action(target: Tensor, noise: Tensor, time: Tensor, velocity: Tensor) -> Tensor:
    """Compute x_t - t*v in FP32 without first rounding the target to bf16."""
    with torch.autocast(device_type=target.device.type, enabled=False):
        t = time.float()[:, None, None]
        noisy = t * noise.float() + (1.0 - t) * target.float()
        return noisy - t * velocity.float()


def _apply_norm(value: Tensor, spec: Mapping[str, Any]) -> Tensor:
    kind = spec.get("norm_type", "identity")
    if kind == "identity":
        return value
    def stat(name: str) -> Tensor:
        raw = torch.as_tensor(spec[name], dtype=value.dtype, device=value.device)
        if raw.ndim == 2 and value.ndim == 3:
            if raw.shape[0] < value.shape[-2]:
                raise ValueError(f"ee_fk normalization horizon {raw.shape[0]} is shorter than {value.shape[-2]}")
            raw = raw[: value.shape[-2]]
        return raw
    if kind == "meanstd":
        return value * (stat("std") + 1e-6) + stat("mean")
    if kind in ("bounds_98", "bounds_98_woclip"):
        low, high = stat("q02"), stat("q98")
        return (value + 1) / 2 * (high - low + 1e-6) + low
    if kind in ("bounds_99", "bounds_99_woclip"):
        low, high = stat("q01"), stat("q99")
        return (value + 1) / 2 * (high - low + 1e-6) + low
    if kind == "std":
        return value * (stat("std") + 1e-6)
    if kind in ("minmax", "minmax_woclip"):
        return (value + 1) / 2 * (stat("max") - stat("min") + 1e-6) + stat("min")
    raise ValueError(f"Unsupported ee_fk normalization type: {kind}")


def _unnormalize_action(action: Tensor, norm_spec: Sequence[Mapping[str, Any]]) -> Tensor:
    raw = action.float().clone()
    for item in norm_spec:
        start, end = int(item["start"]), int(item["end"])
        raw[..., start:end] = _apply_norm(raw[..., start:end], item)
    return raw


def _masked_mean(value: Tensor, valid: Optional[Tensor]) -> Tensor:
    if valid is None:
        return value.mean()
    valid = valid.to(dtype=value.dtype)
    while valid.ndim < value.ndim:
        valid = valid.unsqueeze(-1)
    return (value * valid).sum() / valid.expand_as(value).sum().clamp_min(1.0)


def _rotation_error(pred: Tensor, target: Tensor) -> Tensor:
    relative = pred.transpose(-1, -2) @ target
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    skew = torch.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        dim=-1,
    )
    sine = 0.5 * skew.norm(dim=-1)
    return torch.atan2(sine, cosine.clamp_min(-1.0).clamp_max(1.0))


def _compute_ee_fk_losses_fp32(
    solver: DifferentiableEEFKSolver,
    predicted_action: Tensor,
    target_action: Tensor,
    *,
    action_layout: Mapping[str, Sequence[int]],
    action_norm_spec: Sequence[Mapping[str, Any]],
    position_weight: float,
    orientation_weight: float,
    gripper_weight: float,
    huber_delta: float = 0.01,
    valid_time: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    pred = _unnormalize_action(predicted_action, action_norm_spec)
    target = _unnormalize_action(target_action.detach(), action_norm_spec)
    left = action_layout["left_arm"]
    right = action_layout["right_arm"]
    left_pred, left_target = pred[..., left[0]:left[1]], target[..., left[0]:left[1]]
    right_pred, right_target = pred[..., right[0]:right[1]], target[..., right[0]:right[1]]
    pred_fk = solver(left_pred, right_pred)
    target_fk = solver(left_target, right_target)
    valid = valid_time
    position = 0.5 * (
        _masked_mean(F.smooth_l1_loss(pred_fk["left"][0], target_fk["left"][0], reduction="none", beta=huber_delta), valid)
        + _masked_mean(F.smooth_l1_loss(pred_fk["right"][0], target_fk["right"][0], reduction="none", beta=huber_delta), valid)
    )
    orientation = 0.5 * (
        _masked_mean(_rotation_error(pred_fk["left"][1], target_fk["left"][1]), valid)
        + _masked_mean(_rotation_error(pred_fk["right"][1], target_fk["right"][1]), valid)
    )
    grip_start, grip_end = action_layout["gripper"]
    if grip_end > grip_start:
        grip_pred = pred[..., grip_start:grip_end]
        grip_target = target[..., grip_start:grip_end]
        gripper = _masked_mean(F.smooth_l1_loss(grip_pred, grip_target, reduction="none", beta=huber_delta), valid)
    else:
        gripper = predicted_action.new_zeros(())
    total = position_weight * position + orientation_weight * orientation + gripper_weight * gripper
    return total, {
        "ee_fk/position_loss": position.detach(),
        "ee_fk/orientation_loss": orientation.detach(),
        "ee_fk/gripper_loss": gripper.detach(),
        "ee_fk/weighted_loss": total.detach(),
    }


def compute_ee_fk_losses(solver, predicted_action, target_action, **kwargs):
    """Keep geometry and the loss in FP32 even inside a training autocast."""
    with torch.autocast(device_type=predicted_action.device.type, enabled=False):
        return _compute_ee_fk_losses_fp32(solver, predicted_action.float(), target_action.float(), **kwargs)


def build_action_norm_spec(feature_transform: Any) -> list[dict[str, Any]]:
    """Serialize the exact packed action normalization used by the dataset."""
    normalizer = getattr(feature_transform, "normalizer", None)
    if normalizer is None:
        raise ValueError("ee_fk requires dataset normalization statistics")
    specs = []
    offset = 0
    for joint_name in feature_transform.feature_config.joints:
        width = int(feature_transform.feature_config.joints_max_dim[joint_name])
        key = f"action.{joint_name}"
        if key in feature_transform.actions:
            if key not in normalizer.norm_stats:
                raise ValueError(f"ee_fk requires normalization statistics for {key}")
            stats = normalizer.norm_stats[key]
            item = {
                "key": key,
                "start": offset,
                "end": offset + int(stats["mean"].shape[-1]),
                "norm_type": normalizer.norm_type[key],
            }
            for name in ("mean", "std", "q02", "q98", "q01", "q99", "min", "max"):
                if name in stats:
                    item[name] = stats[name].tolist()
            specs.append(item)
        offset += width
    return specs


def build_action_layout(feature_transform: Any) -> dict[str, list[int]]:
    """Derive arm/gripper slices from named dataset features, never by guesswork."""
    if any(getattr(feature_transform, "action_subtract_state", {}).get(key, False)
           for key in ("action.arm.position", "action.effector.position")):
        raise ValueError("ee_fk currently requires absolute arm and gripper actions")
    offset = 0
    starts = {}
    for joint_name in feature_transform.feature_config.joints:
        width = int(feature_transform.feature_config.joints_max_dim[joint_name])
        key = f"action.{joint_name}"
        if key in feature_transform.actions:
            if key not in feature_transform.normalizer.norm_stats:
                raise ValueError(f"ee_fk requires normalization statistics for {key}")
            dim = int(feature_transform.normalizer.norm_stats[key]["mean"].shape[-1])
            starts[key] = (offset, dim)
        offset += width
    arm = starts.get("action.arm.position")
    grip = starts.get("action.effector.position")
    # `joints_max_dim` is the packed slot width (14 for arm), while the
    # normalized feature itself is 12 values: [left six, right six].
    if arm is None or arm[1] != 12:
        raise ValueError(f"ee_fk requires named action.arm.position with 12 values, got {arm}")
    if grip is None or grip[1] != 2:
        raise ValueError(f"ee_fk requires named action.effector.position with 2 values, got {grip}")
    return {
        "left_arm": [arm[0], arm[0] + 6],
        "right_arm": [arm[0] + 6, arm[0] + 12],
        "gripper": [grip[0], grip[0] + 2],
    }
