"""End-effector diagnostics on existing full-denoising open-loop predictions.

No model forward, threshold decision, RNG, or dataset mutation lives here.
Means use equal trajectory weights; padding is excluded only from EE metrics
so the historical joint GMSE and its reference remain unchanged.
"""

import math
from collections import defaultdict

import torch

from .ee_fk import EEFKRuntime, _rotation_error


def physical_feature(physical, transform, key):
    """Resolve named features through the exact FeatureTransform raw mapping."""
    if key in physical:
        return torch.as_tensor(physical[key], dtype=torch.float32, device="cpu")
    mapping = transform.key_mapping[key]["origin_keys"]
    if isinstance(mapping, str):
        return torch.as_tensor(physical[mapping], dtype=torch.float32, device="cpu")
    pieces = []
    for raw, region in mapping.items():
        value = torch.as_tensor(physical[raw.split("*")[0]], dtype=torch.float32, device="cpu")
        pieces.append(value[..., region["start"]:region["end"]])
    return torch.cat(pieces, dim=-1)


def summarize_ee_records(records):
    """Serializable per-trajectory sufficient statistics, also used by shard merge."""
    if any(row["frames"] <= 0 for row in records.values()):
        raise ValueError("EE trajectory has no non-padding frames")
    rows = list(records.values())
    if not rows:
        raise ValueError("EE evaluation has no non-padding frames")
    metrics = {"valid_frames": sum(row["frames"] for row in rows), "n_trajs": len(rows)}
    for side in ("left", "right"):
        mse = sum(row[side + "_position_sq"] / row["frames"] for row in rows) / len(rows)
        angle = sum(row[side + "_angle"] / row["frames"] for row in rows) / len(rows)
        grip = sum(row[side + "_gripper_abs"] / row["frames"] for row in rows) / len(rows)
        metrics[side + "_position_mse_m2"] = mse
        metrics[side + "_position_rmse_mm"] = math.sqrt(mse) * 1000.0
        metrics[side + "_orientation_mean_deg"] = math.degrees(angle)
        metrics[side + "_gripper_mae"] = grip
    metrics["position_rmse_mm"] = math.sqrt(sum(metrics[s + "_position_mse_m2"] for s in ("left", "right")) / 2) * 1000
    for key in ("orientation_mean_deg", "gripper_mae"):
        metrics[key] = sum(metrics[s + "_" + key] for s in ("left", "right")) / 2
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite EE evaluation result")
    return metrics


class EEErrorAccumulator:
    def __init__(self, spec):
        self.runtime = EEFKRuntime(spec)
        self.records = {}

    @torch.no_grad()
    def add(self, episode, gt, prediction, transform, action_is_pad):
        arm_key, grip_key = "action.arm.position", "action.effector.position"
        ga, pa = (physical_feature(value, transform, arm_key) for value in (gt, prediction))
        gg, pg = (physical_feature(value, transform, grip_key) for value in (gt, prediction))
        if ga.ndim != 2 or ga.shape != pa.shape or ga.shape[-1] != 12:
            raise ValueError("EE evaluation requires matching [horizon,12] absolute arm actions")
        if gg.shape != pg.shape or gg.shape != (len(ga), 2):
            raise ValueError("EE evaluation requires matching [horizon,2] gripper actions")
        if action_is_pad is None:
            raise ValueError("EE evaluation requires the real action_is_pad mask")
        pad = torch.as_tensor(action_is_pad, device="cpu").bool()
        if pad.shape != (len(ga),):
            raise ValueError("EE padding mask and prediction horizon disagree")
        valid = ~pad
        if not valid.any():
            return
        ga, pa, gg, pg = (value[valid] for value in (ga, pa, gg, pg))
        if not all(torch.isfinite(value).all() for value in (ga, pa, gg, pg)):
            raise ValueError("Non-finite action in EE evaluation")
        solver = self.runtime.solver("cpu")
        with torch.autocast(device_type="cpu", enabled=False):
            gold = solver(ga[:, :6], ga[:, 6:])
            pred = solver(pa[:, :6], pa[:, 6:])
            row = self.records.setdefault(str(episode), defaultdict(float))
            row["frames"] += len(ga)
            for i, side in enumerate(("left", "right")):
                row[side + "_position_sq"] += float((pred[side][0] - gold[side][0]).double().square().sum())
                row[side + "_angle"] += float(_rotation_error(pred[side][1], gold[side][1]).double().sum())
                row[side + "_gripper_abs"] += float((pg[:, i] - gg[:, i]).double().abs().sum())

    def result(self):
        records = {key: dict(row) for key, row in self.records.items()}
        return {"ee_metrics": summarize_ee_records(records), "ee_per_traj": records}
