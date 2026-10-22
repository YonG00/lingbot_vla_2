"""Experimental sample-count loss weighting; no effect when weights are absent."""
import torch


def weighted_mean(values, weights=None, mask=None):
    if weights is None:
        return values.mean() if mask is None else (values * mask).sum() / mask.sum().clamp(min=1)
    if weights.ndim != 1 or len(weights) != len(values):
        raise ValueError("sample weights must match batch dimension")
    # Metadata weights are validated on CPU before collation. Never learn weights.
    w = weights.detach().to(device=values.device, dtype=torch.float32)
    w = w.reshape((-1,) + (1,) * (values.ndim - 1))
    if mask is None:
        per_sample = values.float().reshape(len(values), -1).mean(dim=1)
        return (per_sample * w.reshape(-1)).mean()
    valid = mask.float().expand_as(values)
    return (values.float() * valid * w).sum() / valid.sum().clamp(min=1)
