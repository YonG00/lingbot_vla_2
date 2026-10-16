"""Training-only low-noise mixture; disabled branch consumes no extra RNG."""


def low_noise_time_mixture(time, enabled=False):
    if not enabled:
        return time
    import torch

    # x_t = t * noise + (1-t) * action: small t means near the expert action.
    mask = torch.rand(time.shape, device=time.device) < 0.25
    low = torch.rand(time.shape, device=time.device, dtype=time.dtype) * 0.199 + 0.001
    return torch.where(mask, low, time)
