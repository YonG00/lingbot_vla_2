"""Global NEW/Replay fraction -> deterministic, balanced local slots.

The plan is computed in *data-parallel sample* space. Pipeline/tensor parallel
ranks must not be mistaken for independent data-parallel sample consumers.
No torch dependency: runtime and CPU tests share exactly the same rounding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RatioPlan:
    global_batch_size: int
    local_batch_size: int
    dp_size: int
    dp_rank: int
    new_ratio: float
    global_new: int
    global_replay: int
    local_new: int
    local_replay: int


def ratio_plan(*, global_batch_size: int, dp_size: int,
               dp_rank: int, new_ratio: float) -> RatioPlan:
    """Round global NEW count half-up; split exact total across DP ranks.

    E.g. 112 samples at 70%: globally 78 NEW + 34 Replay (rather
    than independently rounding 28 * .7 to 20 on every rank -> 80/32).
    In each rank local_batch_size must remain identical for DDP/FSDP.
    """
    if type(global_batch_size) is not int or global_batch_size < 1:
        raise ValueError('global_batch_size must be a positive integer')
    if type(dp_size) is not int or dp_size < 1:
        raise ValueError('dp_size must be a positive integer')
    if type(dp_rank) is not int or not 0 <= dp_rank < dp_size:
        raise ValueError('dp_rank must be in [0, dp_size)')
    if global_batch_size % dp_size:
        raise ValueError('global_batch_size must be divisible by dp_size')
    if isinstance(new_ratio, bool) or not isinstance(new_ratio, (int, float)) or \
            not math.isfinite(new_ratio) or not 0.0 <= new_ratio <= 1.0:
        raise ValueError('new_ratio must be finite and in [0, 1]')
    local = global_batch_size // dp_size
    global_new = math.floor(global_batch_size * float(new_ratio) + 0.5)
    # Distribute a total among ranks with <= 1 difference, not per-rank round.
    local_new = ((dp_rank + 1) * global_new // dp_size
                 - dp_rank * global_new // dp_size)
    return RatioPlan(
        global_batch_size=global_batch_size, local_batch_size=local,
        dp_size=dp_size, dp_rank=dp_rank, new_ratio=float(new_ratio),
        global_new=global_new, global_replay=global_batch_size-global_new,
        local_new=local_new, local_replay=local-local_new,
    )
