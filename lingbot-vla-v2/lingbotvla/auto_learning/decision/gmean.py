"""Candidate trajectory geometric MSE.  Same log-space aggregation as reference calibration.

Return None for missing/invalid/incomplete inputs. Zero error is a valid GMean=0;
underflow is avoided by summing logs rather than multiplying errors.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Optional


def geometric_mse(values, *, expected_count: int, ids=None) -> Optional[float]:
    """Equal-weight per-trajectory geometric mean, fail closed on malformed data."""
    if isinstance(values, Mapping):
        data = list(values.values())
        if ids is not None and (len(ids) != len(data) or set(ids) != set(values)):
            return None
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        data = list(values)
        if ids is not None and (len(ids) != len(data) or len(set(ids)) != len(ids)):
            return None
    else:
        return None
    if not data or type(expected_count) is not int or expected_count != len(data):
        return None
    try:
        vals = [float(v) for v in data]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) and v >= 0 for v in vals):
        return None
    if any(v == 0 for v in vals):
        return 0.0
    return math.exp(math.fsum(math.log(v) for v in vals) / len(vals))
