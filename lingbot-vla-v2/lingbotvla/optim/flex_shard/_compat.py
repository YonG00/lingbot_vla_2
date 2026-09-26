# Copyright 2026 Robbyant Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Compatibility helpers for stableVLA's supported PyTorch versions."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from torch.distributed.tensor import Shard


_shard_size_and_offset_impl = getattr(Shard, "local_shard_size_and_offset", None)
if _shard_size_and_offset_impl is None:
    _shard_size_and_offset_impl = getattr(Shard, "_local_shard_size_and_offset")
_shard_size_and_offset = cast(
    Callable[[int, int, int], tuple[int, int]],
    _shard_size_and_offset_impl,
)


def local_shard_size_and_offset(
    tensor_dim_size: int,
    num_chunks: int,
    rank: int,
) -> tuple[int, int]:
    """Call the public helper when available, falling back on PyTorch 2.8."""
    return _shard_size_and_offset(tensor_dim_size, num_chunks, rank)
