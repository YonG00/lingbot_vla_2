# Copyright 2026 Robbyant Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LingBot-VLA integration for TorchTitan FlexShard DistMuon.

The vendored implementation in :mod:`lingbotvla.optim.flex_shard` owns the
storage-to-compute redistribution and Muon update. This module only selects
parameters, describes their temporary compute layouts, builds overlap buckets,
and attaches the AdamW fallback used by the existing Muon integration.
"""

from __future__ import annotations

from collections import OrderedDict
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from ..utils import logging
from ..utils.import_utils import is_torch_npu_available
from .flex_shard import BlockShard, BucketConfig, ComputeLayout, Owned, build_dist_muon
from .dist_muon_params import split_muon_adamw_params


logger = logging.get_logger(__name__)


_ROW_HEAD_PROJ_RE = re.compile(r"\.(q_proj|k_proj|v_proj)\.weight$")
_LAYER_RE = re.compile(r"^(.*?\.(?:layers|blocks)\.)(\d+)\.")


def _compute_axis_for_param(fqn: str, param: DTensor) -> str:
    """Choose the one mesh axis on which FlexShard redistributes Muon work."""
    names = param.device_mesh.mesh_dim_names
    if names is None:
        raise ValueError("DistMuon requires named DeviceMesh axes")
    axes = [
        (axis, names[axis], placement)
        for axis, placement in enumerate(param.placements)
    ]

    # Prefer the persistent FSDP shard axis even when its mesh has size one.
    # TorchTitan normalizes such an axis to local compute, and keeping it here
    # makes the same configuration valid for both one-rank smoke tests and
    # multi-rank runs. In HSDP this also avoids communicating over the outer
    # replicated axis when dp_shard happens to be one.
    exact_shards = [entry for entry in axes if type(entry[2]) is Shard]
    active_exact_shards = [
        entry for entry in exact_shards if param.device_mesh.size(entry[0]) > 1
    ]
    if len(active_exact_shards) == 1:
        return active_exact_shards[0][1]
    if len(active_exact_shards) > 1:
        raise NotImplementedError(
            f"DistMuon parameter {fqn!r} is sharded on multiple active mesh axes; "
            "LingBot-VLA currently supports the standard one-axis FSDP2/HSDP layout only"
        )
    if len(exact_shards) == 1:
        return exact_shards[0][1]
    if len(exact_shards) > 1:
        raise NotImplementedError(
            f"DistMuon parameter {fqn!r} has multiple unit-size shard axes; "
            "the optimizer compute axis is ambiguous"
        )

    replicated = [entry for entry in axes if type(entry[2]) is Replicate]
    active_replicated = [
        entry for entry in replicated if param.device_mesh.size(entry[0]) > 1
    ]
    if len(active_replicated) == 1:
        return active_replicated[0][1]
    if len(active_replicated) > 1:
        raise ValueError(
            f"DistMuon cannot select one compute axis for replicated parameter {fqn!r}; "
            f"active axes are {[name for _, name, _ in active_replicated]}"
        )
    if len(replicated) == 1:
        return replicated[0][1]
    raise ValueError(
        f"DistMuon cannot select a compute axis for parameter {fqn!r}; "
        f"placements are {[repr(placement) for _, _, placement in axes]}"
    )


def _attention_head_count(
    fqn: str,
    modules_by_name: Dict[str, nn.Module],
) -> Optional[int]:
    match = _ROW_HEAD_PROJ_RE.search(fqn)
    if match is None:
        return None
    projection = match.group(1)
    parts = fqn.split(".")
    if len(parts) < 3:
        return None
    attention = modules_by_name.get(".".join(parts[:-2]))
    if attention is None:
        return None
    attrs = (
        ("num_key_value_heads", "num_kv_heads", "num_heads", "num_attention_heads")
        if projection in ("k_proj", "v_proj")
        else ("num_heads", "num_attention_heads")
    )
    # Transformers attention implementations are not consistent about exposing
    # head counts directly on the module.  Qwen2 used module attributes in some
    # releases, while current Qwen3-VL keeps them only on ``attention.config``.
    # Check both so per-head BlockShard layouts survive model/version changes.
    for source in (attention, getattr(attention, "config", None)):
        if source is None:
            continue
        for attr in attrs:
            value = getattr(source, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def build_dist_muon_compute_layouts(
    model: nn.Module,
    params: Sequence[torch.Tensor],
    names: Sequence[str],
    *,
    attn_per_head: bool,
) -> Tuple[Dict[str, ComputeLayout], int]:
    """Build the per-parameter temporary layouts consumed by FlexShard."""
    if len(params) != len(names):
        raise ValueError("DistMuon params and names must be aligned")
    modules_by_name = dict(model.named_modules())
    layouts: Dict[str, ComputeLayout] = {}
    per_head_count = 0

    for param, fqn in zip(params, names):
        if not isinstance(param, DTensor):
            raise TypeError(
                f"DistMuon requires FSDP2 DTensor parameters; {fqn!r} is {type(param).__name__}"
            )
        axis_name = _compute_axis_for_param(fqn, param)

        if param.ndim == 3:
            mesh_axis_names = param.device_mesh.mesh_dim_names
            assert mesh_axis_names is not None
            storage_placement = param.placements[mesh_axis_names.index(axis_name)]
            if type(storage_placement) is not Shard or storage_placement.dim != 0:
                raise NotImplementedError(
                    f"DistMuon batched parameter {fqn!r} requires storage Shard(0); "
                    f"got {storage_placement!r}"
                )
            layouts[fqn] = ComputeLayout(shardings_by_mesh_axis={axis_name: Shard(0)})
            continue

        if param.ndim != 2:
            raise ValueError(
                f"DistMuon supports only 2D matrices and batch-first 3D matrix stacks; "
                f"{fqn!r} has shape {tuple(param.shape)}"
            )

        num_heads = _attention_head_count(fqn, modules_by_name) if attn_per_head else None
        if num_heads is not None and param.shape[0] % num_heads == 0:
            head_dim = param.shape[0] // num_heads
            layouts[fqn] = ComputeLayout(
                shardings_by_mesh_axis={
                    axis_name: BlockShard(dim=0, block_size=head_dim),
                }
            )
            per_head_count += 1
        else:
            if attn_per_head and _ROW_HEAD_PROJ_RE.search(fqn):
                logger.warning_rank0(
                    f"[dist_muon] could not resolve a valid head split for {fqn} "
                    f"with shape={tuple(param.shape)}; using whole-matrix Owned compute"
                )
            layouts[fqn] = ComputeLayout(
                shardings_by_mesh_axis={axis_name: Owned()}
            )

    return layouts, per_head_count


def build_dist_muon_bucket_configs(
    names: Sequence[str],
    *,
    layers_per_bucket: int,
) -> Tuple[BucketConfig, ...]:
    """Group adjacent transformer layers into exact-FQN overlap buckets."""
    if layers_per_bucket <= 0:
        raise ValueError("dist_muon_layers_per_bucket must be positive")

    buckets: "OrderedDict[Tuple[str, int], List[str]]" = OrderedDict()
    for fqn in names:
        match = _LAYER_RE.match(fqn)
        if match is None:
            key = ("other", 0)
        else:
            layer_prefix = match.group(1)
            layer_index = int(match.group(2))
            key = (layer_prefix, layer_index // layers_per_bucket)
        buckets.setdefault(key, []).append(fqn)

    configs = []
    for index, ((prefix, bucket_index), fqns) in enumerate(buckets.items()):
        short_prefix = prefix.rstrip(".").rsplit(".", 2)[-1] if prefix != "other" else "other"
        configs.append(
            BucketConfig(
                patterns=tuple(fqns),
                name=f"{index:03d}-{short_prefix}-{bucket_index}",
            )
        )
    return tuple(configs)


def build_flex_shard_dist_muon_optimizer(
    model: nn.Module,
    args_train,
    *,
    lr: float,
    weight_decay: float = 0.0,
    adamw_betas: Tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
) -> Optimizer:
    """Build TorchTitan DistMuon for matrices plus AdamW for other parameters."""
    if getattr(args_train, "data_parallel_mode", None) != "fsdp2":
        raise ValueError("optimizer='dist_muon' currently requires data_parallel_mode='fsdp2'")
    if bool(getattr(args_train, "use_moe_expert_lr", False)):
        raise NotImplementedError(
            "DistMuon currently requires one Muon parameter group and cannot apply per-layer expert LR scaling"
        )

    muon_params, adamw_params, muon_names, _ = split_muon_adamw_params(
        model,
        no_decay_modules=None,
        no_decay_params=None,
        extra_adamw_name_patterns=getattr(args_train, "muon_exclude_name_patterns", None) or None,
    )
    if not muon_params:
        raise RuntimeError("build_flex_shard_dist_muon_optimizer found no Muon matrix parameters")

    layouts, per_head_count = build_dist_muon_compute_layouts(
        model,
        muon_params,
        muon_names,
        attn_per_head=bool(getattr(args_train, "dist_muon_attn_per_head", True)),
    )
    bucket_configs = build_dist_muon_bucket_configs(
        muon_names,
        layers_per_bucket=int(getattr(args_train, "dist_muon_layers_per_bucket", 2)),
    )
    muon_group = {
        "params": tuple(muon_params),
        "param_names": tuple(muon_names),
    }
    muon_opt = build_dist_muon(
        [muon_group],
        compute_sharding_by_fqn=layouts,
        bucket_configs=bucket_configs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=float(getattr(args_train, "muon_momentum", 0.95)),
        nesterov=bool(getattr(args_train, "muon_nesterov", True)),
        ns_steps=int(getattr(args_train, "muon_ns_steps", 5)),
        adjust_lr_fn=getattr(args_train, "muon_adjust_lr_fn", "match_rms_adamw"),
    )

    inner_opts: List[Optimizer] = [muon_opt]
    if adamw_params:
        adamw_opt = AdamW(
            adamw_params,
            lr=lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=weight_decay,
            fused=False,
            foreach=not is_torch_npu_available(),
        )
        inner_opts.append(adamw_opt)

    from .optimizer import CombinedOptimizer

    logger.info_rank0(
        f"[dist_muon] matrix_params={len(muon_params)}, adamw_params={len(adamw_params)}, "
        f"per_head_params={per_head_count}, buckets={len(bucket_configs)}"
    )
    return CombinedOptimizer(inner_opts)


__all__ = [
    "build_dist_muon_bucket_configs",
    "build_dist_muon_compute_layouts",
    "build_flex_shard_dist_muon_optimizer",
]
