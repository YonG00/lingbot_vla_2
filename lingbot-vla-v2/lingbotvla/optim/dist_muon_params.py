# Copyright 2026 Robbyant Team and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""DistMuon parameter selection copied from the local Muon integration.

Kept separate to preserve the existing optimizer='muon' behavior.
"""
from typing import List, Optional, Sequence, Tuple
import torch.nn as nn
from torch import Tensor

_DEFAULT_ADAMW_NAME_PATTERNS: Tuple[str, ...] = (
    "embed_tokens",
    "embedding",
    "lm_head",
    "output_layer",
    # ViT-style learned tokens / positional tables. These are 2D/3D tensors but
    # not weight matrices, so orthogonalizing them is meaningless.
    "pos_embed",
    "cls_token",
    "mask_token",
    "storage_tokens",
    "register_tokens",
)


def _is_adamw_by_name(name: str, extra_patterns: Sequence[str]) -> bool:
    lname = name.lower()
    for pat in _DEFAULT_ADAMW_NAME_PATTERNS:
        if pat in lname:
            return True
    for pat in extra_patterns:
        if pat and pat.lower() in lname:
            return True
    return False


def _is_muon_eligible_shape(param: Tensor) -> bool:
    """Return True for shapes Muon should actually orthogonalize.

    A 3D tensor with a leading dim of 1 (``[1, N, D]`` positional tables,
    resampler queries/latents, ...) is a single learned table, not an expert
    stack of weight matrices, and is routed to AdamW.
    """
    if param.ndim == 2:
        return True
    if param.ndim == 3:
        return param.shape[0] > 1
    return False


def split_muon_adamw_params(
    model: "nn.Module",
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
    extra_adamw_name_patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[Tensor], List[Tensor], List[str], List[str]]:
    """Split model parameters into Muon-eligible weights and AdamW fallback weights."""
    no_decay_modules = no_decay_modules or []
    no_decay_params = no_decay_params or []
    extra_patterns = list(extra_adamw_name_patterns or ())

    forced_adamw_fqns: set = set()
    for module_name, module in model.named_modules():
        cls_name = module.__class__.__name__
        is_embedding = isinstance(module, nn.Embedding)
        is_no_decay = cls_name in no_decay_modules
        if is_embedding or is_no_decay:
            for pname, _p in module.named_parameters(recurse=False):
                fqn = f"{module_name}.{pname}" if module_name else pname
                forced_adamw_fqns.add(fqn)

    muon_params: List[Tensor] = []
    adamw_params: List[Tensor] = []
    muon_names: List[str] = []
    adamw_names: List[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        muon_ok = _is_muon_eligible_shape(param)
        forced_adamw = (
            (not muon_ok)
            or name in forced_adamw_fqns
            or _is_adamw_by_name(name, extra_patterns)
            or any(p and p.lower() in name.lower() for p in no_decay_params)
        )
        if forced_adamw:
            adamw_params.append(param)
            adamw_names.append(name)
        else:
            muon_params.append(param)
            muon_names.append(name)

    return muon_params, adamw_params, muon_names, adamw_names
