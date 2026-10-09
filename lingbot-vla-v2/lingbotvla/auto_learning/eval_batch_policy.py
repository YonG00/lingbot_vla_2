"""Fail-closed adaptive evaluation batching policy.

This module does NOT monkey-patch _infer_one or modify pass metrics. The inference
backend must explicitly implement real batched calls with per-trajectory RNG.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class EvalBatchSettings:
    mode: str = 'serial'     # serial is the backwards-compatible default
    candidates: tuple[int,...] = (1,2,4,8)
    reserve_gib: float = 10.0
    atol: float = 1e-6
    rtol: float = 1e-4

    def __post_init__(self):
        if self.mode not in ('serial','auto'):
            raise ValueError('mode must be serial or auto')
        if not self.candidates or self.candidates[0] != 1 or any(
            not isinstance(x,int) or isinstance(x,bool) or x < 1
            for x in self.candidates
        ) or tuple(sorted(set(self.candidates))) != self.candidates:
            raise ValueError('candidates must be ordered unique positive ints beginning with 1')
        if not (math.isfinite(self.reserve_gib) and 0 < self.reserve_gib < 1024):
            raise ValueError('invalid reserve_gib')
        if not all(math.isfinite(x) and x >= 0 for x in (self.atol,self.rtol)):
            raise ValueError('invalid parity tolerances')


def _flatten_numbers(obj: Any) -> list[float]:
    # Tensor values are detached only for comparison, without altering inference.
    if hasattr(obj, 'detach') and hasattr(obj,'cpu'):
        obj=obj.detach().cpu().tolist()
    if isinstance(obj,dict):
        # 与 scan_accel._flatten 同契约：normalized_action_predictions 产出 list[dict]，
        # 不处理 dict 会让 outputs_close 落进 except 分支恒返回 False（parity 假失败）。
        out=[]
        for k in sorted(obj):
            out.extend(_flatten_numbers(obj[k]))
        return out
    if isinstance(obj,bool):
        raise ValueError('boolean prediction cannot be numerically compared')
    if isinstance(obj,(float,int)):
        x=float(obj)
        if not math.isfinite(x):
            raise ValueError('nonfinite eval output')
        return [x]
    if isinstance(obj,(list,tuple)):
        out=[]
        for v in obj:
            out.extend(_flatten_numbers(v))
        return out
    if isinstance(obj,dict):
        out=[]
        for k in sorted(obj):
            out.extend(_flatten_numbers(obj[k]))
        return out
    raise ValueError(f'unsupported prediction type {type(obj)!r}; cannot prove parity')


def same_structure(a, b) -> bool:
    """数值比较前的**结构守卫**：dict 的 key 集合与嵌套层级必须一致。

    仅按数值打平会让 {"x":[1,2]} 与 {"y":[1,2]} 判成相等 ⇒ **假 PASS**。
    （用户 2026-10-09 要求的反例：key/层级不匹配不得因数值相同而通过。）
    """
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return False
        if sorted(a) != sorted(b):
            return False
        return all(same_structure(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            return False
        return len(a) == len(b) and all(same_structure(x, y) for x, y in zip(a, b))
    return True


def outputs_close(ref:Sequence, candidate:Sequence, *, atol:float,rtol:float) -> bool:
    if len(ref)!=len(candidate):
        return False
    if not same_structure(ref, candidate):
        return False
    try:
        for a,b in zip(ref,candidate):
            fa,fb=_flatten_numbers(a),_flatten_numbers(b)
            if len(fa)!=len(fb):
                return False
            if any(abs(x-y)>atol+rtol*abs(x) for x,y in zip(fa,fb)):
                return False
    except (ValueError,TypeError):
        return False
    return True


def profile_batch_candidates(
    probe_items:Sequence[Any], *,
    serial_infer:Callable[[Any], Any],
    batch_infer:Callable[[Sequence[Any]],Sequence[Any]],
    available_gib:Callable[[],float],
    settings:EvalBatchSettings,
    world_size:int=1,
    before_batch:Callable[[],None]|None=None,
    synchronize:Callable[[],None]|None=None,
    peak_free_gib:Callable[[],float]|None=None,
) -> dict:
    """Actually execute batched inference and choose measured throughput winner.

    Each item must carry its own deterministic trajectory RNG to the backend.
    Tests only batches with enough items, exact order and numerical parity.
    Any unexpected OOM/exception propagates; never resumes poisoned CUDA state.
    """
    if not probe_items:
        raise ValueError('probe_items must not be empty')
    if world_size!=1:
        return {'status':'BLOCKED','reason':'FSDP2 multi-rank needs collective safe batch agreement',
                'selected_batch':1,'observations':[]}
    if settings.mode=='serial':
        return {'status':'SERIAL','selected_batch':1,'observations':[]}
    if peak_free_gib is None:
        return {'status':'BLOCKED','reason':'peak GPU memory sensor required; end-of-call free VRAM is insufficient',
                'selected_batch':1,'observations':[]}
    before=float(available_gib())
    if not math.isfinite(before) or before <= settings.reserve_gib:
        return {'status':'BLOCKED','reason':'insufficient initial free GPU memory',
                'selected_batch':1,'observations':[]}
    # Parity baseline same exact items; serial uses independent, fixed per-item seeds.
    reference=[]
    for item in probe_items:
        reference.append(serial_infer(item))
    observations=[]
    best_batch=1
    best_speed=0.0
    for batch_size in settings.candidates:
        if batch_size>len(probe_items):
            continue
        if available_gib() <= settings.reserve_gib:
            break
        sample=probe_items[:batch_size]
        if before_batch:
            before_batch()
        if synchronize:
            synchronize()
        t0=time.perf_counter()
        # Do not catch CUDA OOM: caller must stop this process and avoid silent retries.
        predicted=list(batch_infer(sample)) if batch_size>1 else [serial_infer(x) for x in sample]
        if synchronize:
            synchronize()
        seconds=time.perf_counter()-t0
        free_after=float(available_gib())
        peak_free=float(peak_free_gib())
        parity=outputs_close(reference[:batch_size],predicted,atol=settings.atol,rtol=settings.rtol)
        safe=(math.isfinite(free_after) and math.isfinite(peak_free) and
              min(free_after,peak_free) >= settings.reserve_gib
              and math.isfinite(seconds) and seconds > 0 and parity)
        obs={'batch':batch_size,'seconds':seconds,'items_per_second':batch_size/seconds,
             'free_after_gib':free_after,'peak_free_gib':peak_free,'parity':parity,'safe':safe}
        observations.append(obs)
        if not safe:
            # A failed candidate is not silently adopted or retried; shrink on next call.
            break
        if obs['items_per_second']>best_speed:
            best_batch,best_speed=batch_size,obs['items_per_second']
    if not observations or best_speed==0:
        return {'status':'BLOCKED','selected_batch':1,'observations':observations}
    return {'status':'CALIBRATED','selected_batch':best_batch,'observations':observations,
            'note':'Integration MUST still check memory before every batched call; do not change formal PASS metrics.'}


def allowed_runtime_batch(requested:int, free_gib:float, *, reserve_gib:float,
                          calibrated_peak_extra_gib:float, world_size:int=1) -> int:
    """Conservative fallback: if current free bytes change, return serial (1)."""
    if requested<=1 or world_size!=1:
        return 1
    if not all(math.isfinite(x) for x in (free_gib,reserve_gib,calibrated_peak_extra_gib)):
        return 1
    if free_gib < reserve_gib+max(0.0,calibrated_peak_extra_gib):
        return 1
    return requested
