"""Pure contracts for opt-in training methods; no torch import or global state."""
from __future__ import annotations

import math


def validate_flags(ee_loss=False, window_sampling=False, ln_fm=False):
    if any(type(x) is not bool for x in (ee_loss, window_sampling, ln_fm)):
        raise ValueError('method switches must be explicit booleans')
    if ee_loss and not ln_fm:
        raise ValueError('existing EE implementation requires --ln-fm with --ee-loss')
    return dict(ee_loss=ee_loss, window_sampling=window_sampling, ln_fm=ln_fm)


def aligned_weights(cache, coordinates):
    lookup = {}
    for episode, record in cache['episodes'].items():
        if len(record['frame_ids']) != len(record['weights']):
            raise ValueError('cache coordinate/weight length mismatch')
        for frame, weight in zip(record['frame_ids'], record['weights']):
            key = (int(episode), int(frame))
            if key in lookup or not math.isfinite(weight) or weight <= 0:
                raise ValueError('duplicate cache coordinate or invalid weight')
            lookup[key] = float(weight)
    keys = [(int(ep), int(frame)) for ep, frame in coordinates]
    if len(set(keys)) != len(keys) or set(keys) != set(lookup):
        raise ValueError('window cache must exactly cover filtered dataset coordinates')
    return [lookup[key] for key in keys]


def make_sampler(enabled, factory, weights, seed):
    # Off branch neither invokes a factory nor consumes any RNG.
    if not enabled:
        return None
    return factory(weights, seed)


def validate_pair(first, second):
    if first['episodes'] != 3 or second['episodes'] != 3:
        raise ValueError('paired experiment requires exactly three episodes each')
    if first['seeds'] != second['seeds']:
        raise ValueError('paired evaluation scene seeds differ')
    if any(x['oidn_errors'] != 0 or x['dtype'] != 'float32' or x['action_chunk'] != 50
           for x in (first, second)):
        raise ValueError('paired evaluation precision/render/chunk mismatch')
    return dict(official_successes=first['successes'], enhanced_successes=second['successes'],
                episodes=3, seeds=first['seeds'],
                difference=second['success_rate'] - first['success_rate'],
                note='three scenes are a small paired comparison, not a stable success-rate estimate')
