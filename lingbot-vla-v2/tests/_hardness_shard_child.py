#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""子进程端：在 dist init 之后跑一次 hardness 分片打分，把结果写到 JSON。

    python _hardness_shard_child.py <rank> <world_size> <out_json>

⚠️ 必须用**脚本**方式启动（`torchrun 脚本 --args`），不能用 `-m torch.distributed.run`
   跑内联 `-c` —— 多进程 spawn 时内联代码没有可导入的 `__main__`。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class NoiseAwareStubCore:
    def __init__(self, seed: int = 1234):
        self.seed = seed

    def score(self, items, sample_ids=None):  # noqa: ANN001
        sids = [int(it['idx']) for it in items] if sample_ids is None else [int(s) for s in sample_ids]
        vals = []
        if sample_ids is None:
            rng = np.random.default_rng(self.seed + len(sids))
            noise = rng.standard_normal(len(sids))
            vals = [float(sid) + float(noise[k]) for k, sid in enumerate(sids)]
        else:
            for sid in sids:
                rng = np.random.default_rng(self.seed ^ (sid * 2654435761 % (1 << 32)))
                vals.append(float(sid) + float(rng.standard_normal()))
        return np.asarray(vals, dtype=float)


class StubDataset:
    def __getitem__(self, idx):  # noqa: ANN001
        return {'joint_mask': True, 'idx': int(idx)}


def main() -> int:
    # rank / world_size 由 torchrun 通过环境变量下发（不要再当位置参数传，torchrun 会吃掉）
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    out_dir = Path(sys.argv[1])
    out_path = out_dir / f'rank{rank}.json'

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    import torch.distributed as dist
    from lingbotvla.auto_learning.real.backend import RealHardnessScorer

    dist.init_process_group(backend='gloo')
    try:
        scorer = RealHardnessScorer(NoiseAwareStubCore(), StubDataset(), max_batch=4, logger=None)
        ids = list(range(23))
        got = scorer.score('shard_task', ids)
        timing = dict(scorer.last_timing)
    finally:
        dist.destroy_process_group()

    out_path.write_text(json.dumps({
        'rank': rank, 'world_size': world,
        'n': len(got),
        'scores': {str(k): v for k, v in sorted(got.items())},
        'timing': timing,
        'env_shard': os.environ.get('AL_HARDNESS_SHARD'),
    }, sort_keys=True), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
