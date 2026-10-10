"""hardness 分片：**真 3 进程 gloo** 集成测试（CPU）。

目的
----
证明"按 rank 分片 + all_gather 汇总"与单卡全量**逐位一致**，并且各 rank 拿到同一份汇总结果。
单测（`test_hardness_shard_parallel.py`）用的是假分片（顺序切片），这里用真的
`torch.distributed` + `AL_HARDNESS_SHARD=1`，把 `all_gather_object` 那条路径也覆盖到。

无 torch / 无 torchrun 时跳过。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHILD = Path(__file__).with_name('_hardness_shard_child.py')
IDS = list(range(23))


def _torchrun():
    exe = shutil.which('torchrun')
    if exe:
        return [exe]
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    return [sys.executable, '-m', 'torch.distributed.run']


def _load_child_module():
    """按文件路径加载桩模块（`tests/` 不是包，不能 `from tests.…`）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location('_hardness_shard_child', CHILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reference_scores() -> dict:
    """单卡全量（绑定 ID）作为基准。"""
    sys.path.insert(0, str(ROOT))
    child = _load_child_module()
    from lingbotvla.auto_learning.real.backend import RealHardnessScorer

    os.environ['AL_HARDNESS_UNIFY_RNG'] = '1'
    os.environ.pop('AL_HARDNESS_SHARD', None)
    try:
        scorer = RealHardnessScorer(child.NoiseAwareStubCore(), child.StubDataset(),
                                    max_batch=5, logger=None)
        got = scorer.score('shard_task', IDS)
    finally:
        os.environ.pop('AL_HARDNESS_UNIFY_RNG', None)
    return {int(k): float(v) for k, v in got.items()}


def test_three_rank_shard_matches_single_rank(tmp_path):
    tr = _torchrun()
    if tr is None:
        pytest.skip('无 torch/torchrun')
    ref = _reference_scores()
    assert set(ref) == set(IDS), '基准必须覆盖全部样本'

    env = dict(os.environ)
    env['AL_HARDNESS_SHARD'] = '1'          # ← 被测开关
    env['MASTER_ADDR'] = '127.0.0.1'
    env['MASTER_PORT'] = str(29500 + (os.getpid() % 500))
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    proc = subprocess.run(
        [*tr, '--nnodes=1', '--nproc-per-node=3', '--master-addr=127.0.0.1',
         '--master-port', env['MASTER_PORT'], str(CHILD), str(tmp_path)],
        capture_output=True, text=True, errors='replace', timeout=300, env=env)

    payloads = [json.loads(p.read_text(encoding='utf-8'))
                for p in sorted(tmp_path.glob('rank*.json'))]
    if not payloads:
        pytest.skip(f'torchrun 在本机跑不起来（非本仓库问题）：rc={proc.returncode} '
                    f'{(proc.stderr or proc.stdout)[-400:]}')
    assert len(payloads) == 3, f'应有 3 个 rank 的输出，实际 {len(payloads)}'

    # 1) 每个 rank 都必须声称自己在分片
    for pay in payloads:
        assert pay['env_shard'] == '1'
        assert pay['timing'].get('sharded') is True, f'rank{pay["rank"]} 未走分片: {pay["timing"]}'

    # 2) 各 rank 汇总后样本数一致且等于全集（all_gather 生效、无缺号）
    for pay in payloads:
        assert pay['n'] == len(IDS), f'rank{pay["rank"]} 汇总后只有 {pay["n"]} 个样本'
        assert set(int(k) for k in pay['scores']) == set(IDS)

    # 3) 与单卡全量**逐位一致**（这是"分片不改数值"的核心断言）
    first = {int(k): v for k, v in payloads[0]['scores'].items()}
    for sid in IDS:
        assert first[sid] == pytest.approx(ref[sid], rel=0, abs=1e-12), \
            f'sid={sid} 分片={first[sid]} 全量={ref[sid]} ⇒ 分片改了数值'

    # 4) 三个 rank 拿到的汇总结果必须完全相同（各 rank 各自算 percentile/weights 的前提）
    for pay in payloads[1:]:
        assert pay['scores'] == payloads[0]['scores'], '各 rank 汇总结果不一致'
