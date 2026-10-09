#!/usr/bin/env python3
"""Opt-in BF16 deployment loader probe. PLAN ONLY unless --execute."""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='HF safetensors checkpoint directory')
    parser.add_argument('--mode', choices=['legacy', 'fast'], required=True)
    parser.add_argument('--output', required=True, help='New JSON report; never overwritten')
    parser.add_argument('--execute', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plan = {
        'mode': args.mode,
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'output': str(Path(args.output).resolve()),
        'status': 'PLAN ONLY' if not args.execute else 'PENDING',
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    # No file or GPU operation precedes this explicit execution switch.
    report_path = Path(args.output)
    if report_path.exists():
        raise FileExistsError(f'Refusing to overwrite {report_path}')
    if not Path(args.checkpoint).is_dir():
        raise NotADirectoryError(args.checkpoint)
    if not report_path.parent.is_dir():
        raise NotADirectoryError(report_path.parent)
    os.environ['LINGBOT_DEPLOY_FAST_LOAD'] = '1' if args.mode == 'fast' else '0'

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for --execute')
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    server = LingbotVLAv2Server(
        path_to_pi_model=args.checkpoint,
        use_bf16=True,
        use_fp32=False,
        use_compile=False,
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    free, total = torch.cuda.mem_get_info()
    param = next(server.vla.parameters())
    if param.device.type != 'cuda' or param.dtype != torch.bfloat16:
        raise RuntimeError(f'Loaded incorrect device/dtype: {param.device}/{param.dtype}')
    report = {
        **plan, 'status': 'LOAD_SUCCESS', 'seconds': seconds,
        'parameter_dtype': str(param.dtype),
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / (1024 ** 3),
        'cuda_free_after_gib': free / (1024 ** 3),
        'cuda_total_gib': total / (1024 ** 3),
        'process_peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
    }
    # Fail closed if a toy/invalid checkpoint managed to produce a misleading result.
    if report['cuda_free_after_gib'] < 10:
        raise RuntimeError('Post-load CUDA reserve below 10 GiB; rejecting acceptance')
    with report_path.open('x', encoding='utf-8') as out:
        json.dump(report, out, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
