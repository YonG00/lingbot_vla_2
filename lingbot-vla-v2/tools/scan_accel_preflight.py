#!/usr/bin/env python3
"""CPU-only scan acceleration preflight; default prints plan, never starts GPU."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from lingbotvla.auto_learning.scout_cache import provenance


def plan(args):
    return {
        'status': 'PLAN_ONLY' if not args.verify else 'BLOCKED',
        'task': '50-task GMean200 scan acceleration',
        'checkpoint':str(args.checkpoint) if args.checkpoint else None,
        'eval_mode': args.eval_mode, 'hardness_mode':args.hardness_mode,
        'scout_cache_mode': 'bootstrap' if args.cache_root else 'off',
        'note': 'Serial is unchanged by default. Probe never affects PASS. '
                'Auto must not be enabled before real GPU parity approval.',
    }


def verify(args):
    p=plan(args)
    if not args.checkpoint:
        p.update(status='BLOCKED', reason='checkpoint HF directory required')
        return p
    missing=[k for k in ('manifest','norm','thresholds','baseline') if getattr(args,k) is None]
    if missing:
        p.update(status='BLOCKED',reason='missing sources: '+','.join(missing))
        return p
    shards=sorted(args.checkpoint.glob('*.safetensors'))
    sources={'eval':ROOT/'lingbotvla/utils/open_loop_validation.py',
             'model':ROOT/'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py',
             'transform':ROOT/'lingbotvla/data/vla_data/transform.py',
             'eval_precision':ROOT/'lingbotvla/utils/eval_precision.py',
             'evaluator':ROOT/'lingbotvla/auto_learning/evaluator.py',
             'gmean':ROOT/'lingbotvla/auto_learning/decision/gmean.py',
             'manifest':args.manifest,'norm':args.norm,
             'thresholds':args.thresholds,'baseline':args.baseline,
             'checkpoint_config':args.checkpoint/'config.json',
             'checkpoint_tokenizer':args.checkpoint/'tokenizer.json'}
    try:
        sig=provenance(weight_files=shards,sources=sources,options={
            'inference_dtype':args.dtype,'noise_seed':1234,
            'scout_trajs':args.scout_trajs,'stride':'per_episode',
            'image_augment':False})
    except (OSError,ValueError) as exc:
        p.update(status='BLOCKED',reason=str(exc))
        return p
    p.update({'status':'READY','provenance_sha256':sig,'weight_shards':len(shards),
             'cache_namespace':str(args.cache_root / sig) if args.cache_root else None,
             'proof':'SHA256 of raw checkpoint shard bytes + code + norm + split + thresholds + baseline'})
    return p


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify',action='store_true',help='CPU hash check; may read 12-24GiB of weight files')
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--manifest',type=Path)
    ap.add_argument('--norm',type=Path)
    ap.add_argument('--thresholds',type=Path)
    ap.add_argument('--baseline',type=Path)
    ap.add_argument('--cache-root',type=Path)
    ap.add_argument('--dtype',choices=['float32','bfloat16'],default='bfloat16')
    ap.add_argument('--scout-trajs',type=int,default=2)
    ap.add_argument('--eval-mode',choices=['serial','probe','auto'],default='serial')
    ap.add_argument('--hardness-mode',choices=['fixed','auto'],default='fixed')
    a=ap.parse_args(argv)
    if a.scout_trajs<=0:
        ap.error('scout-trajs must be positive')
    out=verify(a) if a.verify else plan(a)
    print(json.dumps(out,ensure_ascii=False,indent=2))
    return 0 if out['status'] in ('READY','PLAN_ONLY') else 2


if __name__=='__main__':sys.exit(main())
