#!/usr/bin/env python3
"""CPU plan & fail-closed checks for integrating real auto-batched open-loop eval.

The current production eval API is single-trajectory. This command deliberately
refuses to simulate multi-trajectory GPU speedups or enable unimplemented hooks.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from lingbotvla.auto_learning.eval_batch_policy import EvalBatchSettings


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('serial','auto'),default='serial')
    p.add_argument('--candidates',type=int,nargs='+',default=[1,2,4,8])
    p.add_argument('--reserve-gib',type=float,default=10.)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--backend-support-proof',type=Path,help='integration evidence produced by trainer tests')
    a=p.parse_args(argv)
    s=EvalBatchSettings(mode=a.mode,candidates=tuple(a.candidates),reserve_gib=a.reserve_gib)
    plan={'kind':'eval_batch_capability_gate','mode':s.mode,'candidates':list(s.candidates),
          'reserve_gib':s.reserve_gib,'status':'PLAN_ONLY',
          'note':'This tool never claims GPU multi-trajectory inference exists.'}
    if not a.execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    if s.mode!='auto':
        print(json.dumps({**plan,'status':'SERIAL'},ensure_ascii=False,indent=2));return 0
    if a.backend_support_proof is None or not a.backend_support_proof.is_file():
        print(json.dumps({**plan,'status':'BLOCKED','reason':'missing real inference batch adapter + numerical parity proof'},indent=2))
        return 2
    proof=json.loads(a.backend_support_proof.read_text())
    if proof.get('status')!='PASS' or proof.get('verified_batched_inference') is not True or proof.get('numerical_parity') is not True:
        print(json.dumps({**plan,'status':'BLOCKED','reason':'batched inference not proven'},indent=2));return 2
    print(json.dumps({**plan,'status':'BLOCKED',
                      'reason':'external proof cannot independently verify a real batch backend; integrate and execute parity tests in repository first'},indent=2))
    return 2

if __name__=='__main__': sys.exit(main())
