"""从已输出的优化器步日志计算确认进度，绝不使用DataLoader预取游标充当完成步数。"""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiment.task_recipes.runtime import load_run
from experiment.task_recipes.planning import confirmed_progress


def status(out):
    c,p,_=load_run(out)
    log=Path(out)/'train.log'
    steps=[int(n) for n in re.findall(r'\bStep (\d+)/\d+, Epoch',log.read_text(errors='replace'))] if log.exists() else []
    result=confirmed_progress(p,max(steps,default=0))
    result['remaining_main_steps']={t:v['steps']-result['main_steps'].get(t,0) for t,v in c['train']['main_tasks'].items()} if c['train']['mode'] in ['task_units','sequential'] else None
    result['note']='confirmed logged optimizer updates; sampler prefetch is not completion; no automatic learnability judgment'
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True)
    print(json.dumps(status(a.parse_args().out),indent=2))
