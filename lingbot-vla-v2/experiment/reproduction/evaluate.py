"""Evaluate a completed training run or an HF checkpoint, optionally recording videos."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiment.reproduction.run import load,curriculum
from experiment.task_recipes.recovery_cycle import historical_best


def build_job(a):
    settings=load(a.run_dir/'settings.json')if a.run_dir else {}
    model=a.model_path or Path(load(a.run_dir/'final_model.json')['path'])
    qwen=a.qwen_model or settings.get('qwen_model')
    sim=a.sim_python or settings.get('sim_python')
    if not qwen or not sim:raise ValueError('provide --qwen-model and --sim-python, or --run-dir with settings.json')
    tasks=a.tasks or sorted(historical_best(load(ROOT/curriculum()['history_file'])))
    if not tasks or len(tasks)!=len(set(tasks)):raise ValueError('unique nonempty task list required')
    known=load(ROOT/'refs/al_artifacts/al/task_splits_50/manifest.json')['tasks']
    if set(tasks)-set(known):raise ValueError('unknown evaluation tasks: '+str(sorted(set(tasks)-set(known))))
    if not 0<=a.video_episodes<=a.episodes:raise ValueError('video episodes must be between zero and episode count')
    directory=a.output_dir.expanduser().resolve()
    jobs=[dict(task=t,label='final_model',model=str(Path(model).expanduser().resolve()),destination=str(directory/t),
               episodes=a.episodes,video_episodes=a.video_episodes,require_baseline=False,
               robotwin=str(ROOT.parent/'RoboTwin-lingbot'),qwen=str(Path(qwen).expanduser().resolve()),
               sim_python=str(Path(sim).expanduser().absolute()))for t in tasks]
    return dict(destination=str(directory),jobs=jobs,concurrency_levels=[a.workers],compare_reference=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir',type=Path)
    source.add_argument('--model-path',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--qwen-model',type=Path)
    p.add_argument('--sim-python',type=Path)
    p.add_argument('--tasks',nargs='+')
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--video-episodes',type=int,default=3)
    p.add_argument('--workers',type=int,choices=range(1,9),default=5)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.episodes<1:p.error('positive episode count required')
    job=build_job(a);print(json.dumps(job,ensure_ascii=False,indent=2))
    if not a.execute:return
    from experiment.official_control.baseline_pipeline import verify_checkpoint
    verify_checkpoint(Path(job['jobs'][0]['model']))
    out=Path(job['destination'])
    if out.exists() or Path(str(out)+'_pipeline').exists():raise FileExistsError('evaluation output already exists')
    out.parent.mkdir(parents=True,exist_ok=True)
    request=Path(str(out)+'_request.json')
    with request.open('x')as stream:json.dump(job,stream,indent=2)
    subprocess.run([sys.executable,str(ROOT/'experiment/official_control/parallel_evaluation.py'),
                    '--job',str(request),'--execute'],check=True)

if __name__=='__main__':main()
