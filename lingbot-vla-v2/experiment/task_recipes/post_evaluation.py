"""After a completed pipeline: historical clean estimate and one shared retention session."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiment.task_recipes.clean_estimate import estimate


def execute(config):
    out=Path(config['destination']);out.mkdir(parents=True,exist_ok=False)
    (out/'job.json').write_text(json.dumps(config,indent=2))
    def state(value,**kw):(out/'status.json').write_text(json.dumps(dict(state=value,**kw),indent=2))
    try:
        baseline=json.loads((ROOT/config['historical_file']).read_text())
        deadline=time.monotonic()+config.get('wait_timeout_seconds',21600)
        pipeline=Path(config['wait_for_pipeline'])
        while True:
            previous=json.loads(pipeline.read_text())
            if previous['state']=='complete':break
            if previous['state'] in ['failed','stopped_by_user']:raise RuntimeError('predecessor '+previous['state'])
            if time.monotonic()>deadline:raise TimeoutError('previous pipeline')
            state('waiting_for_primary_evaluation',previous_state=previous['state']);time.sleep(10)
        current=previous['evaluation']
        if {r['task'] for r in current}!=set(config['primary_tasks']):raise ValueError('primary task mismatch')
        if any(Path(r['model']).resolve()!=Path(config['model']).resolve() for r in current):raise ValueError('primary model mismatch')
        before=estimate(baseline['tasks'],baseline['historical_latest'],current)
        (out/'clean_estimate_before_retention.json').write_text(json.dumps(before,indent=2))
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():
            if time.monotonic()>deadline:raise TimeoutError('GPU busy')
            state('waiting_for_gpu');time.sleep(5)
        from experiment.official_control.baseline_pipeline import verify_checkpoint
        from experiment.official_control.parallel_evaluation import run_jobs
        verify_checkpoint(Path(config['model']))
        directory=out/'retention';directory.mkdir()
        jobs=[dict(task=t,label='R16_retention3',model=config['model'],destination=str(directory/t),episodes=3,
            video_episodes=2,require_baseline=False,robotwin=config['robotwin'],qwen=config['qwen'],sim_python=config['sim_python'])
            for t in config['selection']['chosen_tasks']]
        state('retention_evaluation',tasks=[j['task'] for j in jobs],episodes_per_task=3)
        results=run_jobs(jobs,directory,len(jobs))
        after=estimate(baseline['tasks'],baseline['historical_latest'],current+results)
        (out/'clean_estimate_after_retention.json').write_text(json.dumps(after,indent=2))
        old={r['task']:r for r in baseline['historical_latest']};comparison=[]
        for r in results:
            b=old[r['task']]
            comparison.append(dict(task=r['task'],historical_successes=b['successes'],historical_episodes=b['episodes'],
                historical_source=b['source'],successes=r['successes'],episodes=r['episodes'],seeds=r['seeds'],
                observed_drop=r['successes']*b['episodes']<b['successes']*r['episodes']))
        report=dict(state='complete',before=before,after=after,retention=comparison,
            caution='历史最新逐任务结果跨模型估算，不是当前模型全50任务实测；三回合遗忘检查仅作线索')
        (out/'report.json').write_text(json.dumps(report,indent=2))
        text=f"# clean总体估算与遗忘抽检\n\n历史最新逐任务等权估算：抽检前{before['macro_success_rate']:.2%}，抽检后{after['macro_success_rate']:.2%}。覆盖{after['covered_tasks']}/{after['task_count']}任务，其中本轮模型实测{after['current_model_tasks']}个，其余沿用历史结果；不是当前模型全量实测。\n\n|任务|历史最近|本轮800步|观察到下降|\n|---|---|---|\n"
        for r in comparison:text+=f"|{r['task']}|{r['historical_successes']}/{r['historical_episodes']}|{r['successes']}/{r['episodes']}|{'是' if r['observed_drop'] else '否'}|\n"
        text+='\n每任务仅3回合，变化仅作遗忘线索，不能当作确定的统计结论。\n'
        (out/'report.md').write_text(text);state('complete',report=str(out/'report.json'))
    except BaseException as error:
        state('failed',error=repr(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True)
    a=p.parse_args();execute(json.loads(a.job.read_text()))
