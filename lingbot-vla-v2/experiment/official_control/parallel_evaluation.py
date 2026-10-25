"""Bounded parallel simulators, one FP32 B1 inference service; explicit resource-only fallback."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiment.official_control import evaluate
from experiment.official_control.shared_evaluation import open_session


def run_jobs(jobs,directory,workers,opener=None,executor=evaluate.execute,stopper=evaluate.stop,scheduling='batches'):
    if type(workers) is not int or not 1 <= workers <= 8:raise ValueError('workers must be an integer from 1 to 8')
    if scheduling not in ['batches','rolling']:raise ValueError('invalid scheduling')
    from experiment.official_control.episode_parallel import validate_jobs,aggregate
    validate_jobs(jobs)
    if len({str(Path(j['model']).resolve()) for j in jobs})!=1:raise ValueError('one shared model required')
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
    session=None;results=[];start=time.monotonic();event_lock=threading.Lock()
    def execute(job):
        def event(kind):
            with event_lock:
                with (directory/'task_events.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(event=kind,task=job['task'],job_id=job.get('job_id',job['task']),time=time.time()))+'\n')
        event('start')
        try:return executor(job,shared=session)
        finally:event('end')
    def state(value,**extra):
        (directory/'status.json').write_text(json.dumps(dict(state=value,workers=workers,completed=results,**extra),indent=2))
    try:
        state('starting')
        os.environ['LINGBOT_SESSION_AUDIT']=str(directory/'session_audit.json')
        if opener is None:
            session=open_session(jobs[0],directory,server_entry=ROOT/'experiment/robotwin/session_eval_server.py')
        else:session=opener(jobs[0],directory)
        state('evaluating',server_pid=session['process'].pid)
        offsets=[0] if scheduling=='rolling' else range(0,len(jobs),workers)
        for offset in offsets:
            batch=jobs if scheduling=='rolling' else jobs[offset:offset+workers]
            state('evaluating',server_pid=session['process'].pid,batch=offset//workers+1,
                active_workers=min(workers,len(batch)),scheduling=scheduling,batch_tasks=[j['task'] for j in batch])
            with ThreadPoolExecutor(max_workers=min(workers,len(batch))) as pool:
                futures={}
                remaining=iter(batch)
                def submit(job):
                    dest=Path(job['destination']);dest.mkdir(parents=True,exist_ok=False)
                    (dest/'job.json').write_text(json.dumps(job,indent=2))
                    futures[pool.submit(execute,job)]=job
                for _ in range(min(workers,len(batch))):submit(next(remaining))
                while futures:
                    done,_=wait(futures,return_when=FIRST_COMPLETED)
                    for future in done:
                        job=futures.pop(future)
                        if future.result()!=0:raise RuntimeError('simulation failed: '+job['task'])
                        result=json.loads((Path(job['destination'])/'summary.json').read_text())
                        if result.get('server_pid')!=session['process'].pid or not result.get('shared_server'):
                            raise ValueError('shared inference session mismatch')
                        if result.get('task')!=job['task'] or result.get('state')!='complete':
                            raise ValueError('task result mismatch')
                        if job.get('seed_search') is not None:
                            selected=result['seeds'][0];search=job['seed_search'];delta=selected-job['scene_seed']
                            if delta<0 or delta%search['stride'] or delta//search['stride']>=search['max_candidates']:
                                raise ValueError('invalid inline scene selection')
                            job['scene_seed']=selected;job['expected_seeds']=[selected]
                        results.append(result)
                    for _ in done:
                        job=next(remaining,None)
                        if job is not None:submit(job)
                    (directory/'summary.json').write_text(json.dumps(results,indent=2))
                    state('evaluating',server_pid=session['process'].pid,batch=offset//workers+1,active_workers=len(futures),scheduling=scheduling)
        results=aggregate(results,jobs)
        (directory/'summary.json').write_text(json.dumps(results,indent=2))
        state('complete',elapsed_seconds=time.monotonic()-start,server_pid=session['process'].pid,
            model_loads=session.get('model_loads',1),reused_training_model=session.get('reused_training_model',False))
        return results
    except BaseException as error:
        state('failed',error=repr(error));raise
    finally:
        if session is not None:stopper(session['process'])


def resource_failure(directory):
    pattern=re.compile(r'CUDA out of memory|OutOfDeviceMemory|VK_ERROR_OUT_OF_DEVICE_MEMORY|out of host memory',re.I)
    return any(pattern.search(p.read_text(errors='replace')) for p in directory.rglob('*.log'))


def compare_results(results,jobs):
    comparison=[]
    if {r['task'] for r in results}!={j['task'] for j in jobs}:raise ValueError('comparison task mismatch')
    for result,source in zip(sorted(results,key=lambda r:r['task']),sorted(jobs,key=lambda r:r['task'])):
        baseline=json.loads(Path(source['reference_summary']).read_text())
        matched=(result['seeds']==baseline['seeds'] and result['successes']==baseline['successes']
            and Path(result['model']).resolve()==Path(baseline['model']).resolve())
        comparison.append(dict(task=result['task'],parallel_successes=result['successes'],serial_successes=baseline['successes'],
            seeds=result['seeds'],serial_seeds=baseline['seeds'],matched=matched))
    return comparison


def probe_jobs(job,attempt):
    jobs=[dict(j,destination=str(attempt/j['task'])) for j in job['jobs']]
    if not job.get('compare_reference',True):
        for j in jobs:
            j.pop('reference_summary',None)
            j['require_baseline']=False
    return jobs


def benchmark(job):
    levels=job.get('concurrency_levels',[3,2])
    if (not isinstance(levels,list) or not levels or len(set(levels))!=len(levels)
        or any(type(n) is not int or not 1 <= n <= 8 for n in levels)):
        raise ValueError('concurrency_levels must contain unique integers from 1 to 8')
    directory=Path(job['destination']);directory.mkdir(parents=True,exist_ok=False)
    # Identical per-process CPU limits across serial and parallel attempts.
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
    cancelled=threading.Event()
    def monitor():
        with (directory/'gpu_samples.jsonl').open('x') as stream:
            while not cancelled.is_set():
                result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
                stream.write(json.dumps(dict(time=time.time(),gpu=result.stdout.strip(),returncode=result.returncode))+'\n');stream.flush()
                cancelled.wait(2)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        for index,workers in enumerate(levels):
            attempt=directory/f'parallel_{workers}';attempt.mkdir()
            jobs=probe_jobs(job,attempt)
            try:results=run_jobs(jobs,attempt,workers,scheduling=job.get('scheduling','batches'))
            except Exception:
                if index < len(levels)-1 and resource_failure(attempt):continue
                raise
            # Resource probes may use fewer episodes than the archived reference.
            compare=job.get('compare_reference',True)
            comparison=compare_results(results,job['jobs']) if compare else []
            report=dict(state='complete' if all(x['matched'] for x in comparison) else 'parity_mismatch',
                workers=workers,comparison=comparison,reference_compared=compare,attempt=str(attempt))
            (directory/'benchmark_summary.json').write_text(json.dumps(report,indent=2))
            if not all(x['matched'] for x in comparison):raise ValueError('parallel/serial parity mismatch; do not hide with fallback')
            return
        raise RuntimeError('all concurrency levels exhausted')
    finally:cancelled.set();thread.join(timeout=5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();job=json.loads(args.job.read_text())
    if args.worker:
        status=args.job.parent/'status.json'
        try:
            deadline=time.monotonic()+job.get('wait_timeout_seconds',7200)
            if job.get('wait_for_pipeline'):
                while True:
                    state=json.loads(Path(job['wait_for_pipeline']).read_text())['state']
                    if state=='complete':break
                    if state in ['failed','stopped_by_user']:raise RuntimeError('predecessor did not complete: '+state)
                    status.write_text(json.dumps(dict(state='waiting_for_existing_evaluation',predecessor_state=state),indent=2))
                    if time.monotonic()>deadline:raise TimeoutError('waiting for previous evaluation')
                    time.sleep(10)
            status.write_text(json.dumps(dict(state='benchmarking'),indent=2))
            benchmark(job)
            status.write_text(json.dumps(dict(state='complete',report=job['destination']+'/benchmark_summary.json'),indent=2))
        except BaseException as error:
            status.write_text(json.dumps(dict(state='failed',error=repr(error)),indent=2));raise
        return
    print(json.dumps(job,indent=2))
    if args.execute:
        launch=Path(job['destination']+'_pipeline');launch.mkdir(parents=True,exist_ok=False)
        path=launch/'job.json';path.write_text(json.dumps(job,indent=2))
        with (launch/'pipeline.log').open('x') as stream:
            process=subprocess.Popen(['setsid','nohup',sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--worker'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        print('PARALLEL_EVAL_STARTED',process.pid,str(launch))

if __name__=='__main__':main()
