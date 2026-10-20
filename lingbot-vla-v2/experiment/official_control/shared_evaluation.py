"""指定任务串行闭环，整个列表只加载一次模型。"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiment.official_control import evaluate


def open_session(job, directory):
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip(),'GPU busy'
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    audit=directory/'precision_audit.json'
    env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='0',QWEN3VL_PATH=job['qwen'],
        LINGBOT_INFER_NOISE_SEED='0',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        SETUPTOOLS_SCM_PRETEND_VERSION='0.0.0',PYTHONUNBUFFERED='1',LINGBOT_DEPLOY_FAST_LOAD='0',
        LINGBOT_EVAL_PRECISION_AUDIT=str(audit))
    for key in ('LINGBOTVLA_TRAINING_CONFIG','AL_VLM_ATTENTION','LINGBOT_PRECISION_AUDIT'):env.pop(key,None)
    command=[sys.executable,'-u',str(ROOT/'experiment/robotwin/precision_eval_server.py'),
        '--model_path',job['model'],'--use_length','50','--chunk_ret','True',
        '--use_bf16','False','--use_fp32','True','--use_compile','False','--port',str(port)]
    log=(directory/'server.log').open('x')
    try:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    finally:log.close()
    session=dict(model=job['model'],process=process,port=port,audit=audit)
    (directory/'session.json').write_text(json.dumps(dict(server_pid=process.pid,port=port,model=job['model'],command=command),indent=2))
    start=time.monotonic()
    try:
        while True:
            assert process.poll() is None,'inference server exited'
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=1) as r:
                    if r.status==200:break
            except Exception:pass
            if time.monotonic()-start>600:raise TimeoutError('shared inference startup')
            time.sleep(2)
    except BaseException:
        evaluate.stop(process);raise
    return session


def run_jobs(jobs,directory,opener=open_session,executor=evaluate.execute,stopper=evaluate.stop):
    assert jobs and len({str(Path(j['model']).resolve()) for j in jobs})==1,'one shared session requires one model'
    assert len({j['task'] for j in jobs})==len(jobs),'duplicate tasks'
    session=None;results=[]
    def state(value,**extra):
        (directory/'status.json').write_text(json.dumps(dict(state=value,completed=results,**extra),indent=2))
    try:
        state('starting')
        session=opener(jobs[0],directory)
        for job in jobs:
            dest=Path(job['destination']);dest.mkdir(exist_ok=False)
            (dest/'job.json').write_text(json.dumps(job,indent=2))
            state('evaluating',task=job['task'],server_pid=session['process'].pid)
            assert executor(job,shared=session)==0
            result=json.loads((dest/'summary.json').read_text())
            assert result['server_pid']==session['process'].pid and result['shared_server']
            results.append(result)
            (directory/'summary.json').write_text(json.dumps(results,indent=2))
        state('complete',server_pid=session['process'].pid,model_loads=1)
        return 0
    except BaseException as error:
        state('failed',error=repr(error));raise
    finally:
        if session is not None:stopper(session['process'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();job=json.loads(a.job.read_text())
    directory=Path(job['destination'])
    if a.worker:return run_jobs(job['jobs'],directory)
    print(json.dumps(job,indent=2))
    if a.execute:
        directory.mkdir(parents=True,exist_ok=False)
        with (directory/'pipeline.log').open('x') as f:
            process=subprocess.Popen(['setsid','nohup',sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(a.job.resolve()),'--worker'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        print('SHARED_EVALUATION_STARTED',process.pid,str(directory))

if __name__=='__main__':raise SystemExit(main())
