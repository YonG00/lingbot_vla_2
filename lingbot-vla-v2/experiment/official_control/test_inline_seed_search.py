import contextlib,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from experiment.official_control.episode_parallel import compile_episode_client,aggregate,retryable_expert_error
from experiment.official_control import evaluate
from experiment.official_control.model_benchmark import jobs_for

CLIENT='''
seed=0
st_seed=100000 * (1 + seed)
now_id=0
now_seed=st_seed
succ_seed=0
test_num=1
while succ_seed < test_num:
    if now_seed in rejected:
        now_seed+=1
        continue
    result=generate_episode_descriptions('task',[],test_num)
    accepted.append(now_seed)
    succ_seed+=1
    now_seed+=1
'''
class InlineSeedTests(unittest.TestCase):
    def execute(self,start,index,rejected,limit=4):
        ns=dict(rejected=rejected,accepted=[],generate_episode_descriptions=lambda *a:None)
        exec(compile_episode_client(CLIENT,'client.py',start,4,index,dict(stride=4,max_candidates=limit)),ns)
        return ns['accepted']
    def test_known_planner_failures_only(self):
        for filename,function,kind,message,wanted in [
            ('/envs/_base_task.py','grasp_actor','AssertionError','target_pose cannot be None for move action.',True),
            ('/envs/other.py','play_once','AssertionError','target_pose cannot be None for move action.',False),
            ('/envs/put_bottles_dustbin.py','play_once','IndexError','list index out of range',True),
            ('/envs/other.py','play_once','IndexError','list index out of range',False),
            ('/envs/_base_task.py','grasp_actor','RuntimeError','CUDA out of memory',False)]:
            ns={};exec(compile('def '+function+'():\n    raise '+kind+'('+repr(message)+')',filename,'exec'),ns)
            try:ns[function]()
            except Exception as e:self.assertEqual(retryable_expert_error(e),wanted)

    def test_unstable_policy_scene_rebuild_retries_before_inference(self):
        class UnStableError(Exception):pass
        class Task:
            calls={}
            def setup_demo(self,seed):
                self.calls[seed]=self.calls.get(seed,0)+1
                if seed==100000 and self.calls[seed]==2:raise UnStableError()
            def play_once(self):pass
            def close_env(self):pass
        source="""
seed=0
st_seed=100000 * (1 + seed)
now_id=0
now_seed=st_seed
succ_seed=0
test_num=1
suc_test_seed_list=[]
args={}
while succ_seed < test_num:
    render_freq=0
    try:
        TASK_ENV.setup_demo(seed=now_seed)
        TASK_ENV.play_once()
    except Exception as e:
        raise
    succ_seed += 1
    suc_test_seed_list.append(now_seed)
    TASK_ENV.setup_demo(seed=now_seed)
    result=generate_episode_descriptions('task',[],test_num)
    accepted.append(now_seed)
    now_seed += 1
"""
        ns=dict(TASK_ENV=Task(),UnStableError=UnStableError,accepted=[],generate_episode_descriptions=lambda *a:None)
        exec(compile_episode_client(source,'client.py',100000,4,0,dict(stride=4,max_candidates=5)),ns)
        self.assertEqual(ns['accepted'],[100004]);self.assertEqual(ns['suc_test_seed_list'],[100004])

    def test_retry_in_same_episode_and_disjoint_sequences(self):
        result=[self.execute(100000+i,i,{100000,100004,100001})[0] for i in range(4)]
        self.assertEqual(result,[100008,100005,100002,100003])
        self.assertEqual(len(set(result)),4)
    def test_exhaustion_not_infinite_loop(self):
        with self.assertRaisesRegex(RuntimeError,'limit exhausted'):
            self.execute(100000,0,{100000,100004},limit=2)
    def test_fixed_mode_still_rejects(self):
        with self.assertRaisesRegex(RuntimeError,'fixed episode seed rejected'):
            exec(compile_episode_client(CLIENT,'client.py',100000,4),dict(rejected={100000},accepted=[],generate_episode_descriptions=lambda *a:None))
    def test_invalid_stride_rejected(self):
        with self.assertRaises(ValueError):compile_episode_client(CLIENT,'client.py',1,4,seed_search=dict(stride=0,max_candidates=3))
    def test_explicit_seeds_disable_search(self):
        c=dict(model='/m',robotwin='/r',qwen='/q',sim_python='/p',seed_selection='inline',scene_seeds={'demo_clean/x':[10,20]})
        item=dict(task='x',task_config='demo_clean',key='demo_clean/x',episodes=2,video_episodes=0,seeds=[10,20])
        self.assertTrue(all('seed_search' not in j for j in jobs_for(c,item,Path('/tmp/x'))))
        c['scene_seeds']={};jobs=jobs_for(c,item,Path('/tmp/y'))
        self.assertEqual([j['scene_seed'] for j in jobs],[100000,100001]);self.assertTrue(all(j['seed_search']['stride']==2 for j in jobs))
    def test_parallel_aggregation_uses_accepted_not_candidate_seeds(self):
        from experiment.official_control.parallel_evaluation import run_jobs
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            c=dict(model='/m',robotwin='/r',qwen='/q',sim_python='/p',seed_selection='inline')
            item=dict(task='x',task_config='demo_clean',key='demo_clean/x',episodes=2,video_episodes=0,seeds=[100000,100001])
            jobs=jobs_for(c,item,root/'task')
            process=SimpleNamespace(pid=42,poll=lambda:None)
            def execute(job,shared=None):
                seed=job['scene_seed']+job['seed_search']['stride']
                row=dict(state='complete',task='x',task_config='demo_clean',model='/m',
                    episodes=1,successes=0,success_rate=0.0,seeds=[seed],scene_seed=seed,
                    episode_index=job['episode_index'],job_id=job['job_id'],oidn_errors=0,
                    recorded_episodes=0,server_pid=42,shared_server=True,dtype='float32',action_chunk=50,client_sha256='hash')
                (Path(job['destination'])/'summary.json').write_text(json.dumps(row));return 0
            result=run_jobs(jobs,root,2,opener=lambda *_:dict(process=process),executor=execute,stopper=lambda _:None)
            self.assertEqual(result[0]['seeds'],[100002,100003])
            self.assertEqual(result[0]['episodes'],2)

    def test_real_execute_uses_actual_selected_seed_and_records_search(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'robo/script').mkdir(parents=True);(root/'robo/script/eval_policy_client_lingbotvla.py').write_text('client')
            dest=root/'out';dest.mkdir();audit=root/'precision.json';audit.write_text(json.dumps(dict(parameter_numel_by_dtype={'torch.float32':1},forward_samples={'x':{}})))
            shared=dict(model='/m',process=SimpleNamespace(pid=42,poll=lambda:None),port=123,audit=audit)
            job=dict(task='x',label='test',episodes=1,video_episodes=0,model='/m',destination=str(dest),robotwin=str(root/'robo'),qwen='/q',sim_python='/python',require_baseline=False,scene_seed=100002,expected_seeds=[100002],episode_index=2,description_episode_count=4,job_id='x:episode:2',seed_search=dict(stride=4,max_candidates=5))
            def launch(cmd,**kw):
                self.assertEqual(json.loads(kw['env']['LINGBOT_EPISODE_SEED_SEARCH']),job['seed_search'])
                result=dest/'results/x';result.mkdir(parents=True);(result/'_result.txt').write_text('1.0')
                kw['stdout'].write('OIDN 2.5.1 CUDA=True\nSuccess rate: 1/1 current seed: 100006\n')
                return SimpleNamespace(wait=lambda **kw:0,poll=lambda:0)
            with patch.object(evaluate.subprocess,'Popen',side_effect=launch),patch.object(evaluate.subprocess,'check_output',return_value='/site'),patch.object(evaluate.urllib.request,'urlopen',return_value=contextlib.nullcontext(SimpleNamespace(status=200))),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate.execute(job,shared),0)
            result=json.loads((dest/'summary.json').read_text())
            self.assertEqual(result['seeds'],[100006]);self.assertEqual(result['scene_seed'],100006)
            self.assertEqual(result['seed_search']['start_seed'],100002)

if __name__=='__main__':unittest.main()
