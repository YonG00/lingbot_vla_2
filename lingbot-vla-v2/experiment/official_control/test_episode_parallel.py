import json,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from experiment.official_control.episode_parallel import split_jobs,workers_for,aggregate,compile_episode_client,validate_jobs
from experiment.official_control.parallel_evaluation import run_jobs
from experiment.task_recipes.schema import validate
from experiment.task_recipes.test_contracts import fixture

CLIENT="""seed=0
st_seed=100000 * (1 + seed)
now_id=0
now_seed=st_seed
succ_seed=0
test_num=1
while succ_seed < test_num:
    if reject:
        now_seed+=1
        continue
    result=generate_episode_descriptions('task',[],test_num)
    succ_seed+=1
"""
class EpisodeParallelTests(unittest.TestCase):
 def config(self):return dict(tasks=['bowls'],max_parallel_tasks=1,episodes_per_task=3,single_task_episode_workers=3)
 def jobs(self,root):return split_jobs([dict(task='bowls',episodes=3,model='/model',label='test',video_episodes=2,destination=str(root/'bowls'))],self.config())
 def result(self,j):return dict(state='complete',task=j['task'],model=j['model'],label='test',job_id=j['job_id'],episode_index=j['episode_index'],scene_seed=j['scene_seed'],episodes=1,successes=int(j['episode_index']==1),seeds=[j['scene_seed']],oidn_errors=0,recorded_episodes=j['video_episodes'],server_pid=42,shared_server=True,dtype='float32',action_chunk=50,client_sha256='hash')
 def test_shards_unique_seed_output_and_video_limit(self):
  jobs=self.jobs(Path('/tmp/out'));validate_jobs(jobs)
  self.assertEqual([j['scene_seed'] for j in jobs],[100000,100001,100002]);self.assertEqual([j['video_episodes'] for j in jobs],[1,1,0]);self.assertEqual(len({j['destination'] for j in jobs}),3)
  self.assertTrue(all(j['description_episode_count']==3 for j in jobs))
 def test_only_single_task_and_cap_workers(self):
  c=self.config();c['tasks']=['a','b'];c['max_parallel_tasks']=5
  jobs=[dict(task=t) for t in c['tasks']];self.assertIs(split_jobs(jobs,c),jobs);self.assertEqual(workers_for(c),5)
  c=self.config();c['single_task_episode_workers']=8;self.assertEqual(workers_for(c),3)
  c['episodes_per_task']=1;job=[dict(task='bowls',episodes=1)];self.assertIs(split_jobs(job,c),job);self.assertEqual(workers_for(c),1)
 def test_schema_rejects_invalid_workers_and_seeds(self):
  for value in [0,9,True,1.5]:
   c=fixture('place_container_plate_200');c['evaluation']['single_task_episode_workers']=value
   with self.assertRaises(ValueError):validate(c)
  c=fixture('place_container_plate_200');c['evaluation'].update(single_task_episode_workers=3,single_task_episode_seeds=[1,1,2])
  with self.assertRaises(ValueError):validate(c)
 def test_duplicates_not_silently_accepted(self):
  jobs=self.jobs(Path('/tmp/out'));jobs[1]['scene_seed']=jobs[0]['scene_seed']
  with self.assertRaises(ValueError):validate_jobs(jobs)
  with self.assertRaises(ValueError):validate_jobs([dict(task='a',destination='/a'),dict(task='a',destination='/b')])
 def test_client_seed_index_description_and_invalid_seed_failure(self):
  calls=[];env=dict(reject=False,generate_episode_descriptions=lambda *args:calls.append(args[2]))
  exec(compile_episode_client(CLIENT,'fake',100002,3,2),env)
  self.assertEqual(env['st_seed'],100002);self.assertEqual(env['now_id'],2);self.assertEqual(calls,[3])
  with self.assertRaises(RuntimeError):exec(compile_episode_client(CLIENT,'fake',100002,3,2),dict(reject=True))
  with self.assertRaises(ValueError):compile_episode_client(CLIENT.replace('100000 * (1 + seed)','200000'), 'fake',1,3)
 def test_three_episode_processes_overlap_one_model_and_aggregate(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);jobs=self.jobs(root);barrier=threading.Barrier(3);lifecycle=[]
   def execute(j,shared):
    barrier.wait(timeout=3);result=self.result(j);dest=Path(j['destination']);(dest/'summary.json').write_text(json.dumps(result))
    if j['video_episodes']:
     video=dest/'results/bowls/episode0_global.mp4';video.parent.mkdir(parents=True);video.write_bytes(b'video')
    return 0
   def opener(*args):lifecycle.append('open');return dict(process=SimpleNamespace(pid=42),model='/model')
   result=run_jobs(jobs,root,3,opener,execute,lambda _:lifecycle.append('stop'))
   self.assertEqual(lifecycle,['open','stop']);self.assertEqual(result[0]['episodes'],3);self.assertEqual(result[0]['successes'],1);self.assertEqual(result[0]['seeds'],[100000,100001,100002]);self.assertEqual(result[0]['recorded_episodes'],2)
   self.assertTrue((root/'bowls/results/bowls/episode1_global.mp4').exists())
 def test_bad_or_missing_episode_refuses_complete(self):
  with tempfile.TemporaryDirectory() as d:
   jobs=self.jobs(Path(d));rows=[self.result(j) for j in jobs];rows[0]['seeds']=[999]
   with self.assertRaises(ValueError):aggregate(rows,jobs)
   with self.assertRaises(ValueError):aggregate(rows[:2],jobs)

 def test_resident_job_split_and_canonical_read_completed(self):
  from experiment.task_recipes.resident_evaluation import jobs_for,read_completed
  c=fixture('place_container_plate_200')
  c['evaluation'].update(tasks=['bowls'],episodes_per_task=3,shared_model_session=True,reuse_training_model=True,single_task_episode_workers=3,video_episodes=0)
  c=validate(c)
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);directory=out/'evaluation_200';directory.mkdir()
   jobs=jobs_for(c,out,200,dict(model=dict(tokenizer_path='/tmp/qwen')))
   self.assertEqual(len(jobs),3)
   results=aggregate([self.result(j) for j in jobs],jobs)
   (directory/'summary.json').write_text(json.dumps(results))
   (directory/'status.json').write_text(json.dumps(dict(state='complete',reused_training_model=True,model_loads=0)))
   (directory/'handoff.json').write_text(json.dumps(dict(training_pid=42,server_pid=42,weight_loads=0)))
   self.assertEqual(read_completed(directory,c,200),results)
 def test_one_explicit_scene_seed_aggregates_without_multiple_workers(self):
  c=self.config();c.update(episodes_per_task=1,single_task_episode_seeds=[100005])
  jobs=split_jobs([dict(task='bowls',episodes=1,video_episodes=0,model='/model',label='test',destination='/tmp/bowls')],c)
  self.assertEqual(workers_for(c),1);self.assertEqual(jobs[0]['scene_seed'],100005)
