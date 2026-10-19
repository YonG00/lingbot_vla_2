"""联合白名单→实际resolver→混合采样→消费统计→调度；无模型。"""
from collections import Counter
import contextlib
from dataclasses import replace
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch
import pyarrow as pa
import pyarrow.parquet as pq
from mouse_pad_training import test_recipe as fixtures
from lingbotvla.auto_learning.catalog import TaskCatalog,catalog_from_task_split
from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.ports import Backend,TrainRequest,TrainResult
from lingbotvla.auto_learning.types import TrajectoryMetrics
from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
from lingbotvla.auto_learning.real.sampler import AutoLearnSampler
from lingbotvla.auto_learning.sampling.sampler import BatchSampler
from lingbotvla.auto_learning.sampling.auxiliary import AuxiliaryPool,AuxiliaryBatchSampler,preflight_pool,primary_quota,full_catalog_for_auxiliary

class AuxiliaryTests(unittest.TestCase):
    def setUp(self):
        base=fixtures.MouseRecipeTests();base.setUp();self.addCleanup(base.doCleanups)
        self.f,self.args=base.fixture,base.args;self.root=self.args.dataset_root
        self.args.train_all_episodes=True;self.args.micro=20;self.args.gas=1;self.args.ee_position_weight=10
        (self.root/'meta/episodes').mkdir();(self.root/'data').mkdir()
        self.lengths={**{ep:32 for ep in range(800,850)},2300:60,2353:300}
        rows=[];meta=[];labels=[];pos=0
        for idx,(ep,n) in enumerate(self.lengths.items()):
            text='Place mouse on pad' if ep<850 else 'Move mouse to '+('stand' if ep==2300 else 'cabinet')
            labels.append({'task_index':idx,'__index_level_0__':text})
            meta.append(dict(episode_index=ep,length=n,tasks=[text],dataset_from_index=pos,dataset_to_index=pos+n))
            for i in range(n):rows.append(dict(index=pos+i,episode_index=ep,frame_index=i,timestamp=i/10,task_index=idx,**{'observation.state':[0.0]*14}))
            pos+=n
        self.meta={m['episode_index']:m for m in meta}
        for file,contents in [('data/file.parquet',rows),('meta/episodes/file.parquet',meta),('meta/tasks.parquet',labels)]:pq.write_table(pa.Table.from_pylist(contents),self.root/file)
        (self.root/'meta/info.json').write_text(json.dumps(dict(fps=10,total_episodes=52,total_frames=pos,features={})))
        self.approved=self.f.root/'approved';self.approved.mkdir()
        files=[self.root/'meta/info.json',self.root/'meta/tasks.parquet',self.root/'meta/episodes/file.parquet',self.args.manifest]
        stamp=self.root/'data/file.parquet'
        identity=dict(content=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],data=[dict(path='data/file.parquet',size=stamp.stat().st_size,mtime_ns=stamp.stat().st_mtime_ns)],videos=[])
        fp=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();records=[]
        for ep,task,side in [(2300,'place_object_stand','right'),(2353,'put_object_cabinet','left')]:
            m=self.meta[ep];records.append(dict(episode_index=ep,source_task=task,length=m['length'],instructions=m['tasks'],absolute_range=[m['dataset_from_index'],m['dataset_to_index']],review=dict(decision='approved',reviewer='fixture',evidence='review fixture',manipulated_target_verified=True,demonstration_success_verified=True,mouse_arm=side,control_grasp_candidates=[dict(side=side,close_start=10,window=[8,14])],phase_windows={})))
        (self.approved/'source_identity.json').write_text(json.dumps(identity));(self.approved/'approved.json').write_text(json.dumps(dict(training_ready=True,dataset_fingerprint=fp,episodes=records)))
        self.args.auxiliary_manifest=self.approved/'approved.json';self.args.auxiliary_primary_ratio=.7;self.args.auxiliary_grasp_weight=3.;self.args.auxiliary_approach_seconds=1.
        with contextlib.redirect_stdout(io.StringIO()):_,self.plan,_=fixtures.recipe.build_job(self.args)
        self.cfg=AutoLearningConfig(**self.plan['config']);self.full=full_catalog_for_auxiliary(self.args.manifest)
        primary=TaskCatalog({'place_mouse_pad':self.full.entry('place_mouse_pad')},meta=self.full.meta,task_order=self.full.task_order).for_training(train_all_episodes=True)
        self.pool=AuxiliaryPool(self.cfg,self.full,primary,dataset_root=self.root,manifest_path=self.args.manifest);preflight_pool(self.pool,self.root)
        self.resolver=self.pool.resolver;self.catalog=primary.attach_samples(self.resolver)
    def sampler(self):return AuxiliaryBatchSampler(self.cfg,self.resolver,self.catalog,random.Random(3),self.pool)
    def request(self,n=20):
        ids=self.catalog.entry('place_mouse_pad').train_sample_ids
        return TrainRequest('place_mouse_pad',probs={i:1/len(ids) for i in ids},batch_size=n,new_slots=n)
    def test_joint_preflight_and_no_outputs(self):
        self.assertEqual(self.plan['train_ids'],sorted(self.lengths));self.assertEqual(self.plan['config']['task_names'],['place_mouse_pad'])
        self.assertEqual(self.plan['auxiliary']['joint_frames'],1960);self.assertEqual(self.plan['auxiliary']['episodes'],2)
        self.assertEqual(self.plan['auxiliary']['local_batch_primary_step0'],14);self.assertFalse(self.args.output_root.exists())
    def test_actual_batches_balance_weights_and_original_sources(self):
        b=self.sampler();p=b.prepare(self.request());eps=Counter();marked=0
        for step in range(1000):
            comp=b.build(p,logical_step=step);self.assertEqual((len(comp.new),len(comp.old),len(comp.auxiliary)),(14,0,6))
            self.assertEqual(len({r.sample_id for r in comp.refs}),20)
            for r in comp.auxiliary:
                self.assertEqual(self.pool.records[r.episode_id]['source_task'],r.task);self.assertEqual(self.resolver.ref_to_local(r),r.sample_id)
                eps[r.episode_id]+=1;marked+=r.sample_id in self.pool.grasp_ids
        self.assertLess(abs(eps[2300]/6000-.5),.035)
        self.assertLess(abs(marked/6000-.5*(45/90+45/330)),.035)
        self.assertGreater(marked/6000,.5*(15/60+15/300)+.10)
    def test_fractional_global_quota_dp_and_gas(self):
        for local,dp in [(20,1),(10,2),(3,4),(1,3)]:
            total=0
            for step in range(100):
                ns=[primary_quota(local,dp,r,.7,step) for r in range(dp)];self.assertTrue(all(0<=n<=local for n in ns));total+=sum(ns)
                self.assertEqual(total,(step+1)*local*dp*7//10)
        for rank in range(2):
            self.cfg._ratio_dp_rank=rank;self.cfg._ratio_dp_size=2;b=self.sampler();c=b.build(b.prepare(self.request(10)),logical_step=9)
            self.assertEqual((len(c.new),len(c.auxiliary)),(7,3))  # 每rank可拆micro2/GAS5
    def test_grasp_weight_one_keeps_uniform_frames_and_episode_quota(self):
        self.pool.grasp_weight=1.;self.pool.bind(self.resolver)
        b=self.sampler();p=b.prepare(self.request());marked=0;counts=Counter()
        for step in range(1000):
            for r in b.build(p,logical_step=step).auxiliary:
                marked+=r.sample_id in self.pool.grasp_ids;counts[r.episode_id]+=1
        self.assertLess(abs(marked/6000-.5*(15/60+15/300)),.035)
        self.assertLess(abs(counts[2300]/6000-.5),.035)
    def test_incompatible_modes_fail_before_preparation(self):
        for change in [dict(train_all_episodes=False),dict(replay_task_names=['place_object_stand']),
                       dict(new_ratio=.7),dict(auxiliary_primary_ratio=1.),
                       dict(auxiliary_grasp_weight=float('nan')),dict(auxiliary_approach_seconds=-1)]:
            with self.subTest(change=change),self.assertRaises(ValueError):replace(self.cfg,**change)
    def test_consumed_stats_prefetch_and_resume(self):
        self.cfg._ratio_global_batch_size=20;self.cfg._ratio_dp_size=1
        s=AutoLearnSampler(self.sampler(),batch_size=20);s.set_request(self.request());it=iter(s)
        for _ in range(100):next(it)
        st=s.stats_upto(3);self.assertEqual((st.samples_seen,st.n_new,st.n_auxiliary),(60,42,18));self.assertEqual(st.check_invariants(),[])
        self.assertEqual(sum(st.auxiliary_phase_counts.values()),18)
        saved=s.state_dict();s.load_state_dict(saved)
        with self.assertRaises(ValueError):s.load_state_dict({**saved,'auxiliary':{**saved['auxiliary'],'grasp_weight':9}})
        first=[r.sample_id for r in s.compositions[0].refs];s.set_request(self.request());it=iter(s);self.assertEqual([next(it) for _ in range(20)],first)
    def test_invalid_approval_source_and_target_arm(self):
        p=self.approved/'approved.json';original=p.read_text()
        for kind in ['pending','not_ready','duplicate','wrong_task']:
            raw=json.loads(original)
            if kind=='pending':raw['episodes'][0]['review']['decision']='pending'
            elif kind=='not_ready':raw['training_ready']=False
            elif kind=='duplicate':raw['episodes'].append(raw['episodes'][0])
            else:raw['episodes'][0]['source_task']='place_mouse_pad'
            p.write_text(json.dumps(raw))
            with self.subTest(kind=kind),self.assertRaises(ValueError):AuxiliaryPool(self.cfg,self.full,self.catalog,dataset_root=self.root,manifest_path=self.args.manifest)
        p.write_text(original);self.pool.records[2300]['review']['control_grasp_candidates'][0]['side']='left'
        with self.assertRaisesRegex(ValueError,'non-target arm'):self.pool.bind(self.resolver)
        (self.root/'meta/info.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'source content changed'):AuxiliaryPool(self.cfg,self.full,self.catalog,dataset_root=self.root,manifest_path=self.args.manifest)
    def test_disabled_has_legacy_rng_and_batch(self):
        cfg=replace(self.cfg,auxiliary_manifest=None);rng=random.Random(3)
        before=rng.getstate();a=BatchSampler(cfg,self.resolver,self.catalog,rng);pa=a.prepare(self.request())
        for step in range(10):
            c=a.build(pa,logical_step=step)
            self.assertEqual(len(c.new),20);self.assertEqual(c.auxiliary,[])
            self.assertEqual({r.task for r in c.refs},{'place_mouse_pad'})
        self.assertEqual(rng.getstate(),before)
        from lingbotvla.auto_learning.state.persistence import config_fingerprint
        self.assertNotIn('auxiliary',config_fingerprint(cfg,['place_mouse_pad']))
        self.assertIn('auxiliary',config_fingerprint(self.cfg,['place_mouse_pad']))
    def test_scheduler_500_steps_counts_and_primary_only_monitors(self):
        self.cfg.hardness_scan_enabled=False;b=self.sampler();calls=[]
        def evaluate(task,split,ids):
            self.assertEqual(task,'place_mouse_pad');self.assertEqual(split,'train_monitor');calls.append(tuple(ids))
            return TrajectoryMetrics(task,split,list(ids),{i:.01 for i in ids},.01,0.,None,None,False,4)
        def train(req,steps):
            s=AutoLearnSampler(b,batch_size=20);s.set_request(req);it=iter(s)
            for _ in range((steps+2)*20):next(it)
            st=s.stats_upto(steps)
            return TrainResult(loss=.1,steps=steps,samples_seen=st.samples_seen,new_slot_counts=dict(st.new_slot_counts_by_task),auxiliary_slot_counts=dict(st.auxiliary_slot_counts_by_task),auxiliary_episode_counts=dict(st.auxiliary_episode_counts),auxiliary_phase_counts=dict(st.auxiliary_phase_counts),batches_built=steps,unique_batches=st.unique_batches)
        backend=Backend(catalog=self.catalog,resolver=self.resolver,evaluator=NS(evaluate=evaluate,baseline_mse=lambda _:0.),scorer=NS(score=Mock(side_effect=AssertionError('no scoring'))),trainer=NS(train_steps=train))
        sched=Scheduler(backend,self.cfg,logger=Mock());sched.run()
        self.assertEqual((sched.state.global_step,sched.state.global_samples_seen),(500,10000));self.assertEqual(len(calls),6)
        self.assertTrue(all(ids==(800,801,802,803) for ids in calls))
        self.assertEqual(sum(r['n_new'] for r in sched.metrics_rows),7000);self.assertEqual(sum(r['n_auxiliary'] for r in sched.metrics_rows),3000)
        self.assertEqual(sched.registry.summary()['pass'],0)
    def test_real_worker_freezes_manifest(self):
        self.args.steps=750;self.args.hf_save_steps=750
        job,plan,snapshot=fixtures.recipe.build_job(self.args);snapshot.parent.mkdir();snapshot.write_text(json.dumps(plan['training_config']));al=snapshot.with_suffix('.al.json');al.write_text(json.dumps(plan['config']))
        cmd=job['command'][2:];cmd[cmd.index('--al-config')+1]=str(al);cmd[cmd.index('--execute')]='--prepare-only'
        with contextlib.redirect_stdout(io.StringIO()):fixtures.recipe.launcher.main(cmd)
        out=Path(job['output_dir']);cfg=json.loads((out/'al_config.json').read_text());self.assertEqual(Path(cfg['auxiliary_manifest']),out/'auxiliary/approved.json')
        self.assertEqual((out/'auxiliary/approved.json').read_bytes(),(self.approved/'approved.json').read_bytes());self.assertEqual(json.loads((out/'train_ids.json').read_text()),sorted(self.lengths))
        actual=json.loads((out/'provenance.json').read_text())
        self.assertEqual(actual['command'][actual['command'].index('--train.save_steps')+1],'750')
        self.assertEqual(actual['training_config']['train']['save_steps'],750)

    def test_real_loop_hook_drains_auxiliary_counts_without_prefetch(self):
        from lingbotvla.auto_learning.real.hook import AutoLearnLoopHook
        self.cfg.hardness_scan_enabled=False
        def evaluate(task,split,ids):
            self.assertEqual(split,'train_monitor')
            return TrajectoryMetrics(task,split,list(ids),{i:.01 for i in ids},.01,0.,None,None,False,4)
        backend=Backend(catalog=self.catalog,resolver=self.resolver,
            evaluator=NS(evaluate=evaluate,baseline_mse=lambda _:0.),
            scorer=NS(score=Mock(side_effect=AssertionError('no scoring'))),
            trainer=NS(train_steps=Mock(side_effect=AssertionError('deferred hook must own batches'))))
        sched=Scheduler(backend,self.cfg,logger=Mock())
        real=AutoLearnSampler(self.sampler(),batch_size=20)
        hook=AutoLearnLoopHook(scheduler=sched,sampler=real,cfg=self.cfg)
        for step in range(500):
            directive=hook.on_step_begin(step)
            self.assertFalse(directive.finished)
            if directive.rebuild_iterator:it=iter(real)
            for _ in range(20):next(it)
            if hook.step_in_unit+1==hook.unit_steps:
                for _ in range(40):next(it)
            hook.on_step_end(step+1,.1)
        self.assertTrue(hook.on_step_begin(500).finished)
        self.assertEqual(sched.state.global_samples_seen,10000)
        self.assertEqual(sum(r['n_auxiliary'] for r in sched.metrics_rows),3000)
        self.assertEqual(sum(r['n_new'] for r in sched.metrics_rows),7000)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "训练装配依赖torch，在训练机CPU验证")
    def test_real_parts_assembly_keeps_scheduler_primary_and_joint_mapping(self):
        from lingbotvla.auto_learning.real.build import build_auto_learning_parts
        rows=pq.read_table(self.root/'data/file.parquet').to_pylist()
        labels=pq.read_table(self.root/'meta/tasks.parquet').to_pylist()
        dataset=NS(root=self.root,hf_dataset={key:[r[key] for r in rows]
            for key in ('episode_index','index','task_index')},
            meta=NS(episodes=self.meta,tasks=[r['__index_level_0__'] for r in labels]))
        self.cfg.data_audit_file=str(self.f.root/'parts_audit.json')
        path=self.f.root/'parts_config.json';path.write_text(json.dumps(self.cfg.to_dict()))
        with patch('lingbotvla.auto_learning.real.build.catalog_from_task_split',
                   side_effect=lambda path,**kw:full_catalog_for_auxiliary(path)):
            parts=build_auto_learning_parts(config_path=str(path),manifest_path=str(self.args.manifest),train_dataset=dataset)
        self.assertEqual(parts.catalog.task_names(),['place_mouse_pad'])
        self.assertEqual(len(parts.resolver),1960)
        self.assertEqual(len(parts.auxiliary_pool.allowed),360)

    def test_auxiliary_entry_restart_defaults_and_explicit_overrides(self):
        path=Path(fixtures.recipe.__file__).parent/'20261019_place_mouse_pad_mouse_aux.py'
        spec=importlib.util.spec_from_file_location('mouse_aux_entry_test',path)
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        with patch.object(entry.recipe,'main',side_effect=fixtures.recipe.parser().parse_args):
            defaults=entry.main([])
            override=entry.main(['--steps','40','--no-gradient-checkpointing','--micro','32','--no-shutdown'])
        self.assertEqual((defaults.steps,defaults.hf_save_steps),(500,250))
        self.assertTrue(defaults.gradient_checkpointing)
        self.assertEqual((defaults.micro,defaults.gas),(48,1))
        self.assertEqual((defaults.ee_position_weight,defaults.alignment_loss_scale),(12,1.25))
        self.assertTrue(defaults.shutdown);self.assertFalse(override.shutdown)
        self.assertEqual((override.steps,override.gradient_checkpointing,override.micro),(40,False,32))
        self.args.steps=defaults.steps;self.args.gradient_checkpointing=defaults.gradient_checkpointing
        self.args.hf_save_steps=defaults.hf_save_steps
        self.args.micro=32
        job,plan,_=fixtures.recipe.build_job(self.args)
        self.assertEqual(job['steps'],500)
        cmd=plan['command']
        self.assertEqual(cmd[cmd.index('--train.save_steps')+1],'250')
        self.assertEqual(cmd[cmd.index('--train.enable_gradient_checkpointing')+1],'true')
        self.assertEqual(plan['config']['batch_size'],32)

    def test_alignment_scale_survives_real_worker_rebuild(self):
        source=fixtures.recipe.launcher.read_training_config(self.args.train_config)
        source.setdefault('train',{})['align_params']={
            'depth':{'use_future_depth':True},'use_future_video':True,
            'depth_loss_weight':.004,'future_depth_loss_weight':.004,
            'video':{'future_video_loss_weight':.004,'cosine_loss_weight':.2}}
        self.args.train_config.write_text(json.dumps(source))
        self.args.alignment_loss_scale=1.25;self.args.ee_position_weight=12
        job,plan,snapshot=fixtures.recipe.build_job(self.args)
        snapshot.parent.mkdir();snapshot.write_text(json.dumps(plan['training_config']))
        al=snapshot.with_suffix('.al.json');al.write_text(json.dumps(plan['config']))
        cmd=job['command'][2:];cmd[cmd.index('--al-config')+1]=str(al);cmd[cmd.index('--execute')]='--prepare-only'
        with contextlib.redirect_stdout(io.StringIO()):fixtures.recipe.launcher.main(cmd)
        actual=json.loads((Path(job['output_dir'])/'trainer_config.json').read_text())['train']['align_params']
        self.assertEqual((actual['depth_loss_weight'],actual['future_depth_loss_weight'],actual['video']['future_video_loss_weight']),(.005,.005,.005))
        self.assertEqual(actual['video']['cosine_loss_weight'],.2)
        self.assertEqual(json.loads(self.args.train_config.read_text())['train']['align_params']['depth_loss_weight'],.004)
        command=json.loads((Path(job['output_dir'])/'provenance.json').read_text())['command']
        self.assertEqual(float(command[command.index('--train.ee_fk_position_loss_weight')+1]),12)
        for scale in (0,float('nan')):
            self.args.alignment_loss_scale=scale
            with self.assertRaises(ValueError):fixtures.recipe.build_job(self.args)

if __name__=='__main__':unittest.main()
