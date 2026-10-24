import hashlib,json,tempfile,unittest,shutil
from pathlib import Path
from .weights import SelectionWeights
from .inheritance import freeze_registry,NAME

class ImportanceInheritanceTests(unittest.TestCase):
    def config(self,root,task,episode,multiplier):
        p=root/task;p.mkdir(parents=True)
        data=json.dumps([dict(episode_index=episode,frame_index=2)]).encode();(p/'frames.json').write_bytes(data)
        cfg=dict(version=1,task=task,dataset_name='robotwin',dataset_root='/tmp/shared_robotwin',normalization='sample_count',multiplier=multiplier,selected_frames='frames.json',selected_frames_sha256=hashlib.sha256(data).hexdigest())
        path=p/'config.json';path.write_text(json.dumps(cfg));return path
    def test_replay_retains_weight_without_explicit_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);config=self.config(p/'inputs','stack_bowls_two',1450,3)
            first=p/'first';first.mkdir();freeze_registry(p/'unweighted',first,config)
            next_run=p/'other_main_task';next_run.mkdir();registry=freeze_registry(first,next_run)
            policy=SelectionWeights(registry)
            self.assertEqual(policy.weight('robotwin',1450,2),3)
            self.assertEqual(policy.weight('robotwin',1450,3),1)
            self.assertEqual(policy.weight('other_dataset',1450,2),1)
            shutil.rmtree(first);shutil.rmtree(p/'inputs')
            self.assertEqual(SelectionWeights(registry).weight('robotwin',1450,2),3)
    def test_new_task_preserves_previous_task_importance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);a=self.config(p/'a','stack_bowls_two',1450,3);b=self.config(p/'b','new_task',77,2)
            first=p/'first';first.mkdir();freeze_registry(p/'empty',first,a)
            second=p/'second';second.mkdir();registry=freeze_registry(first,second,b)
            policy=SelectionWeights(registry)
            self.assertEqual(policy.weight('robotwin',1450,2),3)
            self.assertEqual(policy.weight('robotwin',77,2),2)
            self.assertEqual(policy.multipliers,{2.,3.})
    def test_explicit_same_task_update_replaces_old_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);a=self.config(p/'a','stack_bowls_two',1450,3);b=self.config(p/'b','stack_bowls_two',1450,4)
            first=p/'first';first.mkdir();freeze_registry(p/'empty',first,a)
            second=p/'second';second.mkdir();registry=freeze_registry(first,second,b)
            self.assertEqual(SelectionWeights(registry).weight('robotwin',1450,2),4)
    def test_narrower_same_task_policy_replaces_old_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);a=self.config(p/'a','stack_bowls_two',1450,3);b=self.config(p/'b','stack_bowls_two',1450,3)
            selected=a.parent/'frames.json';data=json.dumps([dict(episode_index=1450,frame_index=2),dict(episode_index=1450,frame_index=3)]).encode();selected.write_bytes(data)
            cfg=json.loads(a.read_text());cfg['selected_frames_sha256']=hashlib.sha256(data).hexdigest();a.write_text(json.dumps(cfg))
            first=p/'first';first.mkdir();freeze_registry(p/'empty',first,a)
            second=p/'second';second.mkdir();registry=freeze_registry(first,second,b)
            weights=SelectionWeights(registry)
            self.assertEqual(weights.weight('robotwin',1450,2),3)
            self.assertEqual(weights.weight('robotwin',1450,3),1)

    def test_config_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);a=self.config(p/'inputs','stack_bowls_two',1450,3);out=p/'out';out.mkdir();registry=freeze_registry(p/'empty',out,a)
            config=out/'sample_weights/stack_bowls_two/weight_config.json';v=json.loads(config.read_text());v['multiplier']=1;config.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'fingerprint'):SelectionWeights(registry)
    def test_checkpoint_source_and_legacy_audit_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);a=self.config(p/'inputs','stack_bowls_two',1450,3);source=p/'source'
            hf=source/'checkpoints/global_step_200/hf_ckpt';hf.mkdir(parents=True)
            audit=p/'source_pipeline';audit.mkdir();(audit/'sample_weight_audit.json').write_text(json.dumps(dict(weight_config=str(a))))
            out=p/'out';out.mkdir();registry=freeze_registry(hf,out)
            self.assertEqual(SelectionWeights(registry).weight('robotwin',1450,2),3)
    def test_unweighted_baseline_has_no_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);out=p/'out';out.mkdir()
            self.assertIsNone(freeze_registry(p/'empty',out));self.assertFalse((out/NAME).exists())

    def test_new_freeze_canonicalizes_legacy_format_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);source=self.config(p/'inputs','old_task',9,2);registry=freeze_registry(p/'empty',p/'out',source)
            frozen=json.loads((p/'out/sample_weights/old_task/weight_config.json').read_text())
            self.assertEqual(frozen['weighting_unit'],'action_timestep');self.assertEqual(frozen['normalization'],'weight_sum')
            self.assertEqual(json.loads(source.read_text())['normalization'],'sample_count');self.assertTrue(SelectionWeights(registry).weight_sum)

if __name__=='__main__':unittest.main()
