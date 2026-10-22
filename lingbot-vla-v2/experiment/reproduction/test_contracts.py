import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiment.reproduction.run import arguments,plan,curriculum,resolve_template,sequential,recipe,execute,dump,load
from experiment.task_recipes.planning import build_plan
from experiment.task_recipes.schema import validate

class ReproductionTests(unittest.TestCase):
    def settings(self,out='/tmp/reproduction_contract'):
        return arguments(['--base-model','/tmp/base','--dataset-root','/tmp/data','--output-dir',out])[1]
    def test_complete_budget_and_save_interval(self):
        p=plan(self.settings())
        self.assertEqual(p['fixed_steps'],15150);self.assertEqual(p['maximum_remediation_steps'],2500)
        self.assertEqual(p['jobs'][0]['saves'],[2000,4000,5000]);self.assertEqual(p['fixed_hf_count'],16)
        s=self.settings();s['save_every']=700
        self.assertEqual(plan(s)['jobs'][0]['saves'],[700,1400,2100,2800,3500,4200,4900,5000])
    def test_mainchain_and_no_discarded_branches(self):
        c=curriculum();self.assertEqual(len(c['task_order']),50);self.assertEqual(len(set(c['task_order'])),50)
        self.assertEqual([j['label']for j in c['stage2']],['R3','R5','R6','R8','R9','R10','R11','R12','R13','R14','R15','R16'])
        self.assertEqual(c['task_order'][:4],['adjust_bottle','place_container_plate','place_empty_cup','move_can_pot'])
    def test_foundation_reviews_only_completed_tasks_and_preserves_batch(self):
        p=plan(self.settings());r=p['jobs'][0]['recipe'];rows=build_plan(r)['steps'];order=r['train']['task_order']
        for i,row in enumerate(rows):
            current=i//100
            self.assertEqual(row['main_task'],order[current]);self.assertEqual(sum(row['samples_by_task'].values()),56)
            review=set(row['samples_by_task'])-{row['main_task']}
            self.assertLessEqual(review,set(order[:current]))
        self.assertEqual(sum(x['replay_samples']for x in rows[:100]),0)
        self.assertEqual(sum(x['replay_samples']for x in rows),27440)
        self.assertEqual([x['replay_task']for x in rows[200:205]],[rows[200]['replay_task']]*5)
    def test_completed_only_rejects_incompatible_modes(self):
        r=plan(self.settings())['jobs'][0]['recipe'];r['train']['replay']['tasks_per_step']=2
        with self.assertRaises(ValueError):validate(r)
    def test_reference_scan_covers_all_25_once(self):
        p=plan(self.settings());flat=sum(p['scout_batches'],[])
        self.assertEqual(len(p['scout_batches']),5);self.assertEqual(len(flat),len(set(flat)))
        self.assertEqual(sorted(flat),p['reference_tasks']);self.assertIn('place_a2b_right',flat)
        self.assertTrue(all(j['recipe']['evaluation']['video_episodes']==0 for j in p['jobs']))
    def test_template_has_no_server_paths(self):
        c=resolve_template(self.settings());text=json.dumps(c)
        self.assertNotIn('${',text);self.assertNotIn('/data/',text);self.assertNotIn('/root/',text)
        self.assertEqual(c['train']['lr'],.0001);self.assertTrue(c['train']['enable_gradient_checkpointing'])
        self.assertFalse(c['train']['ee_fk_loss_enabled']);self.assertFalse(c['train']['low_noise_fm_enabled'])
    def test_orchestration_and_completed_boundary_resume_without_gpu(self):
        with tempfile.TemporaryDirectory()as folder:
            s=self.settings(folder);trained=[];scans=[]
            def worker(r,launch):
                trained.append(launch.name);Path(r['output_dir']).mkdir()
                dump(launch/'status.json',dict(state='complete',evaluation=[]));return 0
            def evaluate(jobs,out,workers):
                scans.append([j['task']for j in jobs])
                return [dict(task=j['task'],state='complete',episodes=3,successes=0)for j in jobs]
            with patch('experiment.reproduction.run.preflight'),patch('experiment.reproduction.run.provenance',return_value={}),patch('experiment.reproduction.run.convert'),patch('experiment.task_recipes.run.worker',worker),patch('experiment.official_control.baseline_pipeline.verify_checkpoint',return_value={'path':'verified'}),patch('experiment.official_control.parallel_evaluation.run_jobs',evaluate):
                execute(s)
                self.assertEqual(len(trained),19);self.assertEqual(len(scans),5)
                final=load(Path(folder)/'final_model.json');self.assertEqual(len(final['scan_tasks']),25)
                self.assertEqual(load(Path(folder)/'status.json')['state'],'complete')
                execute(s)
                self.assertEqual(len(trained),19);self.assertEqual(len(scans),5)
            self.assertFalse((Path(folder)/'controller.lock').exists())

if __name__=='__main__':unittest.main()
