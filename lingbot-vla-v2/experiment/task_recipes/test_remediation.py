"""无GPU的补课编排契约；训练/评测回调模拟。"""
from copy import deepcopy
from pathlib import Path
import unittest
from experiment.task_recipes.schema import validate, milestones
from experiment.task_recipes.remediation import curriculum, run_rounds
from experiment.task_recipes.test_contracts import fixture


class RemediationContracts(unittest.TestCase):
    def config(self):
        raw=fixture('sequential_plate100_cup200_pot200')
        raw['remediation']=dict(enabled=True)
        return validate(raw)

    def results(self,c,scores=(0,1,2)):
        model=str(Path(c['output_dir'])/f"checkpoints/global_step_{c['train']['steps']}/hf_ckpt")
        return [dict(state='complete',task=t,model=model,episodes=3,successes=s)
                for t,s in zip(c['evaluation']['tasks'],scores)]

    def test_defaults_and_validation(self):
        self.assertEqual(validate(fixture())['remediation'],dict(enabled=False,max_rounds=1))
        for mutate in [lambda c:c['evaluation'].update(episodes_per_task=4),
                       lambda c:c['evaluation'].update(checkpoints=[250]),
                       lambda c:c['remediation'].update(max_rounds=0),
                       lambda c:c['remediation'].update(enabled=1)]:
            c=self.config();mutate(c)
            with self.assertRaises(ValueError):validate(c)

    def test_priority_and_all_pass_skip(self):
        c=self.config();r=self.results(c)
        lessons=curriculum(c['evaluation']['tasks'],r,r[0]['model'])
        self.assertEqual([x['successes'] for x in lessons],[2,1])
        self.assertEqual([x['steps'] for x in lessons],[100,100])
        self.assertEqual(run_rounds(c,self.results(c,(3,3,3)),lambda _:self.fail(),lambda *a:self.fail()),[])
        self.assertEqual(run_rounds(c,self.results(c,(0,0,0)),lambda _:self.fail(),lambda *a:self.fail()),[])
        all_one=self.results(c,(1,1,1))
        self.assertEqual([x['task'] for x in curriculum(c['evaluation']['tasks'],all_one,all_one[0]['model'])],c['evaluation']['tasks'])

    def test_error_missing_duplicate_and_wrong_model_rejected(self):
        c=self.config();good=self.results(c);model=good[0]['model']
        bads=[good[:-1],good+[good[0]]]
        for changes in [dict(state='failed'),dict(episodes=2),dict(successes=True),dict(successes=4),dict(model='wrong')]:
            bad=deepcopy(good);bad[0].update(changes);bads.append(bad)
        for bad in bads:
            with self.assertRaises(ValueError):curriculum(c['evaluation']['tasks'],bad,model)

    def test_round_limit_source_all_task_retest_and_cadence(self):
        for limit in [1,2]:
            c=self.config();c['remediation']['max_rounds']=limit;children=[];records=[]
            def execute(child):
                validate(child);children.append(child)
                self.assertFalse(child['remediation']['enabled'])
                self.assertEqual(child['evaluation']['tasks'],child['train']['task_order'])
                self.assertEqual(milestones(child),[100*len(child['train']['main_tasks'])])
                self.assertEqual(child['train']['replay']['fraction'],0)
                return self.results(child)
            history=run_rounds(c,self.results(c),execute,lambda *args:records.append(args))
            self.assertEqual(len(history),limit);self.assertEqual(len(records),limit)
            self.assertEqual(children[0]['source']['checkpoint'],self.results(c)[0]['model'])
            if limit==2:self.assertEqual(children[1]['source']['checkpoint'],self.results(children[0])[0]['model'])

    def test_disabled_and_training_failure(self):
        c=self.config();c['remediation']['enabled']=False
        self.assertEqual(run_rounds(c,[],lambda _:self.fail(),lambda *a:self.fail()),[])
        c['remediation']['enabled']=True
        def fail(_):raise RuntimeError('training failed')
        with self.assertRaises(RuntimeError):run_rounds(c,self.results(c),fail,lambda *a:None)
        with self.assertRaises(ValueError):run_rounds(c,self.results(c),lambda _:[],lambda *a:None)

if __name__=='__main__':unittest.main()
