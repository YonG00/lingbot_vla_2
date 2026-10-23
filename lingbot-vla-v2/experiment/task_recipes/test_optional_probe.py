import json,tempfile,unittest
from pathlib import Path
from experiment.task_recipes.schema import validate
from experiment.task_recipes.run import preflight
ROOT=Path(__file__).resolve().parents[2]
class OptionalProbeTests(unittest.TestCase):
 def recipe(self):return json.loads((ROOT/'configs/training_recipes/pick_dual_bottles300_from_0928_s200_horizontal_grasp2_replay10pct_parallel3_video.json').read_text())
 def test_default_preserves_old_recipe(self):self.assertIs(validate(self.recipe())['runtime']['gpu_probe_enabled'],True)
 def test_opt_out_is_explicit_boolean(self):
  c=self.recipe();c['runtime']['gpu_probe_enabled']=False;self.assertIs(validate(c)['runtime']['gpu_probe_enabled'],False)
  for bad in [0,1,'false',None]:
   c['runtime']['gpu_probe_enabled']=bad
   with self.assertRaises(ValueError):validate(c)
 def test_skip_still_runs_cpu_without_smoke_or_probe_file(self):
  c=self.recipe();c['runtime']['gpu_probe_enabled']=False;commands=[];states=[]
  with tempfile.TemporaryDirectory()as tmp:
   out=Path(tmp);preflight(c,out,'run.py',lambda cmd,log:commands.append(cmd),lambda name,**kw:states.append(name))
   self.assertEqual([cmd[cmd.index('--stage')+1]for cmd in commands],['cpu']);self.assertEqual(states,['cpu_preflight','gpu_probe_skipped']);self.assertFalse((out/'probe').exists())
 def test_enabled_requires_successful_expected_probe(self):
  c=self.recipe();c['runtime']['gpu_probe_enabled']=True;commands=[]
  with tempfile.TemporaryDirectory()as tmp:
   out=Path(tmp)
   def run(cmd,log):
    commands.append(cmd)
    if 'smoke'in cmd:
     (out/'probe').mkdir();(out/'probe/summary.json').write_text(json.dumps(dict(state='three_step_probe_complete',optimizer_steps=3)))
   preflight(c,out,'run.py',run,lambda *a,**kw:None)
   self.assertEqual([cmd[cmd.index('--stage')+1]for cmd in commands],['cpu','smoke'])
 def test_cpu_failure_blocks_probe_even_when_skipped(self):
  c=self.recipe();c['runtime']['gpu_probe_enabled']=False
  def fail(*a):raise RuntimeError('CPU failed')
  with tempfile.TemporaryDirectory()as tmp:
   with self.assertRaises(RuntimeError):preflight(c,Path(tmp),'run.py',fail,lambda *a,**kw:None)
