import unittest
from experiment.task_recipes.planning import PlannedSampler
class ImportanceSamplingTest(unittest.TestCase):
 def plan(self,n=1000):return dict(global_batch_size=10,steps=[dict(step=i+1,samples_by_task={'main':9,'review':1}) for i in range(n)])
 def test_ratio_and_task_quota(self):
  values=list(PlannedSampler(self.plan(),{'main':list(range(10)),'review':[10]},3,{0:3,1:3,2:3}))
  self.assertEqual(values.count(10),1000)
  ratio=sum(x<3 for x in values)/9000
  self.assertAlmostEqual(ratio,9/16,delta=.02)
 def test_resume_and_selection_signature(self):
  args=(self.plan(10),{'main':list(range(10)),'review':[10]},3)
  a=PlannedSampler(*args,{0:3});it=iter(a);first=[next(it) for _ in range(17)];state=a.state_dict();rest=list(it)
  b=PlannedSampler(*args,{0:3});b.load_state_dict(state);self.assertEqual(rest,list(b))
  self.assertEqual(first+rest,list(PlannedSampler(*args,{0:3})))
  with self.assertRaises(ValueError):PlannedSampler(*args,{0:2}).load_state_dict(state)
 def test_disabled_exact_default(self):
  args=(self.plan(10),{'main':list(range(10)),'review':[10]},3)
  self.assertEqual(list(PlannedSampler(*args)),list(PlannedSampler(*args,{})))
  with self.assertRaises(ValueError):PlannedSampler(*args,{0:float('nan')})
