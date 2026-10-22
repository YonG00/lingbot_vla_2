import unittest,random
from collections import Counter
from experiment.task_recipes.segment_sampling import SegmentSampler
class SegmentSamplingTest(unittest.TestCase):
 def sampler(self,ratio=None):
  p=dict(global_batch_size=56,steps=[dict(step=i+1,samples_by_task={'bowls':56}) for i in range(200)])
  return SegmentSampler(p,{'bowls':list(range(20))},3,'bowls',{'far':[[0,1],[2,3]],'place':[[3,4,5],list(range(6,15))]},list(range(15,20)),.9,ratio or {'far':1,'place':2})
 def test_exact_quota_and_membership(self):
  s=self.sampler();c=Counter();totals=Counter()
  for row in s.plan['steps']:
   ids,roles=s.draw_row(row,random.Random(str(row['step'])),c);totals.update(roles)
   self.assertEqual(len(ids),56)
   for i,role in zip(ids,roles):self.assertIn(i,s.background if role=='background' else [x for e in s.events[role] for x in e])
  self.assertEqual(totals,dict(far=3360,place=6720,background=1120))
 def test_resume_and_signature(self):
  a=self.sampler();it=iter(a);head=[next(it) for _ in range(113)];state=a.state_dict();tail=list(it)
  b=self.sampler();b.load_state_dict(state);self.assertEqual(tail,list(b));self.assertEqual(head+tail,list(self.sampler()))
  with self.assertRaises(ValueError):self.sampler({'far':2,'place':1}).load_state_dict(state)
 def test_partition_failure(self):
  with self.assertRaises(ValueError):SegmentSampler({'steps':[],'global_batch_size':1},{'x':[0,1]},0,'x',{'a':[[0]],'b':[[1]]},[1],.9,{'a':1,'b':2})
