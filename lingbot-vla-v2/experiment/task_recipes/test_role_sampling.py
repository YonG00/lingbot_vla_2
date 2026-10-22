import unittest,random
from collections import Counter
from experiment.task_recipes.role_sampling import RoleSampler
class RoleSamplingTests(unittest.TestCase):
 def sampler(self):
  plan=dict(global_batch_size=56,steps=[dict(step=i+1,samples_by_task={'bowls':56})for i in range(200)])
  return RoleSampler(plan,{'bowls':list(range(10))},3,'bowls',{'left_grasp':[[0,1]],'right_grasp':[[2,3]],'second_bowl_place':[[1,4,5]]},list(range(6,10)),dict(left_grasp=.25,right_grasp=.25,second_bowl_place=.4,background=.1),dict(left_grasp=3,right_grasp=2,second_bowl_place=2,background=1))
 def test_exact_counts_weights_and_overlap_source(self):
  s=self.sampler();counts=Counter();seen_overlap=set();c=Counter()
  for row in s.plan['steps']:
   ids,roles=s.draw_row(row,random.Random(str(row['step'])),c);counts.update(roles)
   for (i,w,k),role in zip(ids,roles):
    self.assertEqual(k,role);self.assertEqual(w,s.loss_weights[role])
    self.assertIn(i,s.background if role=='background' else [x for event in s.events[role]for x in event])
    if i==1:seen_overlap.add((k,w))
  self.assertEqual(counts,dict(left_grasp=2800,right_grasp=2800,second_bowl_place=4480,background=1120))
  self.assertEqual(seen_overlap,{('left_grasp',3),('second_bowl_place',2)})
 def test_resume_and_role_weight_identity(self):
  a=self.sampler();it=iter(a);head=[next(it)for _ in range(71)];state=a.state_dict();tail=list(it)
  b=self.sampler();b.load_state_dict(state);self.assertEqual(tail,list(b));self.assertEqual(head+tail,list(self.sampler()))
 def test_stream_integer_path_unchanged(self):
  from experiment.task_recipes.runtime import StreamDataset
  self.assertEqual(StreamDataset(['original'],56)[0],'original')
