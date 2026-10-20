from datetime import datetime
from pathlib import Path
import unittest
from experiment.task_recipes.naming import dated_run_path


class RunNaming(unittest.TestCase):
    def test_prefix_and_legacy_suffix(self):
        now=datetime(2026,10,20,20,2)
        self.assertEqual(dated_run_path('/data/outputs/switch_20261020',now),'/data/outputs/20261020_2002_switch')
        self.assertEqual(dated_run_path('/data/outputs/switch_20261020_195500',now),'/data/outputs/20261020_2002_switch')

    def test_explicit_prefix_preserved(self):
        path='/data/outputs/20261020_2002_switch'
        self.assertEqual(dated_run_path(path,datetime(2026,10,21)),path)

    def test_child_keeps_parent_and_has_own_prefix(self):
        parent=Path('/data/outputs/20261020_1900_switch')
        child=Path(dated_run_path(parent/'remediation_round_1',datetime(2026,10,20,20,2)))
        self.assertEqual(child.parent,parent)
        self.assertEqual(child.name,'20261020_2002_remediation_round_1')

if __name__=='__main__':unittest.main()
