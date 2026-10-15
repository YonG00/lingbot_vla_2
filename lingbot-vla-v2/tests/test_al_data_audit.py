"""真实 local_idx 映射 + source metadata 对拍，无模型/图像加载。"""
import tempfile
from pathlib import Path
import unittest

from lingbotvla.auto_learning.catalog import TaskCatalog
from lingbotvla.auto_learning.data_audit import audit_mapping
from lingbotvla.auto_learning.ports import TaskEntry
from lingbotvla.auto_learning.resolver import SampleResolver


class DataAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "meta").mkdir()
        (self.root / "meta/info.json").write_text("{}")
        self.cat = TaskCatalog({"cabinet": TaskEntry(name="cabinet",
            train_traj_ids=[0, 1], val_traj_ids=[2], n_total=3)},
            task_order=["cabinet"], episodes_per_task=3)
        self.resolver = SampleResolver(episode_index=[0, 0, 1, 1], frame_index=[0, 1, 2, 3],
                                       task_of_episode=self.cat.task_of_episode)
        self.meta = {i: dict(episode_index=i, length=2, tasks=["put object in cabinet"],
                            dataset_from_index=i*2, dataset_to_index=i*2+2) for i in range(3)}
        self.kw = dict(dataset_root=self.root, episode_getter=self.meta.__getitem__,
            report_file=self.root / "audit.json", instruction_patterns={"cabinet": "cabinet"},
            task_indices=[0]*4, task_label=lambda i: "put object in cabinet")

    def run_audit(self):
        return audit_mapping(self.cat, self.resolver, **self.kw)

    def test_full_then_cached_does_not_repeat_full_check(self):
        first = self.run_audit()
        self.assertFalse(first["cached"])
        self.assertEqual(first["tasks"][0]["train_samples"], 4)
        self.kw["episode_getter"] = lambda _: self.fail("full audit unexpectedly repeated")
        self.assertTrue(self.run_audit()["cached"])

    def test_mapping_mutation_and_source_identity_invalidate_cache(self):
        self.run_audit()
        # 数量完全相同、绝对 frame 单调，但回合边界错位。
        self.resolver = SampleResolver(episode_index=[0, 1, 1, 1], frame_index=[0, 1, 2, 3],
                                       task_of_episode=self.cat.task_of_episode)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.run_audit()
        self.setUp()
        self.run_audit()
        (self.root / "meta/info.json").write_text('{"changed": true}')
        self.assertFalse(self.run_audit()["cached"])

    def test_wrong_task_instruction_and_frame_task_id_fail(self):
        self.meta[0]["tasks"] = ["click bell"]
        with self.assertRaisesRegex(ValueError, "instruction mismatch"):
            self.run_audit()
        self.meta[0]["tasks"] = ["put object in cabinet"]
        self.kw["task_label"] = lambda _: "click bell"
        with self.assertRaisesRegex(ValueError, "task_index"):
            self.run_audit()

    def test_val_leak_and_wrong_absolute_index_fail(self):
        self.resolver = SampleResolver(episode_index=[0, 0, 2, 2], frame_index=[0, 1, 4, 5],
                                       task_of_episode=self.cat.task_of_episode)
        with self.assertRaisesRegex(ValueError, "train episodes mismatch"):
            self.run_audit()
        self.resolver = SampleResolver(episode_index=[0, 0, 1, 1], frame_index=[10, 11, 12, 13],
                                       task_of_episode=self.cat.task_of_episode)
        with self.assertRaisesRegex(ValueError, "wrong absolute frame"):
            self.run_audit()


if __name__ == "__main__":
    unittest.main()
