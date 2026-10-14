import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("al_cache_profile", Path(__file__).resolve().parents[1]
                                           / "tools/auto_learning_pipeline/al_cache_profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class CacheProfileTests(unittest.TestCase):
    def test_counts_actual_steps_eval_and_cache_observations(self):
        report = {"context": "eval_entry", "cleared": False, "gc_seconds": 0.1,
                  "cuda_seconds": 0.2, "before": {"reserved": 30}, "after": {"reserved": 20}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run.log"
            path.write_text(
                "10/14/2026 12:00:00 start\n"
                "[cuda_cache] " + json.dumps(report) + "\n"
                "[cuda_cache] malformed\n"
                '[cuda_cache] {"context": "incomplete"}\n'
                '[cuda_cache] []\n'
                "Step 1/20, Loss 0.2, StepTime 2.0s\n"
                "Step 1/20, Loss 0.2, StepTime 2.0s\n"
                "Step 2/20, Loss 0.1, StepTime 4.0s\n"
                "eval t/val n_ids=2: mse=0.1 (2轨迹/10chunk, 2.1s)\n"
                "10/14/2026 12:00:10 批处理 OOM\n")
            result = profile.summarize(path)
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["step_range"], [1, 2])
        self.assertEqual(result["last_loss"], 0.1)
        self.assertEqual(result["median_step_seconds"], 3)
        self.assertEqual(result["elapsed_log_seconds"], 10)
        self.assertEqual(result["eval_time_sum_seconds"], 2.1)
        self.assertEqual(result["oom_log_lines"], 1)
        self.assertEqual(result["cache"]["eval_entry"]["max_observed_reserved_bytes"], 30)
        self.assertEqual(result["cache"]["eval_entry"]["cleared"], 0)

    def test_empty_log_does_not_invent_measurements(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "empty.log"
            path.touch()
            result = profile.summarize(path)
        self.assertEqual(result["steps"], 0)
        self.assertIsNone(result["elapsed_log_seconds"])
        self.assertIsNone(result["median_step_seconds"])
        self.assertEqual(result["cache"], {})


if __name__ == "__main__":
    unittest.main()
