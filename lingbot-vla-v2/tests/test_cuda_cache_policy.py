"""缓存策略的纯CPU契约；不导入torch，覆盖默认行为与强制清理。"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cuda_cache_policy", ROOT / "lingbotvla/utils/cuda_cache_policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class CachePolicyTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.cuda = SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: self.calls.append("clear"),
            memory_allocated=lambda: 20,
            memory_reserved=lambda: 30,
            mem_get_info=lambda: (70, 100),
        )

    def invoke(self, mode, context, *, trace=False, force=False, logger=None):
        with patch.object(policy, "CACHE_POLICY", mode), patch.object(policy, "CACHE_TRACE", trace):
            return policy.prepare_cuda_cache(
                self.cuda, context=context, force=force, logger=logger,
                collect_garbage=lambda: self.calls.append("gc"))

    def test_default_clear_keeps_gc_and_clear(self):
        self.assertIsNone(self.invoke("clear", "eval_entry"))
        self.assertEqual(self.calls, ["gc", "clear"])

    def test_reuse_only_skips_two_approved_contexts(self):
        for context in ("eval_entry", "periodic_hf", "general", "periodic_dcp", "final"):
            with self.subTest(context=context):
                self.calls.clear()
                self.invoke("reuse", context)
                expected = ["gc"] if context in ("eval_entry", "periodic_hf") else ["gc", "clear"]
                self.assertEqual(self.calls, expected)

    def test_force_overrides_reuse(self):
        self.invoke("reuse", "eval_entry", force=True)
        self.assertEqual(self.calls, ["gc", "clear"])

    def test_cpu_still_collects_without_cuda_calls(self):
        self.cuda.is_available = lambda: False
        self.invoke("clear", "eval_entry", trace=True)
        self.assertEqual(self.calls, ["gc"])

    def test_trace_records_memory_without_extra_synchronization(self):
        logs = []
        report = self.invoke("clear", "periodic_hf", trace=True,
                             logger=SimpleNamespace(info_rank0=logs.append))
        self.assertTrue(report["cleared"])
        self.assertEqual(report["before"]["reserved"], 30)
        self.assertEqual(report["after"]["allocated"], 20)
        self.assertGreaterEqual(report["cuda_seconds"], 0)
        self.assertEqual(len(logs), 1)
        self.assertTrue(logs[0].startswith("[cuda_cache] "))

    def test_trace_reuse_reports_no_clear(self):
        report = self.invoke("reuse", "eval_entry", trace=True)
        self.assertFalse(report["cleared"])
        self.assertEqual(self.calls, ["gc"])

    def test_diagnostic_failure_does_not_block_cleanup(self):
        def fail():
            raise RuntimeError("diagnostics unavailable")
        self.cuda.mem_get_info = fail
        report = self.invoke("clear", "eval_entry", trace=True,
                             logger=SimpleNamespace(info_rank0=lambda _: fail()))
        self.assertEqual(report["before"], {"error": "RuntimeError"})
        self.assertEqual(self.calls, ["gc", "clear"])

    def test_oom_cleanup_remains_unconditional_in_real_eval_branch(self):
        tree = ast.parse((ROOT / "lingbotvla/utils/open_loop_validation.py").read_text())
        handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)
                    and n.type is not None and ast.unparse(n.type) == "torch.cuda.OutOfMemoryError"]
        self.assertTrue(handlers)
        self.assertTrue(any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                            and ast.unparse(n.value.func) == "torch.cuda.empty_cache"
                            for handler in handlers for n in handler.body))


if __name__ == "__main__":
    unittest.main()
