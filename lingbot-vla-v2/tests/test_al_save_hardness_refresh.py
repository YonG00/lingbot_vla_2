"""执行真实 scorer 类的缓存路径；隔离重依赖，CPU/标准库即可验证。"""
import ast
from contextlib import nullcontext
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch


BACKEND = Path(__file__).resolve().parents[1] / "lingbotvla/auto_learning/real/backend.py"


def scorer_class():
    # 不复制缓存算法；提取整个生产类，绕过 backend 模块顶层的 torch/数据集导入。
    tree = ast.parse(BACKEND.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "RealHardnessScorer")
    module = ast.parse("from __future__ import annotations\n")
    module.body.append(cls)
    ns = dict(os=os, time=time, deterministic_sampling=lambda dataset: nullcontext({"n_ft": 1}))
    exec(compile(module, str(BACKEND), "exec"), ns)
    return ns["RealHardnessScorer"]


class Cache:
    def __init__(self):
        self.values = {i: -1.0 for i in range(12)}
        self.reads = self.writes = 0

    def load(self, task, ids):
        self.reads += 1
        return {i: self.values[i] for i in ids if i in self.values}, [i for i in ids if i not in self.values]

    def store(self, task, values):
        self.writes += 1
        self.values.update(values)


class Core:
    def __init__(self):
        self.calls = 0
        self.weight = 1.0

    def score(self, items, **kwargs):
        self.calls += 1
        return [float(i) + self.weight for i in items]


class SaveHardnessCacheTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.cache, self.core = Cache(), Core()
        self.scorer = scorer_class()(self.core, list(range(12)), cache=self.cache, max_batch=4)

    def test_fresh_scan_uses_current_weights_without_reading_or_writing_old_cache(self):
        ids = [1, 2, 3]
        self.assertEqual(self.scorer.score("t0", ids), {i: -1.0 for i in ids})
        self.assertEqual(self.core.calls, 0)
        self.core.weight = 10.0
        self.assertEqual(self.scorer.score_fresh("t0", ids), {i: i + 10.0 for i in ids})
        self.assertEqual(self.core.calls, 1)
        self.assertEqual((self.cache.reads, self.cache.writes), (1, 0))
        self.assertIs(self.scorer.cache, self.cache)
        self.assertEqual(self.scorer.score("t0", ids), {i: -1.0 for i in ids})

    def test_fresh_scan_restores_cache_even_when_model_raises(self):
        with patch.object(self.core, "score", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                self.scorer.score_fresh("t0", [1])
        self.assertIs(self.scorer.cache, self.cache)
        self.assertEqual((self.cache.reads, self.cache.writes), (0, 0))

    def test_uncached_scorer_retains_existing_behavior(self):
        self.scorer.cache = None
        self.assertEqual(self.scorer.score_fresh("t0", [1, 2]), self.scorer.score("t0", [1, 2]))
        self.assertIsNone(self.scorer.cache)


if __name__ == "__main__":
    unittest.main()
