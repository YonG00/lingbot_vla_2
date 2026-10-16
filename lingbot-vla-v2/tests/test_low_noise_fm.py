"""FM mixture tests use CPU tensors; no model assets, CUDA or transformer import."""

import ast
from pathlib import Path
import unittest

from lingbotvla.utils.flow_time import low_noise_time_mixture


class LowNoiseTests(unittest.TestCase):
    def test_disabled_is_identity_without_import_or_rng(self):
        marker = object()
        self.assertIs(low_noise_time_mixture(marker, False), marker)

    def test_mixture_probability_bounds_and_reproducibility(self):
        import torch

        original = torch.full((100000,), 0.8)
        torch.manual_seed(123)
        result = low_noise_time_mixture(original, True)
        changed = result != original
        fraction = changed.float().mean().item()
        self.assertTrue(0.24 < fraction < 0.26, fraction)
        self.assertGreaterEqual(result[changed].min().item(), 0.001 - 1e-8)
        self.assertLessEqual(result[changed].max().item(), 0.2)
        self.assertTrue(torch.all(original == 0.8))
        torch.manual_seed(123)
        self.assertTrue(torch.equal(result, low_noise_time_mixture(original, True)))
        before = torch.random.get_rng_state().clone()
        self.assertIs(low_noise_time_mixture(original, False), original)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_actual_sample_time_preserves_old_disabled_draws_and_v2_inherits(self):
        import torch
        from types import SimpleNamespace as NS

        root = Path(__file__).resolve().parents[1] / "lingbotvla/models/vla/lingbot_vla"
        tree = ast.parse((root / "modeling_lingbot_vla.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "FlowMatching"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "sample_time"
        )
        util = ast.parse((root / "utils.py").read_text())
        beta = next(
            n
            for n in util.body
            if isinstance(n, ast.FunctionDef) and n.name == "sample_beta"
        )
        ns = {"torch": torch}
        exec(
            compile(
                ast.Module(body=[beta, method], type_ignores=[]), "actual_fm", "exec"
            ),
            ns,
        )
        torch.manual_seed(3)
        expected = (ns["sample_beta"](1.5, 1.0, 100, "cpu") * 0.999 + 0.001).float()
        torch.manual_seed(3)
        actual = ns["sample_time"](
            NS(config=NS(low_noise_fm_enabled=False)), 100, "cpu"
        )
        self.assertTrue(torch.equal(actual, expected))
        v2 = ast.parse((root / "modeling_lingbot_vla_v2.py").read_text())
        cls = next(
            n
            for n in v2.body
            if isinstance(n, ast.ClassDef) and n.name == "FlowMatchingV2"
        )
        self.assertNotIn(
            "sample_time", [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
        )


if __name__ == "__main__":
    unittest.main()
