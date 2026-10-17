"""Contracts for opt-in fixed diffusion noise in the deployment launcher."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "deploy/lingbot_vla_v2_policy.py"
LAUNCHER = ROOT / "experiment/robotwin/start_robotwin_infer_and_eval.sh"


class FixedNoiseTests(unittest.TestCase):
    def test_policy_is_opt_in_and_restores_rng(self):
        source = POLICY.read_text(encoding="utf-8")
        self.assertIn("LINGBOT_INFER_NOISE_SEED", source)
        self.assertIn("torch.random.fork_rng", source)
        self.assertIn("with _inference_noise_context():", source)
        self.assertIn("raise ValueError(\"LINGBOT_INFER_NOISE_SEED must be an integer\")", source)

    def test_launcher_accepts_seed_and_leaves_random_default(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("--infer_noise_seed", source)
        self.assertIn('infer_noise_seed="${LINGBOT_INFER_NOISE_SEED:-}"', source)
        self.assertIn("noise_export=\"export LINGBOT_INFER_NOISE_SEED=", source)
        self.assertIn("omit for normal advancing random noise", source)


if __name__ == "__main__":
    unittest.main()
