"""The dated recipe must allocate a fresh output directory per invocation."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT/"experiment/robotwin/recipes/20261017_stand_ee_continue500_fullvision_w3_o08_g15.sh"


class RecipeRunNameTests(unittest.TestCase):
    def test_default_name_is_timestamped_and_output_follows_name(self):
        result = subprocess.run(["bash", str(RECIPE)], capture_output=True, text=True, check=True)
        self.assertRegex(result.stdout, r"al_stand_ee_fullvision_w3_o08_g15_from500_add500_\d{8}_\d{6}")
        self.assertIn("Output: /data/outputs/al_stand_ee_fullvision_w3_o08_g15_from500_add500_", result.stdout)
        self.assertIn("--no-freeze-vision-encoder", result.stdout)
        self.assertIn("--ee-position-weight 3.0", result.stdout)
        self.assertIn("--ee-orientation-weight 0.8", result.stdout)
        self.assertIn("--ee-gripper-weight 1.5", result.stdout)
        self.assertIn("--runtime-dir /data/outputs/skill_launches", result.stdout)
        self.assertNotIn("/launch_runtime", result.stdout)

    def test_stamp_override_is_reproducible_and_does_not_execute(self):
        env = dict(os.environ, RUN_STAMP="20261018_120000", PYTHON="/usr/bin/python3")
        result = subprocess.run(["bash", str(RECIPE)], capture_output=True, text=True, env=env, check=True)
        self.assertIn("al_stand_ee_fullvision_w3_o08_g15_from500_add500_20261018_120000",result.stdout)
        self.assertNotIn("--execute",result.stdout)


if __name__ == "__main__":
    unittest.main()
