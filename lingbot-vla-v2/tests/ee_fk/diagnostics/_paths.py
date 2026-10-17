"""Paths shared by archived FK probes; keep generated outputs out of the repo."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = Path(os.environ.get("FK_VALIDATION_DIR", "/tmp/lingbot_fk_asset_validation"))
ROBOTWIN_ROOT = Path(os.environ.get("FK_ROBOTWIN_ROOT", "/data/code/RoboTwin-lingbot"))
DATASET_ROOT = Path(os.environ.get("FK_DATASET_ROOT", "/data/robotwin_data/demo_clean"))
