#!/usr/bin/env python3
"""place_mouse_pad：原始9680，50条轨迹全部训练，位置EE权重10。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiment/robotwin/recipes/20261018_place_mouse_pad_9680_ee_lnfm.py"
spec = importlib.util.spec_from_file_location("mouse_pad_recipe", SOURCE)
recipe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(recipe)


def main(argv=None):
    args = ["--train-all-episodes", "--ee-position-weight", "10.0"]
    args.extend(sys.argv[1:] if argv is None else argv)
    return recipe.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
