#!/usr/bin/env python3
"""place_mouse_pad：继承鼠标step500权重，50条全训练，位置EE权重10。

文件名中的9680是历史祖先；实际默认来源为鼠标单任务step500。
HF仅继承模型权重，新优化器/LR/步号；--model可显式覆盖。
"""
from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiment/robotwin/recipes/20261018_place_mouse_pad_9680_ee_lnfm.py"
spec = importlib.util.spec_from_file_location("mouse_pad_recipe", SOURCE)
recipe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(recipe)
SOURCE_RUN = Path("/data/outputs/al_mouse_pad_9680_ee_lnfm_add500_20261018_150402")


def main(argv=None):
    args = ["--train-all-episodes", "--ee-position-weight", "10.0",
            "--model", str(SOURCE_RUN / "checkpoints/global_step_500/hf_ckpt"),
            "--train-config", str(SOURCE_RUN / "lingbotvla_cli.yaml")]
    args.extend(sys.argv[1:] if argv is None else argv)
    parsed = recipe.parser().parse_args(args)
    if parsed.run_name is None:
        # 根据实际来源命名；显式覆盖模型时也不冒称源自step500。
        source = "mouse500" if parsed.model == SOURCE_RUN / "checkpoints/global_step_500/hf_ckpt" else "custom"
        name = (f"al_mouse_pad_all50_p{parsed.ee_position_weight:g}_from_{source}"
                f"_ee_lnfm_add{parsed.steps}_{datetime.now():%Y%m%d_%H%M%S}")
        args += ["--run-name", name]
    return recipe.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
