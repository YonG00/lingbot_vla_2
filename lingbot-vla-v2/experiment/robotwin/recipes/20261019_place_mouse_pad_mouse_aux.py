#!/usr/bin/env python3
"""鼠标step500重训500步：48/GAS1、P12、深度视频1.25倍、每250步HF，成功后关机；默认仅CPU预览。"""
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mouse_all50', HERE/'20261018_place_mouse_pad_all50_9680_ee_lnfm.py')
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)
APPROVED = HERE.parents[2]/'refs/data_manifests/mouse/20261019_train_v1/approved.json'


def main(argv=None):
    args = ['--auxiliary-manifest', str(APPROVED),
            '--steps', '500', '--hf-save-steps', '250',
            '--gradient-checkpointing', '--micro', '48', '--gas', '1',
            '--ee-position-weight', '12', '--alignment-loss-scale', '1.25', '--shutdown']
    args.extend(sys.argv[1:] if argv is None else argv)
    return recipe.main(args)


if __name__ == '__main__':
    main()
