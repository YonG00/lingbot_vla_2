"""纯主50条：FP32主参数/BF16计算→FP32 HF→FP32评测；正式micro28/GAS1。"""
import importlib.util
from pathlib import Path
import sys
from datetime import datetime

path = Path(__file__).with_name('20261019_place_mouse_pad_primary50_train_eval.py')
spec = importlib.util.spec_from_file_location('primary50_pipeline', path)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def main(argv=None):
    args = ['--fp32-master', '--micro', '28', '--gas', '1', '--eval-precision', 'fp32', '--hf-export-dtype', 'fp32']
    extra = list(sys.argv[1:] if argv is None else argv)
    if '--run-name' not in extra:
        args += ['--run-name', f'al_mouse_pad_primary50_fp32master_add500_{datetime.now():%Y%m%d_%H%M%S}']
    return pipeline.main(args + extra)


if __name__ == '__main__':
    raise SystemExit(main())
