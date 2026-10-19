"""最新纯主任务FP32 500步续训500步：FK位置6、姿态0.8、夹爪0。"""
import importlib.util
from datetime import datetime
from pathlib import Path
import sys

SOURCE_RUN = Path('/data/outputs/al_mouse_pad_primary50_fp32master_fp32hf_m28_add500_20261019_v1')
spec = importlib.util.spec_from_file_location(
    'gripper_off_continuation', Path(__file__).resolve().parents[1] / 'continue_skill_training.py')
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


def main(argv=None):
    extra = list(sys.argv[1:] if argv is None else argv)
    args = ['--from-hf', str(SOURCE_RUN / 'checkpoints/global_step_500/hf_ckpt'),
            '--al-config', str(SOURCE_RUN / 'al_config.json'),
            '--steps', '500', '--micro', '28', '--gas', '1', '--fp32-master',
            '--no-freeze-vision-encoder', '--gradient-checkpointing',
            '--ee-position-weight', '6', '--ee-gripper-weight', '0']
    if '--run-name' not in extra:
        args += ['--run-name', f'al_mouse_pad_fp32_m28_p6_grip0_add500_{datetime.now():%Y%m%d_%H%M%S}']
    # Defaults are preview-only; --execute/--shutdown remain explicit actions.
    return continuation.main(args + extra)


if __name__ == '__main__':
    raise SystemExit(main())
