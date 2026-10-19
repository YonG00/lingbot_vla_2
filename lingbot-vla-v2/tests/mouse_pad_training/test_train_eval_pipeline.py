"""CPU-only lifecycle tests: no real training, subprocess, GPU or shutdown."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('train_eval_pipeline', ROOT / 'experiment/robotwin/recipes/20261019_place_mouse_pad_primary50_train_eval.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_source_identity_accepts_symlink_alias_but_rejects_other_model(tmp_path, monkeypatch):
    real = tmp_path / 'real'
    real.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(pipeline.source, 'SOURCE_RUN', alias)
    args = SimpleNamespace(steps=500, hf_save_steps=250, micro=56, gas=1,
        gradient_checkpointing=True, train_all_episodes=True, auxiliary_manifest=None,
        ee_position_weight=10., ee_orientation_weight=.8, ee_gripper_weight=1.5,
        alignment_loss_scale=1., closed_loop_episodes=3)
    plan = dict(counts={'place_mouse_pad': {'train': 50, 'val': 0}}, auxiliary=None,
        training_config={'train': {'align_params': dict(depth_loss_weight=.004,
        future_depth_loss_weight=.004, video={'future_video_loss_weight': .004})}})
    pipeline.validate_plan(args, {'source_model': str(real / 'checkpoints/global_step_500/hf_ckpt')}, plan)
    with pytest.raises(AssertionError):
        pipeline.validate_plan(args, {'source_model': str(tmp_path / 'other')}, plan)


@pytest.mark.parametrize('failure', ['training', 'checkpoint', 'evaluation', None])
def test_shutdown_only_after_verified_training_and_completed_evaluation(tmp_path, monkeypatch, failure):
    events = []
    def train(job):
        assert job['shutdown'] is False
        events.append('training')
        return 1 if failure == 'training' else 0
    def verify(job):
        events.append(f'verify{job["steps"]}')
        if failure == 'checkpoint': raise RuntimeError('incomplete')
        return tmp_path / 'hf_ckpt'
    def evaluate(config, model):
        events.append('evaluation')
        if failure == 'evaluation': raise RuntimeError('render error')
        return {'episodes': 3, 'successes': 0, 'success_rate': 0.0}
    shutdown = Mock(side_effect=lambda *a, **kw: events.append('shutdown'))
    monkeypatch.setattr(pipeline.continuation, 'run_job', train)
    monkeypatch.setattr(pipeline.continuation, 'verify_final', verify)
    monkeypatch.setattr(pipeline, 'evaluate', evaluate)
    monkeypatch.setattr(pipeline.subprocess, 'run', shutdown)
    status = tmp_path / 'status.json'
    rc = pipeline.run_pipeline(dict(training_job={'shutdown': False, 'steps': 500},
                                    shutdown=True, pipeline_status=str(status)))
    data = json.loads(status.read_text())
    if failure:
        assert rc == 1 and data['state'] == 'failed'
        shutdown.assert_not_called()
    else:
        assert events == ['training', 'verify250', 'verify500', 'evaluation', 'shutdown']
        assert rc == 0 and data['evaluation_complete'] and data['training_complete']
        shutdown.assert_called_once_with(['bash', '/usr/bin/shutdown'], check=True)


@pytest.mark.parametrize('fault', ['partial', 'oidn', 'video', 'rate', 'nan', None])
def test_eval_requires_completed_rounds_valid_render_result_and_videos(tmp_path, fault):
    directory = tmp_path / 'results/place_mouse_pad'
    directory.mkdir(parents=True)
    for i in range(3): (directory / f'episode{i}_global.mp4').write_bytes(b'video')
    log = '[render_runtime] OIDN 2.5.1 CUDA=[GPU]\n'
    for i in range(1, 4):
        if fault == 'partial' and i == 3: continue
        log += f'Success rate: \x1b[96m0/{i}\x1b[0m => 0%, current seed: {100000+i-1}\n'
    if fault == 'oidn': log += 'OIDN Error: invalid handle\n'
    (tmp_path / 'sim.log').write_text(log)
    value = 'nan' if fault == 'nan' else '0.5' if fault == 'rate' else '0.0'
    (directory / '_result.txt').write_text('Timestamp\n\n' + value)
    if fault == 'video': (directory / 'episode2_global.mp4').unlink()
    if fault:
        with pytest.raises(RuntimeError): pipeline.validate_evaluation(tmp_path, 3, 'sha')
    else:
        result = pipeline.validate_evaluation(tmp_path, 3, 'sha')
        assert result['successes'] == 0 and result['episodes'] == 3
        assert result['seeds'] == [100000, 100001, 100002]
