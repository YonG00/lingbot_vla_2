"""CPU contracts for FP32-source continuation with position6/gripper0."""
import importlib.util
import json
from pathlib import Path
import shutil
import struct

from mouse_pad_training import test_recipe as fixtures

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'gripper_off_recipe', ROOT / 'experiment/robotwin/recipes/20261019_place_mouse_pad_fp32_gripper_off.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def test_entry_defaults_position6_gripper0_preserve_other_controls(monkeypatch):
    captured = []
    monkeypatch.setattr(entry.continuation, 'main', lambda argv: captured.extend(argv))
    entry.main([])
    assert captured[captured.index('--ee-gripper-weight') + 1] == '0'
    assert captured[captured.index('--micro') + 1] == '28'
    assert captured[captured.index('--gas') + 1] == '1'
    assert captured[captured.index('--steps') + 1] == '500'
    assert '--fp32-master' in captured and '--gradient-checkpointing' in captured
    assert '--no-freeze-vision-encoder' in captured
    assert captured[captured.index('--ee-position-weight') + 1] == '6'
    assert '--ee-orientation-weight' not in captured
    assert '--execute' not in captured and '--shutdown' not in captured
    assert 'primary50_fp32master_fp32hf_m28' in captured[captured.index('--from-hf') + 1]


def test_fp32_source_worker_preserves_training_except_position_and_gripper(tmp_path, monkeypatch):
    f = fixtures.MouseRecipeTests()
    f.setUp()
    try:
        a = f.args
        a.fp32_master, a.train_all_episodes, a.gradient_checkpointing = True, True, True
        a.micro, a.gas, a.ee_position_weight = 28, 1, 10.
        cfg = json.loads(a.train_config.read_text())
        cfg['train'] = dict(hf_export_dtype='fp32', lr=1e-4, lr_min=5e-5,
                            loss_type='L1_fm', optimizer='muon', lr_warmup_ratio=0.)
        a.train_config.write_text(json.dumps(cfg))
        _, reference, _ = fixtures.recipe.build_job(a)
        # Materialize the effective source CLI, as a completed real run does.
        source_cfg = reference['training_config']
        cmd = reference['command']
        for i, flag in enumerate(cmd):
            if flag.startswith('--train.'):
                try:
                    value = json.loads(cmd[i + 1])
                except ValueError:
                    value = cmd[i + 1]
                source_cfg['train'][flag.removeprefix('--train.')] = value
        run = tmp_path / 'completed'
        model = run / 'checkpoints/global_step_500/hf_ckpt'
        shutil.copytree(a.model, model)
        header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
        (model / 'model-01.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + bytes(4))
        source_cfg['model']['tokenizer_path'] = str(a.qwen3vl)
        (run / 'lingbotvla_cli.yaml').write_text(json.dumps(source_cfg))
        (run / 'datasets.txt').write_text(f'robotwin {a.dataset_root}\n')
        al = dict(reference['config'], data_audit_file=str(tmp_path / 'audit.json'))
        (run / 'al_config.json').write_text(json.dumps(al))
        monkeypatch.setattr(entry, 'SOURCE_RUN', run)
        captured = {}
        original = entry.continuation.make_job
        def capture(args):
            job, plan = original(args)
            captured.update(job=job, plan=plan)
            return job, plan
        monkeypatch.setattr(entry.continuation, 'make_job', capture)
        entry.main(['--run-name', 'grip0', '--output-root', str(tmp_path / 'outputs'),
                    '--runtime-dir', str(tmp_path / 'launches')])
        job, plan = captured['job'], captured['plan']
        assert job['hf_export_dtype'] == 'fp32' and job['verify_steps'] == [250, 500]
        assert plan['counts'] == {'place_mouse_pad': {'train': 50, 'val': 0}}
        assert plan['train_ids'] == reference['train_ids']
        assert plan['config']['low_noise_fm_enabled'] is True
        assert plan['config']['sample_scan_method'] == 'target_change_window'
        frozen = tmp_path / 'frozen.al.json'
        frozen.write_text(json.dumps(plan['config']))
        command = job['command'][2:]
        command[command.index('--al-config') + 1] = str(frozen)
        command[command.index('--execute')] = '--prepare-only'
        entry.continuation.launcher.main(command)
        rebuilt = json.loads((Path(job['output_dir']) / 'trainer_config.json').read_text())
        previous = source_cfg['train']
        actual = rebuilt['train']
        assert actual['ee_fk_gripper_loss_weight'] == 0
        for key, value in previous.items():
            if key in {'ee_fk_gripper_loss_weight', 'ee_fk_position_loss_weight',
                       'output_dir', 'auto_learning'}:
                continue
            assert actual[key] == value, key
        assert actual['ee_fk_position_loss_weight'] == 6.
        assert actual['ee_fk_orientation_loss_weight'] == .8
        assert actual['hf_export_dtype'] == 'fp32'
        assert actual['save_steps'] == 250
    finally:
        f.doCleanups()
