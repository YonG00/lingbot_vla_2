"""PASS milestone + live HF snapshot regressions (CPU; no model download)."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from lingbotvla.utils.al_checkpoint_policy import (
    learned_nonbootstrap_count, milestone_action, pass_milestone_index,
)

ROOT = Path(__file__).resolve().parents[1]


def state(*, bootstrap=(), trained=(), automatic=(), available=True):
    return SimpleNamespace(bootstrap_passed=list(bootstrap),
                           newly_passed=list(trained), auto_passed=list(automatic),
                           bootstrap_passed_available=available)


def test_initial_two_bootstrap_does_not_export():
    s = state(bootstrap=('A', 'B'), automatic=('A', 'B'))
    assert learned_nonbootstrap_count(s) == 0
    assert milestone_action(observed_index=pass_milestone_index(s, 2),
                            committed_index=0, final_due=False,
                            dcp_due=False) == 'none'


def test_two_new_passes_export_hf_midrun():
    s = state(bootstrap=('A',), automatic=('A',), trained=('X', 'Y'))
    assert pass_milestone_index(s, 2) == 1
    assert milestone_action(observed_index=1, committed_index=0,
                            final_due=False, dcp_due=False) == 'hf'


def test_training_and_rescan_unique_and_repass_not_counted_twice():
    s = state(bootstrap=('A',), trained=('X','X'), automatic=('A','Y','Y', 'X'))
    assert learned_nonbootstrap_count(s) == 2


@pytest.mark.parametrize('final_due,dcp_due', [(True,False),(False,True),(True,True)])
def test_dcp_takes_priority_over_hf(final_due, dcp_due):
    assert milestone_action(observed_index=1, committed_index=0,
                            final_due=final_due, dcp_due=dcp_due) == 'covered_by_dcp'


def test_resume_cursor_does_not_reexport_old_milestone():
    assert milestone_action(observed_index=1, committed_index=1,
                            final_due=False, dcp_due=False) == 'none'


def test_legacy_bootstrap_unknown_never_assumes_auto_passed_is_new():
    s = state(automatic=('initial', 'rescan'), trained=('learned',), available=False)
    assert learned_nonbootstrap_count(s) == 1


def test_bad_interval():
    with pytest.raises(ValueError):
        pass_milestone_index(state(), 0)


def test_live_snapshot_cpu_no_sharded_tensors():
    import torch
    from lingbotvla.utils.direct_hf_checkpoint import collect_full_model_on_cpu
    model = torch.nn.Linear(2,3)
    snap = collect_full_model_on_cpu(model)
    assert set(snap) == set(model.state_dict())
    assert all(t.device.type == 'cpu' for t in snap.values())
    with torch.no_grad():
        model.weight.add_(1)
    assert not torch.equal(snap['weight'], model.weight)


def test_direct_hf_atomic_export_without_dcp(tmp_path, monkeypatch):
    import torch
    from lingbotvla.utils.direct_hf_checkpoint import export_model_hf_direct
    import sys
    from types import ModuleType
    fake_models = ModuleType('lingbotvla.models')
    monkeypatch.setitem(sys.modules, 'lingbotvla.models', fake_models)

    calls = []
    def fake_save(path, state_dict, **kw):
        calls.append((dict(state_dict), kw))
        Path(path).mkdir(parents=True)
        from safetensors.torch import save_file
        save_file(dict(state_dict), str(Path(path)/'model.safetensors'))
    fake_models.save_model_weights = fake_save
    out = export_model_hf_direct(torch.nn.Linear(2,2), global_step=250,
                                 checkpoint_root=str(tmp_path))
    assert Path(out,'model.safetensors').is_file()
    assert calls and 'weight' in calls[0][0]
    assert not (tmp_path/'global_step_250'/'model').exists()
    assert not (tmp_path/'global_step_250'/'optimizer').exists()
    again = export_model_hf_direct(torch.nn.Linear(2,2), global_step=250,
                                   checkpoint_root=str(tmp_path))
    assert again != out
    assert '_retry_001' in again
    assert Path(out,'model.safetensors').is_file()
    assert Path(again,'model.safetensors').is_file()


def test_failed_export_keeps_no_final_or_temp(tmp_path, monkeypatch):
    import torch
    from lingbotvla.utils.direct_hf_checkpoint import export_model_hf_direct
    import sys
    from types import ModuleType
    fake_models = ModuleType('lingbotvla.models')
    monkeypatch.setitem(sys.modules, 'lingbotvla.models', fake_models)
    def fail(path, *a, **kw):
        Path(path).mkdir(parents=True)
        (Path(path)/'half.safetensors').write_text('incomplete')
        raise OSError('simulated disk full')
    fake_models.save_model_weights = fail
    with pytest.raises(RuntimeError, match='simulated disk full'):
        export_model_hf_direct(torch.nn.Linear(2,2), global_step=400,
                               checkpoint_root=str(tmp_path))
    assert not (tmp_path/'global_step_400'/'hf_ckpt').exists()
    assert not list(tmp_path.rglob('.hf_ckpt.tmp.*'))


def test_train_and_launcher_wiring():
    script=(ROOT/'tasks/vla/train_lingbotvla.py').read_text()
    ast.parse(script)
    assert 'hf_pass_interval' in script
    assert 'export_model_hf_direct(' in script
    assert '_al_hook.scheduler.state.finished' in script
    launcher=(ROOT/'experiment/robotwin/al_50task_bf16.sh').read_text()
    assert 'SAVE_EVERY=${SAVE_EVERY:-1000}' in launcher
    assert 'DCP_MODE=${DCP_MODE:-always}' in launcher
    assert 'HF_PASS_INTERVAL=${HF_PASS_INTERVAL:-2}' in launcher
    # 2026-10-10：`SAVE_HF_BOOL=false` 由**写死**改为**默认关 + 可用 SAVE_HF=1 打开**
    # （r2 run 因为写死 false ⇒ 只有 DCP、闭环评测没法跑）。
    assert 'SAVE_HF_BOOL=$([ "${SAVE_HF:-0}" = "1" ] && echo true || echo false)' in launcher
    assert 'SAVE_HF_BOOL=false' not in launcher, '不允许再写死（会再次导致只有 DCP、无 HF）'
    assert '--train.hf_pass_interval' in launcher
    assert 'SAVE_STEPS=$SAVE_EVERY' in launcher
    assert 'hf_milestones' in script

@pytest.mark.parametrize('smoke,expected_steps,expected_interval', [
    ('0', '1000', '2'), ('1', '0', '0'),
])
def test_real_launcher_dry_run_flags(tmp_path, smoke, expected_steps, expected_interval):
    """Exercise real shell and here-doc escaping, without torchrun/GPU."""
    import json
    import os
    import subprocess
    import sys
    splits = tmp_path / 'splits'
    splits.mkdir()
    (splits / 'manifest.json').write_text(json.dumps({
        'tasks': {'a': {'train_frames': 400}},
    }))
    (splits / 'combined.train_ids.json').write_text('[]')
    (splits / 'task_baseline.json').write_text('{}')
    (tmp_path / 'qwen').mkdir()
    env = dict(os.environ, PY=sys.executable, DRY_RUN='1', TB='0',
               SMOKE_NO_CHECKPOINT=smoke, MAX_STEPS='19500',
               SPLIT_DIR=str(splits), QWEN3VL=str(tmp_path / 'qwen'),
               AL_CFG=str(ROOT/'configs/auto_learning/smoke_gbs4_4task_tb5.yaml'),
               TRAIN_OUT=str(tmp_path/'out'), MICRO='1', GAS='4')
    proc = subprocess.run(['bash', str(ROOT/'experiment/robotwin/al_50task_bf16.sh')],
                          env=env, capture_output=True, text=True, errors='replace',
                          timeout=20)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert f'--train.save_steps       {expected_steps} \\' in proc.stdout
    assert f'--train.hf_pass_interval {expected_interval} \\' in proc.stdout
    assert '--train.save_hf_weights  false \\' in proc.stdout
    assert '--train.smoke_no_checkpoint' in proc.stdout
    assert '--train.dcp_save_mode    always \\' in proc.stdout


def test_closed_loop_script_has_precision_switch():
    """闭环脚本原来把精度写死成 fp32（`--use_fp32 true --use_bf16 false`）⇒ 加 `PRECISION` 开关。

    默认仍是 fp32（行为不变）；`PRECISION=bf16` 时传 `--use_bf16 true --use_fp32 false`。
    """
    from pathlib import Path
    script = (Path(__file__).resolve().parents[1] / "tools/closed_loop_eval.sh").read_text(
        encoding="utf-8")
    assert 'PRECISION=${PRECISION:-fp32}' in script
    assert 'USE_BF16=true; USE_FP32=false' in script
    assert '--use_fp32 "$USE_FP32" --use_bf16 "$USE_BF16"' in script
    assert '--use_fp32 true --use_bf16 false' not in script, "不允许再写死 fp32"


def test_policy_server_has_precision_switch():
    """常驻推理服务同样不能写死 fp32（两阶段流程的**阶段 1**）。"""
    from pathlib import Path
    script = (Path(__file__).resolve().parents[1] / "tools/policy_server.sh").read_text(
        encoding="utf-8")
    assert 'PRECISION=${PRECISION:-fp32}' in script
    assert 'USE_BF16=true; USE_FP32=false' in script
    assert '--use_bf16 $USE_BF16' in script and '--use_fp32 $USE_FP32' in script
    assert '--use_bf16 false' not in script, "不允许再写死 fp32"


def test_two_stage_closed_loop_scripts_exist():
    """两阶段闭环：阶段 1 常驻服务 + 阶段 2 只跑 sim（可反复换 task、模型不重载）。
    这是用户 2026-10-10 指定的用法（"先服务，后面可以换task"）。"""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "tools"
    server = (root / "policy_server.sh").read_text(encoding="utf-8")
    tasks = (root / "eval_tasks.sh").read_text(encoding="utf-8")
    assert "start" in server and "stop" in server and "status" in server
    assert "常驻" in server and "模型不重载" in server
    assert "TASKS        必填" in tasks and "只跑 sim 侧" in tasks
    assert "DRY_RUN" in tasks
