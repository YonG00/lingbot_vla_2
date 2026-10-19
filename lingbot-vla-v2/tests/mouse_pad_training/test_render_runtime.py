"""CPU-only startup contracts; never import SAPIEN or torch."""
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def runtime(monkeypatch):
    path = Path(__file__).resolve().parents[3] / 'RoboTwin-lingbot/envs/render_runtime.py'
    spec = importlib.util.spec_from_file_location('render_runtime_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv('ROBOTWIN_OIDN_LIBRARY_DIR', raising=False)
    monkeypatch.delitem(sys.modules, 'sapien', raising=False)
    return module


def test_explicit_bundled_does_not_load(runtime, monkeypatch):
    monkeypatch.setenv('ROBOTWIN_OIDN_LIBRARY_DIR', 'bundled')
    monkeypatch.setattr(runtime.ctypes, 'CDLL', lambda *a, **kw: pytest.fail('unexpected load'))
    runtime.configure_oidn()


def test_no_install_preserves_existing_environment(runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, '__file__', str(tmp_path / 'repo/envs/render_runtime.py'))
    monkeypatch.setattr(runtime.ctypes, 'CDLL', lambda *a, **kw: pytest.fail('unexpected load'))
    runtime.configure_oidn()


def test_missing_explicit_runtime_fails(runtime, monkeypatch, tmp_path):
    monkeypatch.setenv('ROBOTWIN_OIDN_LIBRARY_DIR', str(tmp_path))
    with pytest.raises(RuntimeError, match='Incomplete OIDN'):
        runtime.configure_oidn()


def test_late_override_fails(runtime, monkeypatch, tmp_path):
    monkeypatch.setenv('ROBOTWIN_OIDN_LIBRARY_DIR', str(tmp_path))
    monkeypatch.setitem(sys.modules, 'sapien', object())
    with pytest.raises(RuntimeError, match='before sapien'):
        runtime.configure_oidn()


@pytest.mark.parametrize('device_type', [1, 2, 3])
def test_cuda_required_before_bypassing_bundled_loader(runtime, monkeypatch, tmp_path, device_type):
    for name in ('libOpenImageDenoise_core.so.2.5.1', 'libOpenImageDenoise.so.2.5.1'):
        (tmp_path / name).touch()
    monkeypatch.setenv('ROBOTWIN_OIDN_LIBRARY_DIR', str(tmp_path))
    monkeypatch.delitem(sys.modules, 'sapien._oidn_tricks', raising=False)
    class Function:
        def __init__(self, value): self.value = value
        def __call__(self, *args): return self.value
    class Library:
        oidnGetNumPhysicalDevices = Function(1)
        oidnGetPhysicalDeviceInt = Function(device_type)
        oidnGetPhysicalDeviceString = Function(b'test GPU')
    monkeypatch.setattr(runtime.ctypes, 'CDLL', lambda *a, **kw: Library())
    if device_type != 3:
        with pytest.raises(RuntimeError, match='no CUDA'):
            runtime.configure_oidn()
        assert 'sapien._oidn_tricks' not in sys.modules
    else:
        # Restore the process module table after checking this injection.
        monkeypatch.setitem(sys.modules, 'sapien._oidn_tricks', None)
        runtime.configure_oidn()
        assert sys.modules['sapien._oidn_tricks'].__name__ == 'sapien._oidn_tricks'
