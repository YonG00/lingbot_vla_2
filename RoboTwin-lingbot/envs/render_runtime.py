"""Load the optional OIDN runtime before SAPIEN imports its bundled 2.0.1.

No SAPIEN/physics upgrade or site-packages modification is involved.
"""
import ctypes
import os
from pathlib import Path
import sys
import types

_handles = []


def configure_oidn():
    default = (Path(__file__).resolve().parents[2] / '.runtime' / 'oidn-2.5.1'
               / 'oidn-2.5.1.x86_64.linux' / 'lib')
    setting = os.environ.get('ROBOTWIN_OIDN_LIBRARY_DIR')
    if setting == 'bundled':
        return
    directory = Path(setting).expanduser().resolve() if setting else default
    if not setting and not directory.exists():
        return
    if 'sapien' in sys.modules:
        raise RuntimeError('OIDN override must load before sapien; restart the simulator')
    core = directory / 'libOpenImageDenoise_core.so.2.5.1'
    library = directory / 'libOpenImageDenoise.so.2.5.1'
    for path in (core, library):
        if not path.is_file():
            raise RuntimeError(f'Incomplete OIDN 2.5.1 runtime: {path}')
    # Keep both handles alive. Global symbols let the SAPIEN renderer reuse
    # this ABI-compatible OIDN 2 runtime instead of loading a second version.
    _handles.append(ctypes.CDLL(str(core), mode=ctypes.RTLD_GLOBAL))
    lib = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    _handles.append(lib)
    lib.oidnGetNumPhysicalDevices.restype = ctypes.c_int
    lib.oidnGetPhysicalDeviceInt.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.oidnGetPhysicalDeviceInt.restype = ctypes.c_int
    lib.oidnGetPhysicalDeviceString.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.oidnGetPhysicalDeviceString.restype = ctypes.c_char_p
    devices = [i for i in range(lib.oidnGetNumPhysicalDevices())
               if lib.oidnGetPhysicalDeviceInt(i, b'type') == 3]  # oidn.h: OIDN_DEVICE_TYPE_CUDA
    if not devices:
        raise RuntimeError('OIDN 2.5.1 found no CUDA device; refusing noisy evaluation')
    # SAPIEN 3.0.0b1 unconditionally dlopens absolute 2.0.1 paths here.
    # Bypass only that preload shim; SAPIEN itself remains unchanged.
    sys.modules['sapien._oidn_tricks'] = types.ModuleType('sapien._oidn_tricks')
    names = [lib.oidnGetPhysicalDeviceString(i, b'name').decode() for i in devices]
    print(f'[render_runtime] OIDN 2.5.1 CUDA={names} library={library}', flush=True)
