"""Verify frozen training source and curriculum inputs without importing torch."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def verify(root=ROOT):
    lock = json.loads((root / 'configs/reproduction/source_lock.json').read_text())
    errors = []
    for name, expected in lock['sha256'].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(name)
    if errors:
        raise ValueError('Frozen source mismatch: ' + ', '.join(errors))
    return len(lock['sha256'])

if __name__ == '__main__':
    print('REPRODUCTION_SOURCE_LOCK_PASS', verify())
