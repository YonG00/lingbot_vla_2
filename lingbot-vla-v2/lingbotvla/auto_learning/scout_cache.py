"""Bootstrap-only GMean Scout cache with strict provenance and atomic writes.

Explicitly **not** a Rescan or learned-weight cache. Fingerprint is constructed
from hashes of real weight files, eval source, norm, manifest and threshold table.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

VERSION = 1
MAX_JSON_BYTES = 1_000_000


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def provenance(*, weight_files: Sequence[str | Path], sources: Mapping[str, str | Path],
               options: Mapping[str, Any]) -> str:
    """Hash actual checkpoint shards (not only names, mtimes or 'step500')."""
    if not weight_files or not sources or 'inference_dtype' not in options:
        raise ValueError('Scout cache requires checkpoint shards, eval sources and dtype')
    manifest = {
        'schema': VERSION,
        'weights': [(str(Path(p).name), sha256_file(p)) for p in sorted(weight_files, key=str)],
        'sources': {k: sha256_file(v) for k, v in sorted(sources.items())},
        'options': dict(options),
    }
    if len({n for n, _ in manifest['weights']}) != len(manifest['weights']):
        raise ValueError('ambiguous checkpoint shard names')
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def scout_key(task: str, episode_ids: Sequence[int]) -> str:
    ids = list(map(int, episode_ids))
    if not task or not ids or len(set(ids)) != len(ids):
        raise ValueError('invalid task/episode ID set')
    return hashlib.sha256(json.dumps([task, ids], separators=(',', ':')).encode()).hexdigest()


class BootstrapScoutCache:
    def __init__(self, root: str | Path, *, fingerprint: str):
        if len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
            raise ValueError('invalid provenance fingerprint')
        self.path = Path(root) / fingerprint
        self.fingerprint = fingerprint

    def load(self, task: str, ids: Sequence[int]) -> dict | None:
        key = scout_key(task, ids)
        p = self.path / f'{key}.json'
        if not p.is_file() or p.stat().st_size > MAX_JSON_BYTES:
            return None
        try:
            raw = json.loads(p.read_text(encoding='utf-8'))
            if (raw.get('version') != VERSION or raw.get('fingerprint') != self.fingerprint
                    or raw.get('task') != task or raw.get('episode_ids') != list(map(int, ids))):
                return None
            m = raw['metrics']
            if not isinstance(m, dict) or m.get('task') != task or m.get('episode_ids') != list(map(int, ids)):
                return None
            per = m.get('per_traj_mse', {})
            if not isinstance(per, dict) or set(per) != set(map(str, ids)):
                return None
            vals = [float(v) for v in per.values()]
            if any(not math.isfinite(v) or v < 0 for v in vals):
                return None
            nmse, mse, base = (float(m[k]) for k in ('nmse','mse','baseline_mse'))
            if (m.get('n_trajs') != len(ids) or m.get('metric_valid') is not True
                    or not all(math.isfinite(v) for v in (nmse,mse,base))
                    or min(nmse,mse) < 0 or base <= 0
                    or not math.isclose(nmse, mse/base, rel_tol=1e-5, abs_tol=1e-8)
                    or not math.isclose(mse, sum(vals)/len(vals), rel_tol=1e-4, abs_tol=1e-7)):
                return None
            return m
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def store(self, task: str, ids: Sequence[int], metrics: Mapping[str, Any]) -> None:
        key = scout_key(task, ids)
        if (metrics.get('task') != task or list(metrics.get('episode_ids', [])) != list(map(int, ids))):
            raise ValueError('cannot cache mismatched task or IDs')
        payload = {'version': VERSION, 'fingerprint': self.fingerprint, 'task': task,
                   'episode_ids': list(map(int, ids)), 'metrics': dict(metrics)}
        blob = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(',', ':'))
        if len(blob.encode()) > MAX_JSON_BYTES:
            raise ValueError('scout cache record too large')
        self.path.mkdir(parents=True, exist_ok=True)
        target = self.path / f'{key}.json'
        tmp = self.path / f'.{key}.{os.getpid()}.tmp'
        try:
            with open(tmp, 'x', encoding='utf-8') as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
