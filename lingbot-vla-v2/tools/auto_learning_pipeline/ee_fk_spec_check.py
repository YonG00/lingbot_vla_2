#!/usr/bin/env python3
"""Check FK chain joint names against a RoboTwin embodiment config.

This is intentionally a no-torch preflight.  It validates the only part that
can be checked without loading SAPIEN: the exact left/right active-joint names
and order used by RoboTwin versus the names recorded in the FK spec.
Geometry still needs the separate SAPIEN pose parity check described in
``docs/auto_learning/ee_fk_loss_zh.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _load(path: Path) -> Any:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise SystemExit("PyYAML is required for RoboTwin config checks") from exc
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _names(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain [left, right] joint-name lists")
    result = []
    for side, names in zip(("left", "right"), value):
        if not isinstance(names, list) or len(names) != 6 or not all(isinstance(n, str) and n for n in names):
            raise ValueError(f"{label}[{side}] must contain exactly six non-empty names")
        if len(set(names)) != len(names):
            raise ValueError(f"{label}[{side}] contains duplicate names")
        result.append(names)
    return result


def compare(spec: Mapping[str, Any], embodiment: Mapping[str, Any]) -> dict[str, Any]:
    actual = _names(embodiment.get("arm_joints_name"), "arm_joints_name")
    observed = []
    for side in ("left", "right"):
        chain = spec.get(side)
        if not isinstance(chain, Mapping):
            raise ValueError(f"spec.{side} must be a mapping")
        joints = chain.get("joints")
        if not isinstance(joints, list) or len(joints) != 6:
            raise ValueError(f"spec.{side}.joints must contain exactly six entries")
        names = [joint.get("name") if isinstance(joint, Mapping) else None for joint in joints]
        observed.append(names)
    matches = {side: actual[idx] == observed[idx] for idx, side in enumerate(("left", "right"))}
    return {
        "left": {"expected": actual[0], "spec": observed[0], "match": matches["left"]},
        "right": {"expected": actual[1], "spec": observed[1], "match": matches["right"]},
        "match": all(matches.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="verified FK JSON spec")
    parser.add_argument("--embodiment-config", type=Path, required=True, help="RoboTwin config.yml")
    args = parser.parse_args()
    report = compare(_load(args.spec), _load(args.embodiment_config))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
