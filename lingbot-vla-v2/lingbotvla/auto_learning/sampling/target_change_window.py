"""Static, CPU-only target-change windows. No model, force or actual-qpos claims.

Cache identity uses readable source/settings/episode coordinates, never local sample IDs.
Only this preparation code writes caches; the scheduler is strictly a cache reader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

from ..decision.metrics import normalize_weights


@dataclass(frozen=True)
class WindowSettings:
    window_seconds: float = 0.25
    gripper_speed: float = 0.1  # raw normalized gripper command / second
    wrist_speed: float = 0.2  # raw radians / second, no angle wrapping
    release_ratio: float = 0.5
    persistence_frames: int = 2
    window_weight: float = 2.0

    @classmethod
    def from_config(cls, cfg):
        return cls(
            **{
                key: getattr(cfg, "target_change_" + key)
                for key in cls.__dataclass_fields__
            }
        )

    def validate(self):
        for key in ("gripper_speed", "wrist_speed", "window_weight"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
        if self.window_weight < 1:
            raise ValueError("window_weight must be at least 1")
        if (
            isinstance(self.window_seconds, bool)
            or not isinstance(self.window_seconds, (int, float))
            or not math.isfinite(self.window_seconds)
            or self.window_seconds < 0
        ):
            raise ValueError("window_seconds must be finite and nonnegative")
        if (
            isinstance(self.release_ratio, bool)
            or not isinstance(self.release_ratio, (int, float))
            or not math.isfinite(self.release_ratio)
            or not 0 < self.release_ratio < 1
        ):
            raise ValueError("release_ratio must be in (0,1)")
        if type(self.persistence_frames) is not int or self.persistence_frames < 1:
            raise ValueError("persistence_frames must be a positive integer")


def joint_mapping(embodiment, wrist_names):
    """robot.py reads arm_joints_name[0/1], then appends each gripper command.

    robotwin.yaml proves the 6+1+6+1 raw layout. Wrist selection must explicitly
    name joints present in the *actual* embodiment config, no last-two heuristic.
    """
    arms = embodiment.get("arm_joints_name")
    if (
        not isinstance(arms, list)
        or len(arms) != 2
        or any(
            not isinstance(a, list)
            or len(a) != 6
            or any(not isinstance(name, str) or not name for name in a)
            or len(set(a)) != 6
            for a in arms
        )
    ):
        raise ValueError(
            "expected actual embodiment arm_joints_name with two six-joint lists"
        )
    if (
        not isinstance(wrist_names, list)
        or len(wrist_names) != 2
        or any(
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
            for names in wrist_names
        )
    ):
        raise ValueError(
            "specify target_change_wrist_joint_names as [left_names, right_names]"
        )
    result = []
    for side, offset, names, selected in zip(
        ("left", "right"), (0, 7), arms, wrist_names
    ):
        for name in selected:
            if name not in names:
                raise ValueError(
                    f"{side} wrist joint {name!r} is absent from arm_joints_name={names}"
                )
            result.append(
                dict(
                    name=f"{side}:{name}",
                    index=offset + names.index(name),
                    kind="wrist",
                )
            )
        result.append(
            dict(name=f"{side}:gripper_command", index=offset + 6, kind="gripper")
        )
    return result


def scan_episode(rows, fps, mapping, settings):
    settings.validate()
    if not math.isfinite(fps) or fps <= 0 or not rows:
        raise ValueError("nonempty trajectory and positive FPS required")
    frames = [int(r["index"]) for r in rows]  # resolver.frame_id is absolute index
    if len(set(frames)) != len(frames) or frames != sorted(frames):
        raise ValueError("trajectory absolute frame IDs must be unique and ordered")
    positions = [int(r["frame_index"]) for r in rows]
    if positions != list(range(len(rows))):
        raise ValueError("trajectory must contain all frames from zero, without gaps")
    timestamps = [float(r["timestamp"]) for r in rows]
    if any(not math.isfinite(t) for t in timestamps) or any(
        b <= a or abs((b - a) * fps - 1) > 0.1
        for a, b in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("trajectory timestamps disagree with FPS or have gaps")
    events, signals = [], []
    n = len(rows)
    for joint in mapping:
        values = [float(r["observation.state"][joint["index"]]) for r in rows]
        if any(not math.isfinite(v) for v in values):
            raise ValueError("nonfinite raw target signal")
        # Median of adjacent velocities suppresses isolated spikes. Persistence
        # and separate on/off thresholds keep jitter from creating event pairs.
        velocity = [0.0] + [(values[i] - values[i - 1]) * fps for i in range(1, n)]
        speed = [
            abs(statistics.median(velocity[max(0, i - 1) : min(n, i + 2)]))
            for i in range(n)
        ]
        high = (
            settings.gripper_speed
            if joint["kind"] == "gripper"
            else settings.wrist_speed
        )
        low = high * settings.release_ratio
        active, count = False, 0
        for i, value in enumerate(speed):
            qualifies = value <= low if active else value >= high
            count = count + 1 if qualifies else 0
            if count >= settings.persistence_frames:
                event_i = i - settings.persistence_frames + 1
                events.append(
                    dict(
                        signal=joint["name"],
                        kind="stop" if active else "start",
                        frame=event_i,
                        absolute_index=frames[event_i],
                        seconds=timestamps[event_i],
                        value=values[event_i],
                        speed=speed[event_i],
                        high=high,
                        low=low,
                    )
                )
                active, count = not active, 0
        signals.append(dict(**joint, values=values, speed=speed, high=high, low=low))
    radius = math.ceil(settings.window_seconds * fps)
    windows = []
    for event in sorted(events, key=lambda e: e["frame"]):
        lo, hi = max(0, event["frame"] - radius), min(n - 1, event["frame"] + radius)
        if windows and lo <= windows[-1][1] + 1:
            windows[-1][1] = max(windows[-1][1], hi)
        else:
            windows.append([lo, hi])
    weights = [1.0] * n
    for lo, hi in windows:
        weights[lo : hi + 1] = [settings.window_weight] * (hi - lo + 1)
    return dict(
        frame_ids=frames,
        timestamps=timestamps,
        events=events,
        windows=windows,
        weights=weights,
        signals=signals,
        frames=n,
        window_frames=sum(hi - lo + 1 for lo, hi in windows),
    )


def cache_path(cache_dir, task):
    if not task or Path(task).name != task or task in (".", ".."):
        raise ValueError("task must be a plain filename")
    return Path(cache_dir) / "target_change_window" / (task + ".json")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def cache_identity(cfg, fps, mapping):
    return dict(
        schema=1,
        method="target_change_window",
        fps=fps,
        settings=asdict(WindowSettings.from_config(cfg)),
        joint_mapping=mapping,
    )


def validate_cache(raw, cfg):
    if raw.get("schema") != 1 or raw.get("method") != "target_change_window":
        raise ValueError("unsupported target-change cache schema")
    if raw.get("settings") != asdict(WindowSettings.from_config(cfg)):
        raise ValueError(
            "target-change settings changed; run preparation with --refresh"
        )


def prepare_task(
    cfg, root, task, episode_ids, fps, mapping, read_episode, *, refresh=False
):
    """Reuse a task cache or scan exactly its train episodes. read_episode is lazy."""
    WindowSettings.from_config(cfg).validate()
    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("nonempty unique train episode IDs required")
    path = cache_path(cfg.target_change_cache_dir, task)
    identity = dict(
        **cache_identity(cfg, fps, mapping),
        source_root=str(Path(root).resolve()),
        task=task,
        train_episode_ids=sorted(episode_ids),
    )
    if path.exists() and not refresh:
        raw = json.loads(path.read_text())
        if any(raw.get(key) != value for key, value in identity.items()):
            raise ValueError(
                f"{path}: source/settings/joints/train episodes changed; inspect then --refresh"
            )
        validate_cache(raw, cfg)
        return raw
    episodes = {}
    for ep in sorted(episode_ids):
        rows = read_episode(ep)
        if not rows or {int(r["episode_index"]) for r in rows} != {ep}:
            raise ValueError(f"task={task} episode={ep}: missing/wrong episode rows")
        if any(len(r["observation.state"]) != 14 for r in rows):
            raise ValueError("expected unnormalized Robotwin raw state with 14 targets")
        episodes[str(ep)] = scan_episode(
            rows, fps, mapping, WindowSettings.from_config(cfg)
        )
    raw = dict(**identity, episodes=episodes)
    total_weight = sum(sum(e["weights"]) for e in episodes.values())
    for data in episodes.values():
        data["probabilities"] = [w / total_weight for w in data["weights"]]
    n = sum(e["frames"] for e in episodes.values())
    covered = sum(e["window_frames"] for e in episodes.values())
    raw["summary"] = dict(
        frames=n,
        window_frames=covered,
        coverage=covered / max(1, n),
        events=sum(len(e["events"]) for e in episodes.values()),
        window_probability_mass=(
            covered
            * cfg.target_change_window_weight
            / (covered * cfg.target_change_window_weight + n - covered)
        ),
    )
    atomic_json(path, raw)
    return raw


class TargetChangeScanner:
    def __init__(self, cfg, resolver):
        self.cfg, self.resolver = cfg, resolver
        self._results = {}

    def scan(self, record, entry, *, force_fresh=False):
        from .hardness_scan import HardnessScan

        if record.task_name in self._results:
            result = self._results[record.task_name]
            if set(result.probs) != set(record.sample_ids) or sorted(
                result.scanned_traj_ids
            ) != sorted(entry.train_traj_ids):
                raise ValueError(
                    "static scanner task sample set changed during training"
                )
            return result
        path = cache_path(self.cfg.target_change_cache_dir, record.task_name)
        if not path.is_file():
            raise ValueError(
                f"{path}: static cache missing; run prepare_target_change_windows.py before training"
            )
        raw = json.loads(path.read_text())
        validate_cache(raw, self.cfg)
        if not self.cfg.target_change_dataset_root or raw.get("source_root") != str(
            Path(self.cfg.target_change_dataset_root).resolve()
        ):
            raise ValueError(
                "target-change cache source_root mismatch; set target_change_dataset_root"
            )
        import yaml

        embodiment = yaml.safe_load(
            Path(self.cfg.target_change_embodiment_config).read_text()
        )
        if raw.get("joint_mapping") != joint_mapping(
            embodiment, self.cfg.target_change_wrist_joint_names
        ):
            raise ValueError("target-change cache joint mapping mismatch")
        if raw["task"] != record.task_name or raw["train_episode_ids"] != sorted(
            entry.train_traj_ids
        ):
            raise ValueError("target-change cache task/train episode mismatch")
        coordinates = {}
        for ep, data in raw["episodes"].items():
            if len(data["frame_ids"]) != len(data["weights"]):
                raise ValueError(
                    "target-change cache coordinate/weight length mismatch"
                )
            for frame, weight in zip(data["frame_ids"], data["weights"]):
                key = (int(ep), int(frame))
                if key in coordinates or not math.isfinite(weight) or weight <= 0:
                    raise ValueError("duplicate coordinate or invalid weight in cache")
                coordinates[key] = float(weight)
        weights, used = {}, set()
        for sid in record.sample_ids:
            ref = self.resolver.resolve(record.task_name, sid)
            key = (int(ref.episode_id), int(ref.frame_id))
            if key not in coordinates or key in used:
                raise ValueError(f"target-change cache/resolver mismatch: {key}")
            weights[sid] = coordinates[key]
            used.add(key)
        if used != set(coordinates):
            raise ValueError("target-change cache/current train frame set mismatch")
        result = HardnessScan(
            task=record.task_name,
            version=1,
            scanned_traj_ids=list(entry.train_traj_ids),
            scanned_sample_ids=list(weights),
            losses={},
            difficulties={},
            weights=weights,
            probs=normalize_weights(weights),
            n_total=len(weights),
            n_scanned=len(weights),
            unscored_difficulty=0.0,
            method="target_change_window",
        )
        self._results[record.task_name] = result
        return result


def write_report(raw, path, video_paths=None):
    """Standalone HTML: raw target curves, event windows and optional source videos.

    Clicking an event seeks the episode's videos; report uses trajectory-relative
    time. The user must confirm camera/video FPS and action alignment visually.
    """
    parts = [
        '<!doctype html><meta charset="utf-8"><title>Target change windows</title>',
        "<style>body{font:16px sans-serif;max-width:1200px;margin:auto}svg{width:100%}video{width:32%}</style>",
        "<h1>"
        + html.escape(raw["task"])
        + "</h1><p>控制目标变化，不代表真实接触力。红区为加权锚点窗口；权重不按重复窗口累乘。</p>",
        "<pre>"
        + html.escape(
            json.dumps(
                {k: v for k, v in raw.items() if k != "episodes"},
                ensure_ascii=False,
                indent=2,
            )
        )
        + "</pre>",
    ]
    for ep, data in raw["episodes"].items():
        parts.append(f"<section><h2>Episode {ep}</h2>")
        for video in (video_paths or {}).get(int(ep), []):
            parts.append(
                '<video controls preload="metadata" src="'
                + html.escape(str(video), quote=True)
                + '"></video>'
            )
        for signal in data["signals"]:
            values = signal["values"]
            lo, hi = min(values), max(values)
            scale = max(hi - lo, 1e-6)
            denom = max(1, len(values) - 1)
            parts.append(
                "<p>"
                + html.escape(signal["name"])
                + f' raw range [{lo:.5g},{hi:.5g}] speed thresholds {signal["high"]}/{signal["low"]}</p><svg viewBox="0 0 1000 120">'
            )
            for a, b in data["windows"]:
                parts.append(
                    f'<rect x="{a/denom*1000}" y="0" width="{max(1,(b-a)/denom*1000)}" height="120" fill="red" opacity=".12"/>'
                )
            points = " ".join(
                f"{i/denom*1000:.2f},{110-(v-lo)/scale*100:.2f}"
                for i, v in enumerate(values)
            )
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="blue"/></svg>'
            )
            top = max(max(signal["speed"]), signal["high"]) * 1.1
            points = " ".join(
                f"{i/denom*1000:.2f},{110-v/top*100:.2f}"
                for i, v in enumerate(signal["speed"])
            )
            parts.append('<svg viewBox="0 0 1000 120">')
            for level in (signal["high"], signal["low"]):
                y = 110 - level / top * 100
                parts.append(
                    f'<line x1="0" y1="{y}" x2="1000" y2="{y}" stroke="red" stroke-dasharray="4"/>'
                )
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="green"/></svg>'
            )
        for e in data["events"]:
            seconds = e["seconds"] - data["timestamps"][0]
            label = html.escape(
                f'{e["signal"]} {e["kind"]} frame={e["frame"]} absolute={e["absolute_index"]} t={seconds:.3f}s speed={e["speed"]:.4g}'
            )
            parts.append(
                f"<p><button onclick=\"this.closest('section').querySelectorAll('video').forEach(v=>v.currentTime={seconds})\">{label}</button></p>"
            )
        parts.append("</section>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")
