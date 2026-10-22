"""Read immutable selection coordinates; never use filtered local sample IDs."""
import hashlib
import json
import math
from pathlib import Path


class SelectionWeights:
    def __init__(self, config_path):
        config_path = Path(config_path).resolve()
        c = json.loads(config_path.read_text())
        if c.get("version") != 1 or not c.get("dataset_name"):
            raise ValueError("weight config needs version=1 and dataset_name")
        if c.get("normalization") != "sample_count":
            raise ValueError("weight config must explicitly select sample_count normalization")
        self.dataset_name = c["dataset_name"]
        self.dataset_root = str(Path(c["dataset_root"]).expanduser().resolve())
        self.multiplier = c.get("multiplier", 2.)
        if isinstance(self.multiplier, bool) or not isinstance(self.multiplier, (int,float)) or not math.isfinite(self.multiplier) or self.multiplier <= 0:
            raise ValueError("multiplier must be finite and positive")
        path = Path(c["selected_frames"])
        if not path.is_absolute(): path = config_path.parent / path
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != c["selected_frames_sha256"]:
            raise ValueError("selection fingerprint mismatch")
        self.keys = set()
        for row in json.loads(payload):
            pair = (row["episode_index"], row["frame_index"])
            if any(type(x) is not int or x < 0 for x in pair) or pair in self.keys:
                raise ValueError("invalid or duplicate selection coordinates")
            self.keys.add(pair)
        if not self.keys: raise ValueError("empty selection")

    def weight(self, dataset_name, episode, frame):
        if dataset_name != self.dataset_name: return 1.
        return float(self.multiplier if (int(episode), int(frame)) in self.keys else 1.)
