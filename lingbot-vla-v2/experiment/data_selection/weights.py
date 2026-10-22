"""Read immutable selection coordinates; never use filtered local sample IDs."""
import hashlib
import json
import math
from pathlib import Path


class SelectionWeights:
    def __init__(self, config_path):
        config_path = Path(config_path).resolve()
        c = json.loads(config_path.read_text())
        if 'policies' in c:
            from .inheritance import read_policies
            policies=read_policies(config_path)
            children=[SelectionWeights(path) for path in policies.values()]
            if not children:raise ValueError("empty importance registry")
            identities={(child.dataset_name,child.dataset_root) for child in children}
            if len(identities)!=1:raise ValueError("importance registry requires one merged dataset")
            self.dataset_name,self.dataset_root=identities.pop()
            self.keys=set();self.coordinate_weights={}
            for child in children:
                if self.keys & child.keys:raise ValueError("overlapping task importance coordinates")
                self.keys.update(child.keys)
                self.coordinate_weights.update(child.coordinate_weights)
            self.multipliers=set(self.coordinate_weights.values())
            self.multiplier=next(iter(self.multipliers)) if len(self.multipliers)==1 else None
            return
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
        frame_weights={}
        for row in json.loads(payload):
            pair = (row["episode_index"], row["frame_index"])
            if any(type(x) is not int or x < 0 for x in pair) or pair in self.keys:
                raise ValueError("invalid or duplicate selection coordinates")
            self.keys.add(pair)
            value=row.get("loss_weight",self.multiplier) if c.get("use_selected_frame_weights",False) else self.multiplier
            if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or value<=0:raise ValueError("invalid coordinate weight")
            frame_weights[pair]=float(value)
        if not self.keys: raise ValueError("empty selection")
        self.coordinate_weights=frame_weights
        self.multipliers=set(frame_weights.values())

    def weight(self, dataset_name, episode, frame):
        if dataset_name != self.dataset_name: return 1.
        return self.coordinate_weights.get((int(episode),int(frame)),1.)
