"""Read immutable selection coordinates; never use filtered local sample IDs."""
import hashlib
import json
import math
from pathlib import Path


class SelectionWeights:
    def __init__(self, config_path):
        config_path = Path(config_path).resolve()
        c = json.loads(config_path.read_text())
        self.stage_tables = {}
        if 'policies' in c:
            from .inheritance import read_policies
            policies=read_policies(config_path)
            children=[SelectionWeights(path) for path in policies.values()]
            if not children:raise ValueError("empty importance registry")
            identities={(child.dataset_name,child.dataset_root) for child in children}
            if len(identities)!=1:raise ValueError("importance registry requires one merged dataset")
            self.dataset_name,self.dataset_root=identities.pop()
            self.keys=set();self.coordinate_weights={};self.action_overrides={};self.action_expected_dimensions=set()
            for child in children:
                if self.keys & child.keys:raise ValueError("overlapping task importance coordinates")
                self.keys.update(child.keys)
                self.coordinate_weights.update(child.coordinate_weights)
                self.action_overrides.update(child.action_overrides)
                self.action_expected_dimensions.update(child.action_expected_dimensions)
                if self.stage_tables.keys() & child.stage_tables.keys():raise ValueError("overlapping staged episodes")
                self.stage_tables.update(child.stage_tables)
            self.multipliers=set(self.coordinate_weights.values())
            self.multiplier=next(iter(self.multipliers)) if len(self.multipliers)==1 else None
            return
        if c.get("version") != 1 or not c.get("dataset_name"):
            raise ValueError("weight config needs version=1 and dataset_name")
        if c.get("normalization") != "sample_count" and not (c.get("normalization")=="weight_sum" and "stage_table"in c):
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
        self.action_overrides = {}
        self.action_expected_dimensions = set()
        arm_weights = c.get("arm_action_weights")
        layout = c.get("arm_action_dimensions")
        if arm_weights is not None:
            if not isinstance(arm_weights,dict) or set(arm_weights)!={"left","right"} or not isinstance(layout,dict) or set(layout)!={"left","right"}:
                raise ValueError("explicit left/right action weights and dimensions required")
            for arm in arm_weights:
                value=arm_weights[arm]
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError("invalid arm action weight")
                dims=layout[arm]
                if not isinstance(dims,list) or not dims or any(type(v)is not int or v<0 for v in dims) or len(set(dims))!=len(dims):raise ValueError("invalid arm dimensions")
            if set(layout["left"]) & set(layout["right"]):raise ValueError("overlapping arm action dimensions")
            self.action_expected_dimensions=set(layout["left"]+layout["right"])
        frame_weights={}
        for row in json.loads(payload):
            pair = (row["episode_index"], row["frame_index"])
            if any(type(x) is not int or x < 0 for x in pair) or pair in self.keys:
                raise ValueError("invalid or duplicate selection coordinates")
            self.keys.add(pair)
            value=row.get("loss_weight",self.multiplier) if c.get("use_selected_frame_weights",False) else self.multiplier
            if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or value<=0:raise ValueError("invalid coordinate weight")
            frame_weights[pair]=float(value)
            if arm_weights is not None:
                arms={label.get("arm") for label in row.get("labels",[])}
                if not arms or not arms<={"left","right"}:raise ValueError("arm labels required for action weights")
                self.action_overrides[pair]={dim:float(arm_weights[arm] if arm in arms else 1.)/float(value) for arm,dims in layout.items() for dim in dims}
        if not self.keys: raise ValueError("empty selection")
        if 'stage_table' in c:
            if arm_weights is not None or self.multiplier!=1 or c.get('use_selected_frame_weights',False):raise ValueError('stage policy must replace legacy task weighting')
            from .full_stages import StageTable
            stage=StageTable(c,config_path.parent)
            expected={(e['episode_index'],f)for e in stage.table['episodes']for f in range(e['length'])}
            if self.keys!=expected:raise ValueError('stage policy must register every original frame')
            self.stage_tables={ep:stage for ep in stage.episodes}
        self.coordinate_weights=frame_weights
        self.multipliers=set(frame_weights.values())

    def weight(self, dataset_name, episode, frame):
        if dataset_name != self.dataset_name: return 1.
        return self.coordinate_weights.get((int(episode),int(frame)),1.)

    def relative_action_weights(self, dataset_name, episode, frame, width):
        """Corrections multiply FM only; scalar sample weight still governs shared losses."""
        if not self.action_overrides:return None
        if type(width)is not int or width<=0:raise ValueError("invalid action width")
        result=[1.]*width
        if dataset_name==self.dataset_name:
            for dim,value in self.action_overrides.get((int(episode),int(frame)),{}).items():
                if dim>=width:raise ValueError("action dimension outside model output")
                result[dim]=value
        return result
