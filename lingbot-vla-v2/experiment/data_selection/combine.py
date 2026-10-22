"""Union stable selection coordinates; overlapping rules never multiply weights."""
import hashlib
import json
from pathlib import Path


def combine_directories(directories):
    summary0=None;selected={};sources=[];counts=[]
    for directory in directories:
        directory=Path(directory);summary=json.loads((directory/'summary.json').read_text())
        if summary0 is None: summary0=summary
        for key in ('task','fps','numeric_source_sha256','spec_sha256'):
            if summary.get(key) is None or summary[key]!=summary0.get(key):raise ValueError('incompatible selection '+key)
        data=(directory/'selected_frames.json').read_bytes();rows=json.loads(data);seen=set()
        for row in rows:
            pair=(row['episode_index'],row['frame_index'])
            if any(type(v) is not int or v<0 for v in pair) or pair in seen:raise ValueError('invalid/duplicate coordinates')
            seen.add(pair)
            target=selected.setdefault(pair,dict(episode_index=pair[0],frame_index=pair[1],labels=[]))
            for label in row.get('labels',[]):
                label=dict(label,selection_rule=summary['rule_version'])
                if label not in target['labels']:target['labels'].append(label)
        counts.append(len(seen));sources.append(dict(rule_version=summary['rule_version'],selected_frames=len(seen),selected_sha256=hashlib.sha256(data).hexdigest()))
    if len(sources)<2 or not selected:raise ValueError('need at least two nonempty selections')
    total=summary0.get('frames',summary0.get('total_frames'))
    result=dict(task=summary0['task'],fps=summary0['fps'],numeric_source_sha256=summary0['numeric_source_sha256'],spec_sha256=summary0['spec_sha256'],rule_version='grasp_and_second_placement_union_v1',combination='union_no_stacking',selected_frames=len(selected),total_frames=total,selected_fraction=len(selected)/total,duplicate_memberships=sum(counts)-len(selected),sources=sources,training_modified=False)
    return result,[selected[k]for k in sorted(selected)]
