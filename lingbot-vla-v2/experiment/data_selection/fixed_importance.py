"""User-approved task importance independent of main/replay role and source HF."""
import json
from pathlib import Path
from .weights import SelectionWeights

def fixed_policies(root,active_tasks,dataset_root):
    root=Path(root);catalog=root/'configs/data_selection/fixed_task_importance.json'
    if not catalog.exists():return {}
    value=json.loads(catalog.read_text())
    if value.get('version')!=1 or not isinstance(value.get('policies'),dict):raise ValueError('invalid fixed importance catalog')
    result={}
    for task,relative in value['policies'].items():
        if task not in active_tasks:continue
        path=root/relative;cfg=json.loads(path.read_text());policy=SelectionWeights(path)
        if cfg.get('task')!=task or policy.dataset_root!=str(Path(dataset_root).resolve()) or policy.dataset_name!='robotwin':raise ValueError('fixed task importance dataset/task mismatch')
        result[task]=path
    return result
