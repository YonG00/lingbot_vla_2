"""Versioned, default-off TCP geometry settings; standard library only."""
import json,math,hashlib
from pathlib import Path
STAGES=('idle_or_prepare','reach_aim','fine_align','pre_contact','gripper_close','grasp_lift','transport','target_align','target_hold','unknown')
def load_settings(path):
    path=Path(path).resolve();c=json.loads(path.read_text())
    allowed={'version','dataset_name','dataset_root','spec_path','spec_sha256','cache_metadata','cache_metadata_sha256','tcp_offset_m','huber_delta','time_max','absolute','relative'}
    if set(c)-allowed or c.get('version')!=1:raise ValueError('TCP geometry settings schema')
    for key in ['tcp_offset_m','huber_delta','time_max']:
        v=c.get(key)
        if isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or v<=0:raise ValueError('invalid geometry '+key)
    if c['time_max']>1:raise ValueError('geometry time_max must be <=1')
    for key in ['absolute','relative']:
        section=c[key]
        if set(section)-{'enabled','position_weight','rotation_weight','stages','stage_weights'}or type(section.get('enabled'))is not bool:raise ValueError('geometry section schema')
        for field in ['position_weight','rotation_weight']:
            v=section[field]
            if isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or v<0:raise ValueError('invalid geometry coefficient')
        if section['enabled']and section['position_weight']+section['rotation_weight']<=0:raise ValueError('enabled geometry objective requires positive coefficient')
        if not isinstance(section.get('stages',[]),list)or set(section.get('stages',[]))-set(STAGES):raise ValueError('unknown geometry gate stage')
        if not isinstance(section.get('stage_weights',{}),dict):raise ValueError('geometry stage weights must be a mapping')
        for name,value in section.get('stage_weights',{}).items():
            if name not in STAGES or isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or value<0:raise ValueError('invalid geometry stage weight')
    for key in ['spec_path','cache_metadata']:
        p=Path(c[key]);c[key]=str((p if p.is_absolute()else path.parent/p).resolve())
    c['dataset_root']=str(Path(c['dataset_root']).resolve())
    return c

def checked(path,sha):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('TCP geometry artifact fingerprint mismatch: '+str(path))
    return path

def freeze_settings(path,destination):
    import shutil
    c=load_settings(path);destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    meta_path=checked(c['cache_metadata'],c['cache_metadata_sha256']);meta=json.loads(meta_path.read_text())
    if Path(meta['arrays']).name!=meta['arrays']:raise ValueError('cache arrays must be a local basename')
    shutil.copyfile(checked(c['spec_path'],c['spec_sha256']),destination/'robot_spec.json')
    shutil.copyfile(checked(meta_path.parent/meta['arrays'],meta['arrays_sha256']),destination/meta['arrays'])
    shutil.copyfile(meta_path,destination/'cache_metadata.json')
    c['spec_path']='robot_spec.json';c['cache_metadata']='cache_metadata.json'
    target=destination/'geometry_config.json';target.write_text(json.dumps(c,indent=2)+'\n');return target
