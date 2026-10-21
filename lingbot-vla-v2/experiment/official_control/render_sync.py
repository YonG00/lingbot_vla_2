"""Process-local profiling and deferred synchronization during policy actions only."""
import atexit
import json
from pathlib import Path
import time


def install_render_sync(task_class, mode, audit_path):
    if mode not in ['eager','on_demand']:raise ValueError('invalid render_sync')
    original_update=task_class._update_render
    original_action=task_class.take_action
    original_obs=task_class.get_obs
    original_init=task_class._init_task_env_
    original_video=task_class._start_eval_global_video
    stats=dict(mode=mode,skipped_sync_calls=0,actual_sync_calls=0,sync_seconds=0.,
        action_calls=0,action_seconds=0.,observation_calls=0,observation_seconds=0.)
    def init(task,*args,**kwargs):
        task._render_policy_active=False
        task._render_action_depth=task._render_observation_depth=0
        return original_init(task,*args,**kwargs)
    def start(task,*args,**kwargs):
        # The client invokes this at policy rollout start even when its recording limit is reached.
        task._render_policy_active=True
        return original_video(task,*args,**kwargs)
    def update(task,*args,**kwargs):
        safe=(mode=='on_demand' and getattr(task,'_render_policy_active',False)
            and getattr(task,'_render_action_depth',0)>0 and not getattr(task,'_render_observation_depth',0)
            and getattr(task,'eval_mode',False) and not getattr(task,'render_freq',1)
            and not getattr(task,'crazy_random_light',True) and getattr(task,'eval_video_path',None) is None)
        if safe:
            stats['skipped_sync_calls']+=1
            return None
        before=time.perf_counter()
        try:return original_update(task,*args,**kwargs)
        finally:
            stats['actual_sync_calls']+=1;stats['sync_seconds']+=time.perf_counter()-before
    def action(task,*args,**kwargs):
        before=time.perf_counter();task._render_action_depth=getattr(task,'_render_action_depth',0)+1
        try:return original_action(task,*args,**kwargs)
        finally:
            task._render_action_depth-=1;stats['action_calls']+=1;stats['action_seconds']+=time.perf_counter()-before
    def obs(task,*args,**kwargs):
        before=time.perf_counter();task._render_observation_depth=getattr(task,'_render_observation_depth',0)+1
        try:return original_obs(task,*args,**kwargs)
        finally:
            task._render_observation_depth-=1;stats['observation_calls']+=1;stats['observation_seconds']+=time.perf_counter()-before
    task_class._init_task_env_=init
    task_class._start_eval_global_video=start
    task_class._update_render=update
    task_class.take_action=action
    task_class.get_obs=obs
    def flush():Path(audit_path).write_text(json.dumps(stats,indent=2))
    atexit.register(flush)
    return stats,flush


def install_policy_trace(client_class, path):
    import hashlib
    original=client_class.infer
    def infer(client,observation):
        request=hashlib.sha256(client._packer.pack(observation)).hexdigest()
        result=original(client,observation)
        action=hashlib.sha256(client._packer.pack(result.get('action'))).hexdigest()
        with Path(path).open('a') as stream:
            stream.write(json.dumps(dict(reset=bool(observation.get('reset')),request_sha256=request,action_sha256=action))+'\n')
        return result
    client_class.infer=infer
