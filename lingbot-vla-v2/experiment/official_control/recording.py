"""Per-process global-view recording policy, independent of rollout length."""


def video_count(episodes, limit=None):
    if type(episodes) is not int or episodes<=0:raise ValueError('positive episodes required')
    if limit is not None and (type(limit) is not int or limit<0):raise ValueError('video_episodes must be nonnegative integer or null')
    return episodes if limit is None else min(episodes,limit)


def install_recording_limit(task_class, limit):
    if type(limit) is not int or limit<0:raise ValueError('invalid recording limit')
    original=task_class._start_eval_global_video
    def start(task,*args,**kwargs):
        if task.test_num<limit:return original(task,*args,**kwargs)
    task_class._start_eval_global_video=start
