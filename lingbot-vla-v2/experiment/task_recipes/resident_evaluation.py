"""Final-checkpoint evaluation in the training process; no checkpoint weight reads."""
import asyncio
import contextlib
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace


def unwrap(model):
    seen=set()
    while hasattr(model,'_orig_mod'):
        if id(model) in seen:raise ValueError('compiled wrapper cycle')
        seen.add(id(model));model=model._orig_mod
    return model


def validate_handoff(c,step):
    if not c['evaluation'].get('reuse_training_model'):raise ValueError('resident handoff not enabled')
    if step!=c['train']['steps']:raise ValueError('only the final model can be handed off')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('resident handoff supports single GPU only')


def prepare_view(model,inference_class):
    import copy
    model=unwrap(model)
    if any(hasattr(p,'placements') for p in model.parameters()):
        raise ValueError('sharded parameters are not supported by resident evaluation')
    model.gradient_checkpointing_disable()
    view=copy.copy(model);view.__class__=inference_class
    if view.model is not model.model:raise RuntimeError('resident model identity changed')
    for module in view.modules():
        config=getattr(module,'config',None)
        if config is not None:
            if hasattr(config,'attention_implementation'):config.attention_implementation='eager'
            if hasattr(config,'precompute_grid_thw'):config.precompute_grid_thw=False
            if hasattr(config,'use_cache'):config.use_cache=True
        if hasattr(module,'get_attention_interface'):
            module.attention_interface=module.get_attention_interface()
        for key in ['pos_embeds','position_embeddings','cu_seqlens','visual_split_sizes','visual_max_seqlen']:
            if hasattr(module,key):setattr(module,key,None)
    view.model._use_compile_predict_velocity=False
    view.model._compiled_predict_velocity=None
    view.float().eval()
    return view


def make_policy(model,cfg):
    """Attach deployment preprocessing to the existing policy, preserving its modules."""
    import torch
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server,LingBotVlaV2InferencePolicy
    from lingbotvla.models import build_processor
    view=prepare_view(model,LingBotVlaV2InferencePolicy)
    policy=LingbotVLAv2Server.__new__(LingbotVLAv2Server)
    policy.vla=view;policy.config=view.config
    policy.use_bf16=False;policy.use_fp32=True;policy.use_compile=False
    policy.adaptive_ensemble_alpha=0.1;policy.action_ensemble_horizon=8
    policy.use_length=50;policy.chunk_ret=True;policy.task_description=None
    policy.global_step=0;policy.last_action_chunk=None;policy.last_normalized_action_chunk=None
    policy.action_key='action';policy.model_name='qwen3vl'
    policy.data_config=deployment_data_config(cfg['data'])
    policy.robot_norm_path=policy.data_config.norm_stats_file
    policy.processor=build_processor(cfg['model']['tokenizer_path'])
    policy.language_tokenizer=policy.processor.tokenizer
    policy.sample_actions_fn=view.model.sample_actions
    torch.cuda.empty_cache()
    return policy


def deployment_data_config(data):
    # Training CLI normalizes structured JSON fields before FeatureTransform;
    # the handoff bypasses that parser, so retain its literal-string contract.
    data=dict(data)
    for name in ['joints','norm_type']:
        data[name]=[repr(value) if isinstance(value,dict) else value for value in data[name]]
    return SimpleNamespace(**data)


def validate_preprocessing(cfg):
    """Exercise the real deployment reset contract on CPU before allocating a model."""
    from lingbotvla.data.vla_data.utils import FeatureTransform
    root=Path(__file__).resolve().parents[2]
    return FeatureTransform(str(root/'configs/robot_configs/robotwin.yaml'),deployment_data_config(cfg['data']),
        SimpleNamespace(**cfg['model']),None,norm_stats_path=cfg['data']['norm_stats_file'])


def jobs_for(c,out,step,cfg):
    root=Path(__file__).resolve().parents[2];directory=out/f'evaluation_{step}'
    jobs=[]
    for task in c['evaluation']['tasks']:
        j=dict(task=task,label=f'recipe_{step}',model=str(out/f'checkpoints/global_step_{step}/hf_ckpt'),
            destination=str(directory/task),episodes=c['evaluation']['episodes_per_task'],require_baseline=False,
            robotwin=str(root.parent/'RoboTwin-lingbot'),qwen=cfg['model']['tokenizer_path'],
            sim_python=os.environ.get('LINGBOT_SIM_PYTHON','/data/miniconda3/envs/RoboTwin/bin/python'))
        for key in ['video_episodes','render_sync']:
            if key in c['evaluation']:j[key]=c['evaluation'][key]
        jobs.append(j)
    return jobs


def read_completed(directory,c,step):
    status=json.loads((directory/'status.json').read_text())
    handoff=json.loads((directory/'handoff.json').read_text())
    if (status['state']!='complete' or not status.get('reused_training_model') or status['model_loads']!=0
        or handoff['training_pid']!=handoff['server_pid'] or handoff['weight_loads']!=0):
        raise ValueError('resident evaluation incomplete; refusing reload fallback')
    results=json.loads((directory/'summary.json').read_text())
    if {r['task'] for r in results}!=set(c['evaluation']['tasks']):raise ValueError('resident task mismatch')
    for r in results:
        if (r['server_pid']!=handoff['training_pid'] or r['episodes']!=c['evaluation']['episodes_per_task']
            or Path(r['model']).resolve()!= (directory.parent/f'checkpoints/global_step_{step}/hf_ckpt').resolve()):
            raise ValueError('resident result identity mismatch')
    return results


async def serve_and_evaluate(server,work):
    service=asyncio.create_task(server.run())
    worker=asyncio.create_task(asyncio.to_thread(work))
    try:
        done,_=await asyncio.wait([service,worker],return_when=asyncio.FIRST_COMPLETED)
        if service in done:
            service.result()
            raise RuntimeError('resident inference service exited before evaluation')
        return worker.result()
    finally:
        service.cancel()
        with contextlib.suppress(asyncio.CancelledError):await service
        # On server failure the bounded simulator jobs must finish/fail before returning.
        if not worker.done():
            with contextlib.suppress(Exception):await worker


def evaluate_resident(model,step,out):
    from experiment.task_recipes.runtime import load_run
    from experiment.official_control.baseline_pipeline import verify_checkpoint
    out=Path(out);c,_,_=load_run(out);validate_handoff(c,step)
    verify_checkpoint(out/f'checkpoints/global_step_{step}/hf_ckpt')
    launch=out.with_name(out.name+'_pipeline')
    if launch.is_dir():
        (launch/'status.json').write_text(json.dumps(dict(state='evaluation',step=step,
            reused_training_model=True,server_pid=os.getpid(),completed=[]),indent=2))
    cfg=json.loads((out/'trainer_config.json').read_text())
    directory=out/f'evaluation_{step}';directory.mkdir(exist_ok=False)
    policy=make_policy(model,cfg)
    jobs=jobs_for(c,out,step,cfg)
    serve_policy(policy,jobs,directory,c['evaluation']['max_parallel_tasks'],c['evaluation'].get('scheduling','batches'))


def serve_policy(policy,jobs,directory,workers,scheduling='batches'):
    from experiment.official_control.parallel_evaluation import run_jobs
    from experiment.robotwin.session_eval_server import install_session_handler
    from experiment.robotwin.precision_eval_server import instrument
    from deploy.websocket_policy_server import WebsocketPolicyServer
    os.environ['LINGBOT_INFER_NOISE_SEED']='0'
    os.environ['LINGBOT_SESSION_AUDIT']=str(directory/'session_audit.json')
    audit=directory/'precision_audit.json'
    instrument(policy,str(audit));install_session_handler()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=WebsocketPolicyServer(policy,host='127.0.0.1',port=port)
    server.model_loads=0
    process=SimpleNamespace(pid=os.getpid(),poll=lambda:None)
    session=dict(model=jobs[0]['model'],process=process,port=port,audit=audit,
        model_loads=0,reused_training_model=True)
    (directory/'handoff.json').write_text(json.dumps(dict(training_pid=os.getpid(),server_pid=os.getpid(),
        weight_loads=0,model_core_identity_preserved=True,precision='float32'),indent=2))
    def work():
        return run_jobs(jobs,directory,workers,opener=lambda *_:session,
            stopper=lambda _:None,scheduling=scheduling)
    asyncio.run(serve_and_evaluate(server,work))
