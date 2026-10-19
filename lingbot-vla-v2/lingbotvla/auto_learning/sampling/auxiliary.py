"""独立辅助池：原始来源/指令，episode均衡，坐标重映射；不进入PASS/Replay。"""
from __future__ import annotations

from collections import Counter
import ast
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random

from ..catalog import TaskCatalog
from ..ports import TaskEntry
from ..types import BatchComposition
from .rng import WeightedSampler
from .sampler import BatchSampler


def full_catalog_for_auxiliary(manifest_path):
    """直接读取权威常量，保留block严格校验，不为CPU预览import pandas。"""
    source = Path(__file__).resolve().parents[3]/'tools/robotwin_curriculum.py'
    constants = {}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in ('TASK_ORDER','EPISODES_PER_TASK'):
                    constants[name.id] = ast.literal_eval(node.value)
    if set(constants) != {'TASK_ORDER','EPISODES_PER_TASK'}:
        raise ValueError('missing authoritative curriculum constants')
    return TaskCatalog.from_manifest(str(manifest_path), task_order=constants['TASK_ORDER'],
                                     episodes_per_task=constants['EPISODES_PER_TASK'], strict=True)


def primary_quota(local_batch, dp_size, rank, ratio, step):
    """按全局累积配额取整；非整数比例跨步补齐，rank合计永远等于全局配额。"""
    if min(local_batch, dp_size) < 1 or not 0 <= rank < dp_size or step < 0:
        raise ValueError('invalid auxiliary batch topology/step')
    q = Fraction(str(ratio)); total = local_batch * dp_size
    if not 0 < q < 1:
        raise ValueError('auxiliary primary ratio must be between 0 and 1')
    count = ((step+1)*total*q).__floor__() - (step*total*q).__floor__()
    return (rank+1)*count//dp_size - rank*count//dp_size


def verify_source(identity, expected, root, manifest_path):
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if digest != expected:
        raise ValueError('auxiliary source identity fingerprint mismatch')
    origins = [r['path'][:-len('/meta/info.json')] for r in identity['content']
               if r['path'].endswith('/meta/info.json')]
    if len(origins) != 1:
        raise ValueError('auxiliary identity lacks unique dataset root')
    origin = Path(origins[0]); root = Path(root).resolve()
    def local(relative):
        p = (root / relative).resolve()
        if not p.is_relative_to(root):
            raise ValueError('auxiliary source path escapes dataset')
        return p
    for row in identity['content']:
        old = Path(row['path'])
        if old.is_relative_to(origin):
            path = local(old.relative_to(origin))
        elif old.name == 'manifest.json':
            path = Path(manifest_path)
        else:
            continue  # 审核时源码/机器人配置记录作归档，不把旧机器绝对路径当输入。
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'auxiliary source content changed: {path}')
    for row in identity['data'] + identity.get('videos', []):
        stamp = local(row['path']).stat()
        if stamp.st_size != row['size'] or stamp.st_mtime_ns != row['mtime_ns']:
            raise ValueError(f'auxiliary data/video identity changed: {row["path"]}')


class AuxiliaryPool:
    def __init__(self, cfg, full_catalog, primary_catalog, *, dataset_root, manifest_path):
        path = Path(cfg.auxiliary_manifest).resolve()
        raw = json.loads(path.read_text())
        if raw.get('training_ready') is not True:
            raise ValueError('auxiliary manifest has not been promoted for training')
        identity_path = path.with_name('source_identity.json')
        self.identity = json.loads(identity_path.read_text())
        verify_source(self.identity, raw['dataset_fingerprint'], dataset_root, manifest_path)
        self.fingerprint = raw['dataset_fingerprint']
        self.manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.primary_names = primary_catalog.task_names()
        if len(self.primary_names) != 1:
            raise ValueError('auxiliary pool requires one primary task')
        primary_ids = set(primary_catalog.entry(self.primary_names[0]).train_traj_ids)
        self.records = {}; entries = dict(primary_catalog._tasks)
        by_task = {}
        for row in raw.get('episodes', []):
            ep, task = row['episode_index'], row['source_task']
            review = row.get('review', {})
            if (type(ep) is not int or ep in self.records or ep in primary_ids
                    or task in self.primary_names or task not in full_catalog
                    or full_catalog.task_of_episode(ep) != task
                    or ep not in full_catalog.entry(task).train_traj_ids + full_catalog.entry(task).val_traj_ids):
                raise ValueError('auxiliary episode/source overlaps or disagrees with task manifest')
            if (review.get('decision') != 'approved' or review.get('manipulated_target_verified') is not True
                    or review.get('demonstration_success_verified') is not True
                    or not review.get('evidence') or not review.get('reviewer')):
                raise ValueError('auxiliary requires explicitly approved manipulation evidence')
            if type(row.get('length')) is not int or row['length'] < 1:
                raise ValueError('invalid auxiliary episode length')
            lo, hi = row['absolute_range']
            if hi-lo != row['length']:
                raise ValueError('auxiliary absolute range mismatch')
            self.records[ep] = row; by_task.setdefault(task, []).append(ep)
        if not self.records:
            raise ValueError('auxiliary approved episode pool is empty')
        for task, eps in by_task.items():
            entries[task] = TaskEntry(name=task, train_traj_ids=sorted(eps), n_total=len(eps),
                                     strategy='approved_auxiliary_full_episode')
        self.catalog = TaskCatalog(entries, meta={**primary_catalog.meta, 'train_all_episodes': True})
        if self.catalog.verify():
            raise ValueError('invalid auxiliary runtime catalog')
        self.fps = float(json.loads((Path(dataset_root)/'meta/info.json').read_text())['fps'])
        self.grasp_weight = cfg.auxiliary_grasp_weight
        self.context = cfg.auxiliary_approach_seconds
        self.tables = {}; self.grasp_ids = set(); self.allowed = {}

    def bind(self, resolver):
        self.resolver = resolver
        self.tables = {}; self.grasp_ids = set(); self.allowed = {}
        for ep, row in self.records.items():
            ids = resolver.local_indices_of_episode(ep)
            lo, hi = row['absolute_range']
            if len(ids) != row['length'] or [resolver.resolve(row['source_task'], i).frame_id for i in ids] != list(range(lo, hi)):
                raise ValueError(f'auxiliary resolver coordinate mismatch: episode={ep}')
            reviewed = row['review']
            windows = []
            # 人工精确标注优先；当前鼠标清单仅有目标臂绑定的控制候选，不声称真实接触。
            for kind in ('approach', 'grasp_lift'):
                windows.extend(reviewed.get('phase_windows', {}).get(kind, []))
            for phase in reviewed.get('control_grasp_candidates', []):
                if phase['side'] != reviewed.get('mouse_arm'):
                    raise ValueError('auxiliary grasp candidate belongs to non-target arm')
                start, stop = phase['window']
                start = min(start, max(0, phase['close_start'] - math.ceil(self.context*self.fps)))
                windows.append([start, stop])
            marked = set()
            for pair in windows:
                if (len(pair)!=2 or any(type(v) is not int for v in pair)
                        or not 0 <= pair[0] <= pair[1] < len(ids)):
                    raise ValueError('auxiliary phase window outside episode')
                marked.update(ids[pair[0]:pair[1]+1])
            self.grasp_ids.update(marked)
            self.tables[ep] = WeightedSampler(ids, [self.grasp_weight if i in marked else 1.0 for i in ids])
            self.allowed.update({i:(row['source_task'], ep) for i in ids})
        return self

    def sample(self, n, rng):
        # 先均匀选episode，再在episode内加权；长轨迹不会因此占更多配额。
        episodes = sorted(self.tables)
        counts = Counter(rng.choices(episodes, k=n)); refs=[]
        for ep, count in counts.items():
            table = self.tables[ep]
            if count > len(table.keys):
                raise ValueError('auxiliary batch repeats an episode beyond available anchors')
            ids = table.draw_without_replacement(count, rng)
            refs.extend(self.resolver.resolve(self.records[ep]['source_task'], i) for i in ids)
        rng.shuffle(refs)
        return refs

    def summary(self):
        return dict(episodes=len(self.records), frames=sum(r['length'] for r in self.records.values()),
                    by_task=dict(Counter(r['source_task'] for r in self.records.values())),
                    dataset_fingerprint=self.fingerprint, manifest_sha256=self.manifest_sha256,
                    grasp_weight=self.grasp_weight, approach_seconds=self.context,
                    weighted_anchor_frames=len(self.grasp_ids),
                    phase_kind='reviewed target arm + control candidate, not physical success detector')


class AuxiliaryBatchSampler(BatchSampler):
    def __init__(self, cfg, resolver, catalog, rng, pool):
        super().__init__(cfg, resolver, catalog, rng); self.pool = pool

    def prepare(self, req):
        if req.replay_slots or req.replay.tasks:
            raise ValueError('auxiliary slots cannot be combined with Replay')
        return super().prepare(req)

    def build(self, prepared, *, logical_step=None):
        if logical_step is None:
            raise ValueError('auxiliary sampling requires logical_step')
        size = prepared.request.batch_size
        rank = int(getattr(self.cfg, '_ratio_dp_rank', 0))
        dp = int(getattr(self.cfg, '_ratio_dp_size', 1))
        primary = primary_quota(size, dp, rank, self.cfg.auxiliary_primary_ratio, logical_step)
        local = replace(prepared, request=replace(prepared.request, batch_size=primary, new_slots=primary))
        base = super().build(local, logical_step=logical_step)
        # 独立确定性RNG；不改变scheduler的状态，也不因预取深度改变有效batch。
        seed = f'auxiliary:{self.cfg.seed}:{rank}:{logical_step}'
        rng = random.Random(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], 'big'))
        aux = self.pool.sample(size-primary, rng)
        for ref in aux:
            if self.pool.allowed.get(ref.sample_id) != (ref.task, ref.episode_id):
                raise AssertionError('auxiliary provenance violation')
        comp = BatchComposition(new=base.new, old=base.old, auxiliary=aux)
        self._assert_train_only(comp)
        if len(comp.refs)!=size:
            raise AssertionError('auxiliary batch size mismatch')
        return comp


def preflight_pool(pool, dataset_root):
    """真实CPU白名单对拍；预览不写run/cache，不加载模型或图像。"""
    import numpy as np
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    from ..resolver import SampleResolver
    from tools.select_object_episodes import validate_frames
    root = Path(dataset_root)
    episodes = sorted(ep for name in pool.catalog.task_names()
                      for ep in pool.catalog.entry(name).train_traj_ids)
    columns = ['index', 'episode_index', 'frame_index', 'timestamp', 'task_index', 'observation.state']
    table = ds.dataset([str(p) for p in sorted((root/'data').rglob('*.parquet'))], format='parquet')
    rows = table.to_table(columns=columns, filter=ds.field('episode_index').isin(episodes), use_threads=False).to_pylist()
    rows.sort(key=lambda r:r['index'])
    metadata = {}
    for p in sorted((root/'meta/episodes').rglob('*.parquet')):
        for r in pq.read_table(p, columns=['episode_index','length','tasks','dataset_from_index','dataset_to_index']).to_pylist():
            if r['episode_index'] in episodes:
                if r['episode_index'] in metadata:
                    raise ValueError('duplicate auxiliary preflight metadata episode')
                metadata[r['episode_index']] = r
    labels = {r['task_index']:r.get('__index_level_0__',r.get('task'))
              for r in pq.read_table(root/'meta/tasks.parquet').to_pylist()}
    grouped = {}
    for r in rows:
        grouped.setdefault(r['episode_index'],[]).append(r)
    if set(grouped) != set(episodes) or set(metadata) != set(episodes):
        raise ValueError('auxiliary joint whitelist differs from actual dataset')
    for ep in episodes:
        validate_frames(grouped[ep], metadata[ep], pool.fps, labels)
        if ep in pool.records and metadata[ep]['tasks'] != pool.records[ep]['instructions']:
            raise ValueError('auxiliary approved instruction differs from actual metadata')
    resolver = SampleResolver(episode_index=np.array([r['episode_index'] for r in rows]),
                              frame_index=np.array([r['index'] for r in rows]),
                              task_of_episode=pool.catalog.task_of_episode)
    if resolver.verify():
        raise ValueError('invalid auxiliary joint resolver mapping')
    pool.bind(resolver)
    summary = pool.summary()
    summary.update(joint_episodes=len(episodes), joint_frames=len(rows),
                   mapping_sha256=hashlib.sha256(resolver.episode_map().tobytes()+resolver.frame_map().tobytes()).hexdigest())
    return summary
