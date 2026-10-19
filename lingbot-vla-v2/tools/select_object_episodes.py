#!/usr/bin/env python3
"""目标物episode只读审计；输出审核队列，不修改数据或训练配置，不导入torch。"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from lingbotvla.auto_learning.sampling.target_change_window import (
    WindowSettings, joint_mapping, scan_episode,
)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def target_terms(target, keywords=(), object_ids=()):
    defaults = {'mouse': (['mouse', 'mice', '鼠标'], ['047_mouse'])}
    words, ids = defaults.get(target.casefold(), ([target], []))
    return sorted(set(words + list(keywords))), sorted(set(ids + list(object_ids)))


def matches(text, words):
    return any(re.search(r'(?<!\w)' + re.escape(w) + r'(?!\w)', text, re.I)
               if w.isascii() else w.casefold() in text.casefold() for w in words)


def literal_strings(path):
    """只读AST常量，不import仿真环境；源码出现物体仅代表可能存在。"""
    return [n.value for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('metadata path escapes dataset root')
    return path


def video_segments(root, info, row):
    result = []
    for key, feature in info.get('features', {}).items():
        if feature.get('dtype') != 'video':
            continue
        prefix = 'videos/' + key + '/'
        needed = ['chunk_index', 'file_index', 'from_timestamp', 'to_timestamp']
        if any(row.get(prefix + x) is None for x in needed):
            continue  # 没有v3坐标时不猜偏移。
        start, stop = (float(row[prefix + x]) for x in needed[2:])
        if not (math.isfinite(start) and math.isfinite(stop) and 0 <= start < stop):
            raise ValueError('invalid video segment timestamps')
        path = inside(root, info['video_path'].format(
            video_key=key, chunk_index=int(row[prefix + 'chunk_index']),
            file_index=int(row[prefix + 'file_index'])))
        result.append(dict(camera=key, path=str(path), exists=path.is_file(),
                           from_timestamp=start, to_timestamp=stop))
    return result


def validate_frames(rows, meta, fps, task_labels):
    n = int(meta['length'])
    if len(rows) != n or [r['frame_index'] for r in rows] != list(range(n)):
        raise ValueError('length/frame_index mismatch')
    lo, hi = int(meta['dataset_from_index']), int(meta['dataset_to_index'])
    if hi - lo != n or [r['index'] for r in rows] != list(range(lo, hi)):
        raise ValueError('absolute index mismatch')
    for i, r in enumerate(rows):
        if r['episode_index'] != meta['episode_index']:
            raise ValueError('cross-episode row')
        if not math.isfinite(r['timestamp']) or abs(r['timestamp'] - i / fps) > 0.01 / fps:
            raise ValueError('timestamp/FPS mismatch')
        if task_labels.get(r['task_index']) not in meta['tasks']:
            raise ValueError('frame task_index/instruction mismatch')
        if len(r['observation.state']) != 14 or any(not math.isfinite(x) for x in r['observation.state']):
            raise ValueError('invalid 14D state')


def apply_reviews(records, review, fingerprint):
    if review.get('dataset_fingerprint') != fingerprint:
        raise ValueError('review dataset fingerprint mismatch')
    by_id = {r['episode_index']: r for r in records}
    seen, approved = set(), []
    for d in review.get('decisions', []):
        ep = d['episode_index']
        if ep in seen or ep not in by_id:
            raise ValueError('duplicate or unknown reviewed episode')
        seen.add(ep)
        r = by_id[ep]
        if d.get('decision') not in ('pending', 'rejected', 'approved'):
            raise ValueError('invalid review decision')
        r['review'] = d
        if d['decision'] != 'approved':
            continue
        if (not r['quality_ok'] or not d.get('reviewer') or not d.get('evidence')
                or d.get('manipulated_target_verified') is not True
                or d.get('demonstration_success_verified') is not True):
            raise ValueError('approval requires valid data, reviewer and manipulation/success evidence')
        windows = d.get('phase_windows', {})
        if set(windows) - {'approach', 'grasp_lift', 'transport_place'}:
            raise ValueError('unknown phase label')
        for intervals in windows.values():
            for pair in intervals:
                if (len(pair) != 2 or any(type(x) is not int for x in pair)
                        or not 0 <= pair[0] <= pair[1] < r['length']):
                    raise ValueError('phase window outside episode')
        # 不把控制事件自动变为人工确认窗口，不持久化dataset local_idx。
        approved.append(dict(episode_index=ep, source_task=r['source_task'],
                             instructions=r['instructions'], length=r['length'],
                             absolute_range=r['absolute_range'], review=d))
    return approved


def storyboard(record, out, fps, ffmpeg):
    video = next((v for v in record['videos'] if v['camera'].endswith('cam_high') and v['exists']), None)
    if not video:
        return
    indices = sorted(set(round(i * max(0, record['length'] - 2) / 7) for i in range(8)))
    expr = '+'.join(f'eq(n,{i})' for i in indices)
    image = out / f"episode_{record['episode_index']}.jpg"
    cmd = [ffmpeg, '-nostdin', '-v', 'error', '-threads', '1', '-ss', str(video['from_timestamp']),
           '-i', video['path'], '-an', '-vf', f"select='{expr}',scale=320:-1,tile=4x2",
           '-frames:v', '1', '-threads', '1', '-n', str(image)]
    subprocess.run(cmd, check=True, timeout=180, capture_output=True)
    record['storyboard'] = dict(path=image.name, frame_indices=indices,
                                video_timestamps=[video['from_timestamp'] + i / fps for i in indices])


def write_html(path, records, summary):
    e = lambda x: html.escape(str(x), quote=True)
    parts = ['<!doctype html><meta charset="utf-8"><title>目标物数据审核</title>',
             '<style>body{font-family:sans-serif;max-width:1300px;margin:auto}img{max-width:100%}video{width:480px}</style>',
             '<h1>目标物数据审核：候选不等于批准</h1>', '<pre>' + e(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>',
             '<p>视频按v3元数据时间偏移定位；控制事件不是成功证明。未命中关键词的可能来源保留在折叠队列。</p>']
    for r in sorted(records, key=lambda r: (not r['keyword_hit'], r['episode_index'])):
        parts += [f"<details {'open' if r['keyword_hit'] else ''}><summary>episode {r['episode_index']} · {e(r['source_task'])} · 关键词={r['keyword_hit']} · 数据完整={r['quality_ok']}</summary>",
                  '<p>' + e(' | '.join(r['instructions'])) + '</p>',
                  '<pre>' + e(json.dumps(dict(control_grasp_candidates=r.get('control_grasp_candidates', []),
                                               review=r.get('review', 'pending'), errors=r['quality_errors']), ensure_ascii=False)) + '</pre>']
        if 'storyboard' in r:
            parts += [f'<img loading="lazy" src="{e(r["storyboard"]["path"])}">',
                      '<p>按行阅读；episode帧号：' + e(r['storyboard']['frame_indices']) + '</p>']
        for v in r['videos']:
            if v['exists']:
                uri = Path(v['path']).as_uri()
                parts += [f'<p>{e(v["camera"])}：{v["from_timestamp"]:.3f}–{v["to_timestamp"]:.3f}s</p>',
                          f'<video controls preload="none" data-start="{v["from_timestamp"]}" data-stop="{v["to_timestamp"]}" src="{e(uri)}#t={v["from_timestamp"]},{v["to_timestamp"]}"></video>']
        parts.append('</details>')
    parts.append('''<script>document.querySelectorAll('video').forEach(v=>{
      v.addEventListener('loadedmetadata',()=>{v.currentTime=Number(v.dataset.start)});
      v.addEventListener('timeupdate',()=>{if(v.currentTime>=Number(v.dataset.stop))v.pause()});
      v.addEventListener('play',()=>{if(v.currentTime<Number(v.dataset.start)||v.currentTime>=Number(v.dataset.stop))v.currentTime=Number(v.dataset.start)});
    });</script>''')
    path.write_text('\n'.join(parts))


def run(args):
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    import yaml
    root = args.dataset_root.resolve()
    out = args.output_dir.resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('output must be independent of source dataset')
    if out.exists():
        raise FileExistsError('use a fresh output directory')
    words, object_ids = target_terms(args.target, args.keyword, args.object_id)
    info = json.loads((root/'meta/info.json').read_text())
    fps = float(info['fps'])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('positive finite FPS required')
    manifest = json.loads(args.manifest.read_text())['tasks']
    owners = {}
    for task, entry in manifest.items():
        ids = entry['train_ids'] + entry['val_ids']
        if len(set(ids)) != len(ids) or len(ids) != entry['n_total']:
            raise ValueError('invalid manifest task split')
        for ep in ids:
            if ep in owners:
                raise ValueError('cross-task episode in manifest')
            owners[ep] = task
    if args.primary_task not in manifest:
        raise ValueError('primary task not in manifest')
    # 不猜episode//50；直接使用已批准manifest的归属，保存其内容指纹。
    possible, source_files = set(args.candidate_task), []
    for p in sorted(args.source_envs.glob('*.py')):
        if any(matches(s, words) or any(i.casefold() in s.casefold() for i in object_ids)
               for s in literal_strings(p)):
            possible.add(p.stem)
            source_files.append(p)
    meta_files = sorted((root/'meta/episodes').rglob('*.parquet'))
    if not meta_files:
        raise ValueError('no episode metadata')
    rows = []
    schema_names = set()
    for p in meta_files:
        names = pq.read_schema(p).names
        schema_names.update(names)
        cols = [c for c in names if not c.startswith(('stats/', 'meta/'))]
        rows.extend(pq.read_table(p, columns=cols).to_pylist())
    eps = [int(r['episode_index']) for r in rows]
    if len(set(eps)) != len(eps) or set(eps) != set(owners) or len(eps) != info['total_episodes']:
        raise ValueError('episode inventory disagrees with manifest/info or has duplicates')
    if sum(int(r['length']) for r in rows) != info['total_frames']:
        raise ValueError('metadata frame total disagrees with info')
    files = sorted((root/'data').rglob('*.parquet'))
    # 元数据/清单/规则内容SHA；大Parquet和视频使用路径、大小、mtime_ns，非逐字节内容SHA。
    content_files = [root/'meta/info.json', root/'meta/tasks.parquet', args.manifest,
                     args.embodiment_config, *meta_files, *source_files]
    identity = dict(target=args.target, keywords=words, object_ids=object_ids, possible_tasks=sorted(possible),
                    rules_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    control_window_settings=asdict(WindowSettings()),
                    content=[dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in content_files],
                    data=[dict(path=str(p.relative_to(root)), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files])
    labels = {}
    for r in pq.read_table(root/'meta/tasks.parquet').to_pylist():
        text = r.get('__index_level_0__', r.get('task'))
        if text is None or r['task_index'] in labels:
            raise ValueError('invalid task instruction table')
        labels[r['task_index']] = text
    selected, primary, inventory = [], [], []
    for row in sorted(rows, key=lambda r: r['episode_index']):
        ep = row['episode_index']; task = owners[ep]
        instructions = row['tasks']
        if not isinstance(instructions, list) or not instructions or not all(isinstance(s, str) for s in instructions):
            raise ValueError('episode tasks must be nonempty instruction list')
        hit = matches(' '.join(instructions), words + object_ids)
        item = dict(episode_index=ep, source_task=task, instructions=instructions,
                    length=int(row['length']), keyword_hit=hit, source_task_possible=task in possible,
                    absolute_range=[row['dataset_from_index'], row['dataset_to_index']],
                    videos=video_segments(root, info, row), quality_ok=False, quality_errors=[])
        inventory.append(dict(episode_index=ep, source_task=task, keyword_hit=hit,
                              source_task_possible=task in possible, primary=task == args.primary_task))
        if task == args.primary_task:
            primary.append(item)
        elif hit or task in possible:
            selected.append(item)
    video_paths = sorted({v['path'] for r in primary + selected for v in r['videos'] if v['exists']})
    identity['videos'] = [dict(path=str(Path(p).relative_to(root)), size=Path(p).stat().st_size,
                               mtime_ns=Path(p).stat().st_mtime_ns) for p in video_paths]
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    # 只载入候选及主任务的CPU状态列；不读action/image，不建立模型或训练dataset。
    table = ds.dataset([str(p) for p in files], format='parquet')
    columns = ['index', 'episode_index', 'frame_index', 'timestamp', 'task_index', 'observation.state']
    grouped = defaultdict(list)
    selected_ids = [r['episode_index'] for r in primary + selected]
    for batch in table.scanner(columns=columns, filter=ds.field('episode_index').isin(selected_ids), batch_size=8192, use_threads=False).to_batches():
        for r in batch.to_pylist():
            grouped[r['episode_index']].append(r)
    mapping = joint_mapping(yaml.safe_load(args.embodiment_config.read_text()), [['fl_joint6'], ['fr_joint6']])
    metadata = {r['episode_index']: r for r in rows}
    for record in primary + selected:
        ep = record['episode_index']
        frames = sorted(grouped.pop(ep, []), key=lambda r: r['frame_index'])
        try:
            validate_frames(frames, metadata[ep], fps, labels)
            result = scan_episode(frames, fps, mapping, WindowSettings())
            record.update(quality_ok=True, control_grasp_candidates=result['grasp_phases'],
                          control_grasp_arms=sorted({p['side'] for p in result['grasp_phases']}),
                          control_release_candidates=[e for e in result['events']
                            if e['signal'].endswith('gripper_command') and e['kind']=='start'
                            and any(s['name']==e['signal'] and s['values'][min(e['frame']+3,len(frames)-1)] > e['value'] for s in result['signals'])])
        except ValueError as exc:
            record['quality_errors'].append(str(exc))
    approved = []
    if args.review:
        approved = apply_reviews(selected, json.loads(args.review.read_text()), fingerprint)
    out.mkdir(parents=True)
    print(f'元数据与候选数据校验完成：primary={len(primary)} candidates={len(selected)}', flush=True)
    if args.extract_keyframes:
        for r in selected:
            if r['keyword_hit'] and r['quality_ok']:
                print(f'关键帧审核图：episode={r["episode_index"]}', flush=True)
                try:
                    storyboard(r, out, fps, args.ffmpeg)
                except (subprocess.SubprocessError, FileNotFoundError) as exc:
                    r['storyboard_error'] = str(exc)
    summary = dict(target=args.target, total_episodes=len(rows), total_frames=info['total_frames'],
                   primary_episodes=len(primary), primary_frames=sum(r['length'] for r in primary),
                   candidate_episodes=len(selected), candidate_frames=sum(r['length'] for r in selected),
                   keyword_auxiliary_episodes=sum(r['keyword_hit'] for r in selected),
                   keyword_by_task=dict(Counter(r['source_task'] for r in selected if r['keyword_hit'])),
                   possible_source_tasks=sorted(possible & set(manifest)),
                   quality_errors=sum(not r['quality_ok'] for r in selected + primary),
                   approved_episodes=len(approved), dataset_fingerprint=fingerprint,
                   missing_object_identity=True, missing_success_label=True,
                   metadata_fields=sorted(schema_names - {c for c in schema_names if c.startswith('stats/')}))
    dump(out/'source_identity.json', identity)
    dump(out/'summary.json', summary)
    dump(out/'primary_episodes.json', primary)
    for name, records in [('candidates.jsonl', selected), ('inventory.jsonl', inventory)]:
        (out/name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    dump(out/'approved.json', dict(schema=1, target=args.target, dataset_fingerprint=fingerprint,
                                  episodes=approved, training_ready=False,
                                  note='本轮仅筛选；训练接入尚未实现。无窗口表示完整轨迹，不自动产生阶段权重。'))
    dump(out/'review_template.json', dict(dataset_fingerprint=fingerprint, decisions=[dict(
        episode_index=r['episode_index'], decision='pending', reviewer='', evidence='',
        manipulated_target_verified=False, demonstration_success_verified=False, phase_windows={}) for r in selected]))
    write_html(out/'review.html', selected, summary)
    (out/'report.md').write_text('# 目标物数据审计\n\n```json\n' + json.dumps(summary, ensure_ascii=False, indent=2) + '\n```\n\n关键词与源码召回均不是对象操作或成功证据；未召回不等于确认无目标物。审核通过后仍需另行接入混合训练。\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--primary-task', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--keyword', action='append', default=[])
    p.add_argument('--object-id', action='append', default=[])
    p.add_argument('--candidate-task', action='append', default=[])
    p.add_argument('--manifest', type=Path, default=REPO/'refs/al_artifacts/al/task_splits_50/manifest.json')
    p.add_argument('--source-envs', type=Path, default=REPO.parent/'RoboTwin-lingbot/envs')
    p.add_argument('--embodiment-config', type=Path, required=True)
    p.add_argument('--review', type=Path)
    p.add_argument('--extract-keyframes', action='store_true')
    p.add_argument('--ffmpeg', default='ffmpeg')
    return run(p.parse_args(argv))


if __name__ == '__main__':
    main()
