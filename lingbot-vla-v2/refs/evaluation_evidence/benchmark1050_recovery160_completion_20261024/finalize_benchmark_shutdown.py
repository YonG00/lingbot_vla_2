"""Verify all requested evidence, archive, then honor explicit shutdown authorization."""
import hashlib,json,os,subprocess,tarfile,time,traceback
from pathlib import Path
CHAIN=Path('/data/eval_results/20261024_2258_1050_repair_then_recovery160_pipeline')
SOURCE=Path('/data/eval_results/20261024_1923_1050_benchmark_clean50_random50_parallel4_inline_novideo')
RECOVERY=Path('/data/eval_results/20261024_2258_recovery3_s160_benchmark_clean50_random50_parallel4_matched1050_novideo')
STATUS=CHAIN/'completion_shutdown_status.json'
def write(value):
 p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2));p.replace(STATUS)
def verify(root):
 public=json.loads((root/'benchmark_results.json').read_text())
 audit=json.loads((root/'benchmark_audit.json').read_text())
 if public['status']!='complete' or public['overall']['completed_task_count']!=100 or public['overall']['episode_count']!=400 or public['overall']['error_task_count']!=0 or public['overall']['pending_task_count']!=0:
  raise ValueError('incomplete report: '+str(root))
 rows=public['tasks'];by_key={r['task_config']+'/'+r['task']:r for r in rows}
 if len(rows)!=100 or len(by_key)!=100 or len(audit['items'])!=100:raise ValueError('invalid task coverage')
 if any(sum(r['task_config']==c for r in rows)!=50 for c in ['demo_clean','demo_randomized']):raise ValueError('invalid clean/random coverage')
 seeds={};files=[root/n for n in ['benchmark_results.json','benchmark_audit.json','manifest.json','selected_scene_seeds.json']]
 total=0;clients=set()
 for row in audit['items']:
  key=row['key'];pub=by_key[key]
  if row['state']!='complete' or len(row['seeds'])!=4 or len(set(row['seeds']))!=4:raise ValueError('invalid task completion '+key)
  summary_path=Path(row['summary_path']);summary=json.loads(summary_path.read_text());files.append(summary_path)
  episodes=summary['episode_results']
  if summary['episodes']!=4 or len(episodes)!=4 or summary['seeds']!=pub['seeds'] or row['seeds']!=pub['seeds']:raise ValueError('task summary mismatch '+key)
  n=0
  for episode in episodes:
   if episode['state']!='complete' or episode['episodes']!=1 or episode['successes'] not in (0,1) or episode['oidn_errors']!=0 or episode['recorded_episodes']!=0 or episode['dtype']!='float32' or episode['action_chunk']!=50:raise ValueError('invalid episode '+key)
   clients.add(episode['client_sha256']);n+=episode['successes']
   epdir=summary_path.parent/'episodes'/('episode_%03d'%episode['episode_index'])
   log=epdir/'sim.log';text=log.read_text(errors='replace')
   if 'OIDN Error' in text or 'OIDN 2.5.1 CUDA=' not in text or 'Traceback (most recent call last)' in text:raise ValueError('renderer/runtime evidence failed '+str(log))
   for name in ['summary.json','precision_audit.json','commands.json','sim.log']:
    files.append(epdir/name)
  if n!=pub['successes'] or n!=row['successes']:raise ValueError('success count mismatch '+key)
  total+=n;seeds[key]=row['seeds']
 if total!=public['overall']['successes'] or abs(public['overall']['success_rate_percent']-100*total/400)>1e-8:raise ValueError('aggregate arithmetic mismatch')
 return public,seeds,files,clients
try:
 write(dict(state='waiting_for_evaluations',shutdown_authorized=True))
 while True:
  state=json.loads((CHAIN/'status.json').read_text())
  if state['state']=='failed':raise RuntimeError('evaluation chain failed: '+state.get('error',''))
  if state['state']=='complete':break
  time.sleep(15)
 source,seeds,source_files,client1=verify(SOURCE)
 recovery,seeds2,recovery_files,client2=verify(RECOVERY)
 if seeds!=seeds2 or client1!=client2 or len(client1)!=1:raise ValueError('paired task/seed/client mismatch')
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():raise RuntimeError('GPU still occupied; shutdown blocked')
 comparison=dict(state='verified_complete',source=source['overall'],recovery=recovery['overall'],identical_scene_seeds=True,episodes_per_model=400,episode_workers=4,recording=False,
  source_report=str(SOURCE/'benchmark_results.json'),recovery_report=str(RECOVERY/'benchmark_results.json'),
  source_task_configs=source['task_configs'],recovery_task_configs=recovery['task_configs'])
 (CHAIN/'verified_comparison.json').write_text(json.dumps(comparison,indent=2))
 archive=CHAIN/'verified_benchmark_evidence.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  tar.add(CHAIN/'verified_comparison.json',arcname='comparison.json')
  for label,root,files in [('1050',SOURCE,source_files),('recovery160',RECOVERY,recovery_files)]:
   for path in dict.fromkeys(files):tar.add(path,arcname=label+'/'+str(path.relative_to(root)))
 h=hashlib.sha256(archive.read_bytes()).hexdigest();os.sync()
 write(dict(state='results_verified_archive_ready',archive=str(archive),sha256=h,size=archive.stat().st_size,shutdown_authorized=True))
 # Allow a connected local archiver to retrieve evidence before shutdown.
 for _ in range(8):
  if (CHAIN/'local_archive_verified.json').exists():break
  time.sleep(15)
 os.sync()
 write(dict(state='shutdown_requested',archive=str(archive),sha256=h,shutdown_authorized=True,local_archive_receipt=(CHAIN/'local_archive_verified.json').exists()))
 subprocess.run(['/usr/bin/shutdown','-h','now'],check=True)
except BaseException as e:
 write(dict(state='blocked_not_shutdown',error=repr(e),shutdown_authorized=True));traceback.print_exc();raise
