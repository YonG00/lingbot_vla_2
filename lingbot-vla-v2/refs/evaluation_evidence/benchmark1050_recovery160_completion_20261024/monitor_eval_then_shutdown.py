import json,subprocess,time
from pathlib import Path
CHAIN=Path('/data/eval_results/20261024_2258_1050_repair_then_recovery160_pipeline')
PID=87701
STATUS=CHAIN/'auto_shutdown_status.json'
def identity(pid):
 try:
  fields=Path('/proc/'+str(pid)+'/stat').read_text().split(') ',1)[1].split()
  return None if fields[0]=='Z' else fields[19]
 except FileNotFoundError:return None
def write(state):STATUS.write_text(json.dumps(dict(state=state,target_pid=PID,mode='process_exit_no_result_validation',command='/usr/bin/shutdown'),indent=2))
initial=identity(PID)
write('monitoring_evaluation_process')
while initial is not None and identity(PID)==initial:
 time.sleep(5)
write('shutdown_requested')
subprocess.run(['/usr/bin/shutdown'],check=True)
