#!/usr/bin/env bash
# HF 直出独立验收：复用 launcher 打印过的完整 torchrun 参数，只替换入口脚本与 4 个开关
set -u
cd /data/code/lingbot-vla-v2
PY=/data/miniconda3/envs/lingbotvla/bin/python
OUT=/data/outputs/al_hf_direct_acceptance_$(date +%m%d_%H%M)
echo "$OUT" > /data/tmp/hf_accept_outdir.txt
"$PY" - "$OUT" <<'PYEOF' > /data/tmp/hf_accept_cmd.sh
import re, sys, pathlib
out = sys.argv[1]
log = pathlib.Path("/data/tmp/dcp_test_job.log").read_text(encoding="utf-8", errors="ignore")
line = ""
for l in log.splitlines():
    if l.strip().startswith("+ torchrun"):
        line = l.strip()[2:]          # 去掉 "+ "
if not line:
    sys.exit("找不到 launcher 打印的 torchrun 命令行")
cmd = line.replace("tasks/vla/train_lingbotvla.py", "tools/hf_direct_export_acceptance.py")
cmd = re.sub(r"--train\.output_dir \S+", f"--train.output_dir {out}", cmd)
cmd = re.sub(r"--train\.max_steps \S+", "--train.max_steps 502", cmd)
cmd = re.sub(r"--train\.save_steps \S+", "--train.save_steps 1000", cmd)
cmd = re.sub(r"--train\.hf_pass_interval \S+", "--train.hf_pass_interval 1", cmd)
cmd = cmd.replace("--master-port 62500", "--master-port 62511")
print("export TORCH_COMPILE_DISABLE=1")
print(cmd)
PYEOF
echo "  --- 将执行（B）---"; sed 's/^/    /' /data/tmp/hf_accept_cmd.sh | cut -c1-200
echo "  --- 开始（前台，日志同时落盘）---"
bash /data/tmp/hf_accept_cmd.sh 2>&1 | tee /data/tmp/hf_accept_run.log | tail -45
echo "  --- 退出码: ${PIPESTATUS[0]} ---"
