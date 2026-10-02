#!/bin/bash
# ============================================================================
# start_robotwin_infer_and_eval.sh 的 --task_list_file 行为测试
#
# 无需 GPU:
#   通过 --inference_workdir <不存在> 让 launcher 在「打印任务列表之后、
#   启动任何推理 server 之前」干净退出 (Phase 1 的 cd 校验), 因此本测试
#   只覆盖参数解析 / 任务清单读取 / 护栏, 不会真的起进程。
#
# 覆盖:
#   T1  bash -n 语法
#   T2  从文件读任务清单 (跳过 # 注释与空行)
#   T3  非法任务名 -> 非 0 退出
#   T4  重复任务名 -> 非 0 退出
#   T5  清单文件不存在 -> 非 0 退出
#   T6  清单为空 -> 非 0 退出
#   T7  向后兼容 (不传 --task_list_file)
#   T8  num_tasks 超上限 -> 非 0 退出
#   T9  phase{1..4}_eval.txt 行数 4/8/12/16
#   T10 真实 phase1_eval.txt 被加载
#   T11 --test_num 解析 + 向后兼容
#   T12 --test_num 接线完整 (静态)
#   T13 副本陈旧检测存在 (静态)
#   T13b video 开关可双向控制 (--enable_video / --no_video, 静态 + 实跑)
#   T14 共享副本与源码一致 (只读)
#
# 用法 (在远端 /data/code/lingbot-vla-v2 下):
#   bash tests/test_launcher_task_list.sh
# ============================================================================
set -uo pipefail

REPO="/data/code/lingbot-vla-v2"
LAUNCHER="${REPO}/experiment/robotwin/start_robotwin_infer_and_eval.sh"
EVAL_WORKDIR="/data/code/RoboTwin-lingbot"
CONDA_SH="/data/miniconda3/etc/profile.d/conda.sh"
TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT

PASS=0; FAIL=0
ok()   { echo "  [OK]   $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

# run_launcher <输出文件> [额外参数...]
# 故意给一个不存在的 --inference_workdir, 让它尽早退出。
run_launcher() {
    local out="$1"; shift
    ( cd "$REPO" && timeout 60 bash "$LAUNCHER" \
        --eval_workdir "$EVAL_WORKDIR" \
        --conda_sh "$CONDA_SH" \
        --model_path /nonexistent/checkpoints/global_step_0/hf_ckpt \
        --inference_workdir /nonexistent/inference \
        --output_base "$TMPD/out" \
        "$@" ) > "$out" 2>&1
    echo $?
}

expect_contains() {  # <文件> <子串> <说明>
    if grep -qF "$2" "$1"; then ok "$3"; else
        bad "$3  (未找到: $2)"; echo "    --- 实际输出尾部 ---"; tail -6 "$1" | sed 's/^/    /'
    fi
}

echo "=================================================================="
echo " launcher --task_list_file 行为测试"
echo "=================================================================="
echo

# ---------------------------------------------------------------- T1 语法
if bash -n "$LAUNCHER" 2>/dev/null; then ok "T1  bash -n 语法检查"; else bad "T1  bash -n 语法检查"; fi

# ---------------------------------------------------------------- T2 合法清单
printf '# 注释行\nlift_pot\n\nclick_bell\nturn_switch\nplace_shoe\n' > "$TMPD/good.txt"
rc=$(run_launcher "$TMPD/t2.log" --task_list_file "$TMPD/good.txt" --num_gpus 4)
expect_contains "$TMPD/t2.log" "Task list file: $TMPD/good.txt (4 tasks)" "T2  读文件并统计任务数 (4)"
expect_contains "$TMPD/t2.log" "Tasks this run (4): lift_pot click_bell turn_switch place_shoe" \
    "T2  任务顺序正确 + 跳过 # 注释与空行"

# ---------------------------------------------------------------- T3 非法任务名
printf 'lift_pot\nnot_a_real_task\n' > "$TMPD/bad.txt"
rc=$(run_launcher "$TMPD/t3.log" --task_list_file "$TMPD/bad.txt" --num_gpus 4)
if [ "$rc" != "0" ]; then ok "T3  非法任务名 -> 非 0 退出 (rc=$rc)"; else bad "T3  非法任务名应非 0 退出"; fi
expect_contains "$TMPD/t3.log" "unknown task 'not_a_real_task'" "T3  报出非法任务名"

# ---------------------------------------------------------------- T4 重复任务名
printf 'lift_pot\nclick_bell\nlift_pot\n' > "$TMPD/dup.txt"
rc=$(run_launcher "$TMPD/t4.log" --task_list_file "$TMPD/dup.txt" --num_gpus 4)
if [ "$rc" != "0" ]; then ok "T4  重复任务名 -> 非 0 退出 (rc=$rc)"; else bad "T4  重复任务名应非 0 退出"; fi
expect_contains "$TMPD/t4.log" "duplicate task(s): lift_pot" "T4  报出重复的任务名"

# ---------------------------------------------------------------- T5 文件不存在
rc=$(run_launcher "$TMPD/t5.log" --task_list_file "$TMPD/nope.txt" --num_gpus 4)
if [ "$rc" != "0" ]; then ok "T5  文件不存在 -> 非 0 退出 (rc=$rc)"; else bad "T5  文件不存在应非 0 退出"; fi
expect_contains "$TMPD/t5.log" "not found" "T5  报文件不存在"

# ---------------------------------------------------------------- T6 空文件
: > "$TMPD/empty.txt"
rc=$(run_launcher "$TMPD/t6.log" --task_list_file "$TMPD/empty.txt" --num_gpus 4)
if [ "$rc" != "0" ]; then ok "T6  空清单 -> 非 0 退出 (rc=$rc)"; else bad "T6  空清单应非 0 退出"; fi
expect_contains "$TMPD/t6.log" "contains no task" "T6  报清单为空"

# ---------------------------------------------------------------- T7 向后兼容
rc=$(run_launcher "$TMPD/t7.log" --num_tasks 4 --num_gpus 4)
expect_contains "$TMPD/t7.log" "Tasks this run (4): lift_pot hanging_mug stack_bowls_three scan_object" \
    "T7  不传 --task_list_file 时取 task_list_all 前 N 个 (原行为)"

# ---------------------------------------------------------------- T8 num_tasks 边界
rc=$(run_launcher "$TMPD/t8.log" --num_tasks 51 --num_gpus 4)
if [ "$rc" != "0" ]; then ok "T8  num_tasks=51 -> 非 0 退出 (rc=$rc)"; else bad "T8  num_tasks=51 应非 0 退出"; fi
expect_contains "$TMPD/t8.log" "exceeds max 50" "T8  报 num_tasks 超上限"

# ---------------------------------------------------------------- T9 真实 eval 清单
for p in 1 2 3 4; do
    f="/data/train/phases/phase${p}_eval.txt"
    if [ -f "$f" ]; then
        want=$(( p * 4 ))
        got=$(grep -vc '^[[:space:]]*#\|^[[:space:]]*$' "$f")
        if [ "$got" = "$want" ]; then ok "T9  phase${p}_eval.txt 正文 $got 行 (期望 $want)"; else
            bad "T9  phase${p}_eval.txt 正文 $got 行, 期望 $want"; fi
    else
        bad "T9  缺少 $f"
    fi
done

# ---------------------------------------------------------------- T10 真实清单能被 launcher 接受
rc=$(run_launcher "$TMPD/t10.log" --task_list_file /data/train/phases/phase1_eval.txt --num_gpus 4)
expect_contains "$TMPD/t10.log" "Tasks this run (4): lift_pot click_alarmclock turn_switch place_shoe" \
    "T10 真实 phase1_eval.txt 被正确加载"

# ---------------------------------------------------------------- T11 --test_num 被解析
rc=$(run_launcher "$TMPD/t11.log" --task_list_file "$TMPD/good.txt" --test_num 4 --num_gpus 4)
if grep -q "Unknown argument" "$TMPD/t11.log"; then
    bad "T11 --test_num 未被识别为合法参数"
else
    ok "T11 --test_num 被正确解析"
fi
# 不传 --test_num 也要能跑 (向后兼容)
rc=$(run_launcher "$TMPD/t11b.log" --task_list_file "$TMPD/good.txt" --num_gpus 4)
if grep -q "Unknown argument" "$TMPD/t11b.log"; then
    bad "T11b 不传 --test_num 时报错"
else
    ok "T11b 不传 --test_num 正常 (向后兼容)"
fi

# ---------------------------------------------------------------- T12 --test_num 已接入 eval client 调用
# 静态断言: 解析 -> 拼参数 -> 拼进 eval client 命令行, 三处都要在。
if grep -qF 'test_num_arg="--test_num ${test_num}"' "$LAUNCHER" \
   && grep -qF '${test_num_arg}" >> "$log_file"' "$LAUNCHER" \
   && grep -qF -- '--test_num)          test_num="$2"' "$LAUNCHER"; then
    ok "T12 --test_num 已完整接入 (解析 -> 拼参数 -> 传入 eval client)"
else
    bad "T12 --test_num 接线不完整"
fi

# ---------------------------------------------------------------- T13 副本陈旧会被同步 (静态)
if grep -qF 'cmp -s "$eval_client_src" "$eval_client_dst"' "$LAUNCHER" \
   && grep -qF 'cmp -s "$deploy_pkg_src/$f" "$deploy_pkg_dst/$f"' "$LAUNCHER"; then
    ok "T13 eval client / deploy 辅助模块 有陈旧检测与同步"
else
    bad "T13 缺少副本陈旧检测"
fi

# ---------------------------------------------------------------- T13b video 开关可双向控制
# 上游的 enable_video 默认 False 且只有 --no_video —— 没有开关能打开它,
# 于是调度器的 --video 变成摆设。这里锁住"补上了对称的 --enable_video"。
if grep -qF -- '--enable_video)' "$LAUNCHER"; then
    ok "T13b launcher 已支持 --enable_video"
else
    bad "T13b launcher 缺少 --enable_video (调度器的 --video 会变成摆设)"
fi
# 死代码: enable_video 被连续赋值两次, 前一次无效
if [ "$(grep -c '^enable_video=' "$LAUNCHER")" = "1" ]; then
    ok "T13b launcher 的 enable_video 默认值只有一处赋值 (无死代码)"
else
    bad "T13b launcher 里 enable_video 仍有多处赋值 (存在被覆盖的死代码)"
fi
# 真跑一遍, 确认 --enable_video 不会被当成未知参数
rc=$(run_launcher "$TMPD/t13b.log" --task_list_file "$TMPD/good.txt" --enable_video --num_gpus 4)
if grep -q "Unknown argument" "$TMPD/t13b.log"; then
    bad "T13b --enable_video 未被识别为合法参数"
else
    ok "T13b --enable_video 被正确解析"
fi

# ---------------------------------------------------------------- T14 当前共享副本与源码一致 (只读)
SRC_CLIENT="${REPO}/experiment/robotwin/eval_policy_client_lingbotvla.py"
DST_CLIENT="${EVAL_WORKDIR}/script/eval_policy_client_lingbotvla.py"
if [ -f "$DST_CLIENT" ]; then
    if cmp -s "$SRC_CLIENT" "$DST_CLIENT"; then
        ok "T14 RoboTwin/script 里的 eval client 副本与源码一致"
    else
        bad "T14 eval client 副本与源码不一致 (下次跑 launcher 会被自动同步)"
    fi
else
    ok "T14 RoboTwin/script 里暂无副本 (launcher 首次运行会创建)"
fi

echo
echo "=================================================================="
echo "  $PASS 通过, $FAIL 失败"
echo "=================================================================="
[ "$FAIL" -eq 0 ]
