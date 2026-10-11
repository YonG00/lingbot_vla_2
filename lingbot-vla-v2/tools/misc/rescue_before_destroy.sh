#!/usr/bin/env bash
# =============================================================================
# 销毁实例前的挽救脚本：把 overlay 上要留的模型搬到持久卷 /workspace/keep
# 用法：
#   bash rescue_before_destroy.sh                 # 只列出清单（默认，安全）
#   bash rescue_before_destroy.sh --copy          # 复制常见的 HF 导出到 /workspace/keep
#   bash rescue_before_destroy.sh --copy --run <run_dir_name>
# =============================================================================
set -uo pipefail
OVERLAY_ROOT=${OVERLAY_ROOT:-/models/robotwin-persistent/outputs}
KEEP_ROOT=${KEEP_ROOT:-/workspace/keep}
DO_COPY=0; RUN_FILTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --copy) DO_COPY=1 ;;
    --run) RUN_FILTER=${2:-}; shift ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac; shift
done

echo "== overlay 存档（$OVERLAY_ROOT）=="
if [ ! -d "$OVERLAY_ROOT" ]; then echo "  (不存在)"; else
  du -sh "$OVERLAY_ROOT"/* 2>/dev/null | sort -rh | head -20
fi
echo
echo "== HF 导出（要长期保留的就是它）=="
find "$OVERLAY_ROOT" -maxdepth 4 -type d -name hf_ckpt 2>/dev/null | while read -r d; do
  printf "  %-70s %s\n" "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done
echo
echo "== /workspace 现状 =="
df -h /workspace | tail -1
du -sh "$KEEP_ROOT" 2>/dev/null || echo "  ($KEEP_ROOT 尚未创建)"
echo
echo "== ⛔ 销毁前检查清单 =="
cat <<'EOF'
  [ ] 所有要留的 hf_ckpt 已复制到 /workspace/keep（或已 rsync 回外部机器）
  [ ] /workspace 剩余空间充足（HF 导出约 12 GB/份）
  [ ] 训练日志/事件记录已一并保留（与 hf_ckpt 同目录的 run 元数据）
  [ ] 确认无误后再在平台控制台 Destroy Instance
EOF

if [ "$DO_COPY" = "1" ]; then
  mkdir -p "$KEEP_ROOT"
  echo
  echo "== 开始复制 =="
  find "$OVERLAY_ROOT" -maxdepth 4 -type d -name hf_ckpt 2>/dev/null | while read -r src; do
    run=$(basename "$(dirname "$(dirname "$src")")")
    [ -n "$RUN_FILTER" ] && [ "$run" != "$RUN_FILTER" ] && continue
    dst="$KEEP_ROOT/${run}_$(basename "$src")"
    if [ -e "$dst" ]; then echo "  跳过（已存在）: $dst"; continue; fi
    echo "  复制: $src -> $dst"
    cp -a "$src" "$dst" && echo "    ✓ $(du -sh "$dst" | cut -f1)"
  done
  echo "== 完成，/workspace 现状 =="
  df -h /workspace | tail -1
fi
