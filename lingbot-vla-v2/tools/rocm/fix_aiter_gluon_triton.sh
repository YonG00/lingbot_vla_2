#!/usr/bin/env bash
# =============================================================================
# 修复 AMD 镜像的 aiter gluon / triton 版本硬失败（**实例重建后必须重跑**）
# -----------------------------------------------------------------------------
# 症状链（一个根因，串起"训练起不来"的全部表象）：
#   /opt/aiter/aiter/ops/triton/gluon/__init__.py 在 import 时校验 triton>=3.6.0，
#   而镜像里是 triton 3.5.1+rocm7.2.1 ⇒ raise RuntimeError
#     ⇒ flash_attn（ROCm 版）硬依赖 aiter（该 import 不在 try/except 内）导入失败
#       ⇒ lingbotvla 模型模块导入失败（注册表架构数 = 0）
#         ⇒ get_loader() 退回 HuggingfaceLoader
#           ⇒ AutoModel.from_config 抛 `ValueError: Unrecognized configuration class`
#
# 为什么会在重建后复发：修复改的是 /opt/aiter（**容器可写层**），实例重建即丢失。
#
# ---------------------------------------------------------------------------
# 两种修法（本脚本优先用无需改文件的那种）
# ---------------------------------------------------------------------------
# 方案 A（推荐，零侵入）：环境变量 `AITER_USE_SYSTEM_TRITON=1`
#   aiter 源码里本来就是"设了这个变量 ⇒ 只 warnings.warn 而不 raise"：
#       if int(os.environ.get("AITER_USE_SYSTEM_TRITON", 0)): warnings.warn(...)
#       else: raise RuntimeError(...)
#   ⇒ 本脚本会把它写进 /etc/profile.d（登录 shell 生效）与
#     /root/.bashrc（非登录/交互 shell 生效），并**当场验证**。
#
# 方案 B（兜底，改文件）：把那条 `raise RuntimeError(...)` 整句替换为
#   `warnings.warn(...)`，改动前备份为 `__init__.py.bak-<时间戳>`；
#   `--revert` 可还原。仅当方案 A 因故不生效时使用。
#
# 用法：
#   bash tools/rocm/fix_aiter_gluon_triton.sh            # 方案 A（若有效则不改文件）
#   bash tools/rocm/fix_aiter_gluon_triton.sh --patch    # 强制用方案 B 改文件
#   bash tools/rocm/fix_aiter_gluon_triton.sh --revert   # 还原方案 B 的改动
#   bash tools/rocm/fix_aiter_gluon_triton.sh --check    # 只检查当前状态（不改任何东西）
# 幂等：重复执行安全。
# =============================================================================
set -uo pipefail

PY=${PY:-/opt/robotwin-env/bin/python}
[ -x "$PY" ] || PY=python3
GLUON_INIT="/opt/aiter/aiter/ops/triton/gluon/__init__.py"
PROFILE_D="/etc/profile.d/aiter-triton-compat.sh"
BASHRC="/root/.bashrc"
MARK="# aiter-triton-compat (fix_aiter_gluon_triton.sh)"

MODE=auto
case "${1:-}" in
  --patch)  MODE=patch ;;
  --revert) MODE=revert ;;
  --check)  MODE=check ;;
  "")       MODE=auto ;;
  *) echo "未知参数: $1"; exit 2 ;;
esac

echo "== aiter gluon / triton 兼容修复（模式=$MODE，解释器=$PY）=="

verify_env() {
  AITER_USE_SYSTEM_TRITON=1 "$PY" - <<'EOF' 2>&1 | tail -5
import warnings
warnings.simplefilter("ignore")
try:
    import aiter
    print("  aiter OK")
except Exception as exc:
    print("  aiter FAIL:", str(exc)[:120])
try:
    import flash_attn
    print("  flash_attn OK")
except Exception as exc:
    print("  flash_attn FAIL:", str(exc)[:120])
EOF
}

verify_registry() {
  ( cd /workspace/lingbot_vla_2/lingbot-vla-v2 2>/dev/null || exit 0
    AITER_USE_SYSTEM_TRITON=1 PYTHONPATH="$PWD" "$PY" - <<'EOF' 2>&1 | tail -2
import warnings
warnings.simplefilter("ignore")
try:
    from lingbotvla.models.registry import get_registry
    print("  注册表架构数:", len(get_registry().supported_models))
except Exception as exc:
    print("  注册表查询失败:", str(exc)[:120])
EOF
  )
}

if [ "$MODE" = "check" ]; then
  echo "-- 当前状态（未设 AITER_USE_SYSTEM_TRITON）--"
  "$PY" -c "import aiter" 2>&1 | tail -2
  echo "-- 设了 AITER_USE_SYSTEM_TRITON=1 --"
  verify_env
  verify_registry
  exit 0
fi

if [ "$MODE" = "revert" ]; then
  bak=$(ls -1t "${GLUON_INIT}".bak-* 2>/dev/null | head -1 || true)
  if [ -n "${bak:-}" ]; then
    cp -a "$bak" "$GLUON_INIT" && echo "已还原: $GLUON_INIT ← $bak"
  else
    echo "没有找到备份（${GLUON_INIT}.bak-*），无需还原"
  fi
  rm -f "$PROFILE_D"
  sed -i "/aiter-triton-compat/d" "$BASHRC" 2>/dev/null || true
  exit 0
fi

# ---------------- 方案 A：环境变量 ----------------
write_env() {
  cat > "$PROFILE_D" <<EOF
$MARK
# aiter gluon 在 triton<3.6 时硬 raise；设此变量后改为 warning（flash_attn 需要它）
export AITER_USE_SYSTEM_TRITON=1
EOF
  chmod 644 "$PROFILE_D"
  if ! grep -q "$MARK" "$BASHRC" 2>/dev/null; then
    { echo ""; echo "$MARK"; echo "export AITER_USE_SYSTEM_TRITON=1"; } >> "$BASHRC"
  fi
  echo "已写入: $PROFILE_D 与 $BASHRC"
}

if [ "$MODE" = "auto" ]; then
  write_env
  echo "-- 验证（方案 A）--"
  out=$(verify_env)
  echo "$out"
  if echo "$out" | grep -q "aiter OK" && echo "$out" | grep -q "flash_attn OK"; then
    echo "== 方案 A 生效，未改动任何文件 =="
    verify_registry
    exit 0
  fi
  echo "!! 方案 A 未生效 ⇒ 自动转方案 B（改文件）"
  MODE=patch
fi

# ---------------- 方案 B：改文件 ----------------
if [ "$MODE" = "patch" ]; then
  if [ ! -f "$GLUON_INIT" ]; then
    echo "找不到 $GLUON_INIT（非 AMD 镜像？）"; exit 1
  fi
  if grep -q "raise RuntimeError" "$GLUON_INIT"; then
    bak="${GLUON_INIT}.bak-$(date +%Y%m%d_%H%M%S)"
    cp -a "$GLUON_INIT" "$bak"
    "$PY" - "$GLUON_INIT" <<'EOF'
import re, sys
p = sys.argv[1]
src = open(p, encoding="utf-8").read()
# 把 `raise RuntimeError(\n ... \n )` 整句替换为等价 warnings.warn(...)
new, n = re.subn(
    r"raise RuntimeError\(\s*(f\"aiter gluon kernels require triton>=3\.6\.0[^\"]*\")\s*\)",
    r"warnings.warn(\1)", src, count=1)
if n == 0:
    print("  未匹配到 raise 语句（可能已修过或有版本差异）")
    sys.exit(0)
open(p, "w", encoding="utf-8").write(new)
print(f"  已把 raise 改为 warning（{n} 处）")
EOF
    echo "  备份: $bak"
  else
    echo "  文件里没有 raise RuntimeError（已修过）"
  fi
  write_env
  echo "-- 验证（方案 B）--"
  verify_env
  verify_registry
fi
