#!/usr/bin/env bash
# 默认只预览；追加 --execute 才启动，--shutdown 控制正常完成后的关机。
set -euo pipefail
RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-/data/miniconda3/envs/lingbotvla/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python}"
fi
exec "$PYTHON_BIN" "$RECIPE_DIR/20261018_place_mouse_pad_all50_9680_ee_lnfm.py" "$@"
