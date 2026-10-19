#!/usr/bin/env bash
# 默认CPU预览；显式--execute才启动，--shutdown由用户选择。
set -euo pipefail
RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-/data/miniconda3/envs/lingbotvla/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python}"
fi
exec "$PYTHON_BIN" "$RECIPE_DIR/20261019_place_mouse_pad_mouse_aux.py" "$@"
