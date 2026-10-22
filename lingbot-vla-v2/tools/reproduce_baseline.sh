#!/usr/bin/env bash
set -euo pipefail
package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$package_root"
exec "${PYTHON:-python}" experiment/reproduction/run.py "$@"
