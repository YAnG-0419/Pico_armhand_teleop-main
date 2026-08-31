#!/usr/bin/env bash
# Compatibility alias for the dedicated PICO + MANUS/Wuji runtime.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${repo_root}/ops/run/run_teleop.sh" "$@"
