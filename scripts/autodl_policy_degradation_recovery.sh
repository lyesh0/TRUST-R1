#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias retained for existing AutoDL notes/commands.
# The canonical B0 pilot-safe launcher is scripts/run_b0_pilot_safe.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_b0_pilot_safe.sh" "$@"
