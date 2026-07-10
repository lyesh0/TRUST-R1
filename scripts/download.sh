#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SEARCH_DATA_ROOT="${SEARCH_DATA_ROOT:-/root/autodl-fs}"

python "$REPO_ROOT/scripts/download.py" --data-root "$SEARCH_DATA_ROOT"
