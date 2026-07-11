#!/usr/bin/env bash
set -euo pipefail

CHECK_ONLY="false"
SKIP_FLASH_ATTN="false"
FORCE="false"

# Official Search-R1 README defaults:
#   conda create -n searchr1 python=3.9
#   pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
#   pip3 install vllm==0.6.3
#   pip install -e .
#   pip3 install flash-attn --no-build-isolation
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
VLLM_VERSION="${VLLM_VERSION:-0.6.3}"

usage() {
  cat <<'USAGE'
Usage: bash scripts/setup_dependencies.sh [options]

Check and install TRUST-R1 dependencies in the current Python environment.
This follows the official Search-R1 install order more closely:
  1) install PyTorch
  2) install vLLM
  3) install the project dependencies
  4) install the project itself
  5) install flash-attn separately

Options:
  --check-only       Only report dependency status; do not install anything.
  --no-flash-attn    Skip flash-attn check/install.
  --force            Run installation steps even if checks already pass.
  -h, --help         Show this help.

This script only manages Python dependencies. It does not download data/models,
build indexes, start retriever/Ray/vLLM, run smoke tests, train, or evaluate.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only) CHECK_ONLY="true"; shift ;;
    --no-flash-attn) SKIP_FLASH_ATTN="true"; shift ;;
    --force) FORCE="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REQ_FILE="$ROOT_DIR/requirements.txt"
LOG_DIR="$ROOT_DIR/logs/dependency_setup"
TMP_REQ=""

cleanup() {
  if [[ -n "$TMP_REQ" && -f "$TMP_REQ" ]]; then
    rm -f "$TMP_REQ"
  fi
}
trap cleanup EXIT

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

python_cmd() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    echo "python3"
  fi
}

PYTHON_BIN="$(python_cmd)"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found. Activate the intended environment first." >&2
  exit 1
fi

if [[ ! -f "$REQ_FILE" ]]; then
  echo "requirements.txt not found: $REQ_FILE" >&2
  exit 1
fi

print_env_info() {
  log "Environment"
  echo "repo: $ROOT_DIR"
  echo "python: $(command -v "$PYTHON_BIN")"
  "$PYTHON_BIN" --version || true
  "$PYTHON_BIN" -m pip --version || true
  "$PYTHON_BIN" - <<'PY' || true
import sys
if sys.version_info[:2] != (3, 9):
    print(f"WARNING: official Search-R1 README uses Python 3.9; current Python is {sys.version.split()[0]}")
PY
  echo "os: $(uname -a)"
  echo "CUDA_HOME: ${CUDA_HOME:-<unset>}"
  if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc: $(command -v nvcc)"
    nvcc --version | sed 's/^/  /' || true
  else
    echo "nvcc: not found"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi: $(command -v nvidia-smi)"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/  /' || nvidia-smi || true
  else
    echo "nvidia-smi: not found"
  fi
  "$PYTHON_BIN" - <<'PY' || true
try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
except Exception as exc:
    print(f"torch: unavailable ({exc})")
PY
}

check_dependencies() {
  local skip_flash="$1"
  "$PYTHON_BIN" - "$REQ_FILE" "$skip_flash" "$TORCH_VERSION" "$VLLM_VERSION" <<'PY'
from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import re
import sys
from pathlib import Path

req_file = Path(sys.argv[1])
skip_flash = sys.argv[2] == "true"
torch_version = sys.argv[3]
vllm_version = sys.argv[4]

try:
    from packaging.requirements import Requirement
    from packaging.version import Version
except Exception as exc:
    print(f"packaging: missing ({exc})")
    sys.exit(1)

bootstrap_requirements = [
    "pip",
    "setuptools",
    "wheel",
    "packaging",
    "ninja",
    "pybind11",
]
official_requirements = [
    f"torch=={torch_version}",
    f"vllm=={vllm_version}",
]

file_requirements = []
for line in req_file.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    if re.match(r"^flash[-_]attn(\b|[<>=!~])", line, re.IGNORECASE):
        continue
    if re.match(r"^vllm(\b|[<>=!~])", line, re.IGNORECASE):
        continue
    file_requirements.append(line)

requirements = []
for item in bootstrap_requirements + official_requirements + file_requirements:
    try:
        requirements.append(Requirement(item))
    except Exception as exc:
        print(f"requirement-parse: bad requirement {item!r}: {exc}")
        sys.exit(1)

ok = True
seen = set()
print("\nDependency packages:")
for req in requirements:
    name = req.name
    norm = name.lower().replace("_", "-")
    if norm in seen:
        continue
    seen.add(norm)
    try:
        installed = metadata.version(name)
    except metadata.PackageNotFoundError:
        print(f"  missing          {name}{req.specifier}")
        ok = False
        continue
    if req.specifier and Version(installed) not in req.specifier:
        print(f"  version-mismatch {name}{req.specifier} installed={installed}")
        ok = False
    else:
        print(f"  ok               {name} installed={installed}")

imports = ["torch", "transformers", "ray", "datasets", "vllm", "verl"]
if not skip_flash:
    imports.append("flash_attn")

print("\nImport smoke checks:")
for mod in imports:
    spec = importlib.util.find_spec(mod)
    if spec is None:
        print(f"  missing          {mod}")
        ok = False
    else:
        print(f"  ok               {mod}")

sys.exit(0 if ok else 1)
PY
}

create_filtered_requirements() {
  TMP_REQ="$(mktemp)"
  "$PYTHON_BIN" - "$REQ_FILE" "$TMP_REQ" <<'PY'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
lines = []
for line in src.read_text().splitlines():
    stripped = line.strip()
    if re.match(r"^flash[-_]attn(\b|[<>=!~])", stripped, re.IGNORECASE):
        continue
    if re.match(r"^vllm(\b|[<>=!~])", stripped, re.IGNORECASE):
        continue
    lines.append(line)
dst.write_text("\n".join(lines) + "\n")
PY
}

ensure_not_local_macos_install() {
  if [[ "$(uname -s)" == "Darwin" && "$CHECK_ONLY" != "true" ]]; then
    echo "Refusing to install dependencies on local macOS." >&2
    echo "Run this script on AutoDL, or use --check-only for local inspection." >&2
    exit 3
  fi
}

check_flash_attn_prereqs() {
  "$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"PyTorch is not importable; install torch before flash-attn. error={exc}", file=sys.stderr)
    sys.exit(1)
if torch.version.cuda is None:
    print("PyTorch is installed without CUDA support; flash-attn needs a CUDA PyTorch build.", file=sys.stderr)
    sys.exit(1)
print(f"PyTorch for flash-attn: torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
PY
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "WARNING: nvcc not found. flash-attn may fail if no compatible wheel is available." >&2
  fi
}

print_flash_attn_debug_info() {
  local flash_log="$1"
  echo "flash-attn installation failed." >&2
  echo "Full install log: $flash_log" >&2
  echo >&2
  echo "Diagnostic summary:" >&2
  "$PYTHON_BIN" --version >&2 || true
  "$PYTHON_BIN" - <<'PY' >&2 || true
import os
try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
except Exception as exc:
    print(f"torch: unavailable ({exc})")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', '<unset>')}")
PY
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version >&2 || true
  else
    echo "nvcc: not found" >&2
  fi
  echo >&2
  echo "Common causes: PyTorch/CUDA version mismatch, missing CUDA toolkit headers/compiler," >&2
  echo "or an unsupported Python/PyTorch/flash-attn wheel combination." >&2
}

run_logged() {
  local label="$1"
  shift
  mkdir -p "$LOG_DIR"
  local log_file="$LOG_DIR/${label}_$(date +%Y%m%d_%H%M%S).log"
  log "Running $label; log: $log_file"
  if ! "$@" 2>&1 | tee "$log_file"; then
    echo "Step failed: $label" >&2
    echo "See log: $log_file" >&2
    return 1
  fi
}

install_dependencies() {
  ensure_not_local_macos_install

  run_logged bootstrap "$PYTHON_BIN" -m pip install -U pip setuptools wheel packaging ninja pybind11

  run_logged torch "$PYTHON_BIN" -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"

  run_logged vllm "$PYTHON_BIN" -m pip install "vllm==$VLLM_VERSION"

  create_filtered_requirements
  run_logged project-requirements "$PYTHON_BIN" -m pip install -r "$TMP_REQ"

  run_logged editable-install "$PYTHON_BIN" -m pip install -e . --no-deps

  if [[ "$SKIP_FLASH_ATTN" == "true" ]]; then
    log "Skipping flash-attn install because --no-flash-attn was set"
    return 0
  fi

  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("flash_attn") else 1)
PY
  then
    log "flash-attn already importable; skipping install"
    return 0
  fi

  check_flash_attn_prereqs
  mkdir -p "$LOG_DIR"
  local flash_log="$LOG_DIR/flash_attn_install_$(date +%Y%m%d_%H%M%S).log"
  log "Installing flash-attn; log: $flash_log"
  if ! "$PYTHON_BIN" -m pip install flash-attn --no-build-isolation > "$flash_log" 2>&1; then
    print_flash_attn_debug_info "$flash_log"
    return 1
  fi
}

print_env_info

log "Checking dependencies"
if check_dependencies "$SKIP_FLASH_ATTN"; then
  if [[ "$FORCE" != "true" ]]; then
    log "All requested dependencies are present. Nothing to install."
    exit 0
  fi
  log "All requested dependencies are present, but --force was set."
else
  if [[ "$CHECK_ONLY" == "true" ]]; then
    log "Dependency check failed and --check-only was set; not installing."
    exit 1
  fi
fi

install_dependencies

log "Re-checking dependencies"
check_dependencies "$SKIP_FLASH_ATTN"
log "Dependency setup complete"
