#!/usr/bin/env bash
set -euo pipefail

MODE="smoke"
ALGO="grpo"
MODEL="/root/autodl-tmp/models/Qwen2.5-3B"
DATA_DIR="/root/autodl-tmp/data/nq_search"
RUN_ROOT="/root/autodl-tmp/runs"
SEARCH_DATA_ROOT="${SEARCH_DATA_ROOT:-/root/autodl-fs}"
RETRIEVER_URL="http://127.0.0.1:8000/retrieve"
RETRIEVER_INDEX="${RETRIEVER_INDEX:-$SEARCH_DATA_ROOT/indexes/wiki-18/e5_Flat.index}"
RETRIEVER_CORPUS="${RETRIEVER_CORPUS:-$SEARCH_DATA_ROOT/data/wiki-18-extracted.jsonl}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-/root/e5-base-v2}"
FAISS_GPU="${FAISS_GPU:-false}"
AUTO_START_RETRIEVER="true"
PREPARE_DATA="auto"
EXPERIMENTS="B0,B1,M1,M2"
FAULT_MODE="mixed"
FAULT_RATE="0.2"
SEED="42"
DRY_RUN_STEPS="2"
TOTAL_STEPS="100"
GPUS_PER_NODE="4"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NNODES="1"
MAX_TURNS="2"
TOPK="3"
TRUST_LOGGING="true"
SKIP_SMOKE="false"

usage() {
  cat <<'USAGE'
Usage: bash scripts/autodl_one_click_experiment.sh [options]

One-click AutoDL launcher for TRUST-R1 experiments. It runs the existing
scripts/run_trust_r1_experiments.sh with a safe sequence:

  smoke: check -> prepare data if missing -> retriever health/start -> B0 dry-run -> M2 dry-run
  core:  smoke -> train B0,B1,M1,M2
  eval:  check -> retriever health/start -> eval B0,B1,M1,M2

Common options:
  --mode smoke|core|eval
  --algo grpo|ppo
  --model PATH_OR_HF_ID
  --data-dir PATH
  --run-root PATH
  --search-data-root PATH      Default: /root/autodl-fs for shared retriever data
  --retriever-url URL
  --retriever-index PATH       Default: $SEARCH_DATA_ROOT/indexes/wiki-18/e5_Flat.index
  --retriever-corpus PATH      Default: $SEARCH_DATA_ROOT/data/wiki-18-extracted.jsonl
  --retriever-name NAME        Default: e5
  --retriever-model PATH_OR_ID Default: intfloat/e5-base-v2
  --faiss-gpu true|false       Default: false, avoids retriever competing with 4-GPU training
  --auto-start-retriever true|false
  --prepare-data auto|true|false
  --experiments B0,B1,M1,M2
  --fault-mode clean|empty|drop_top|duplicate|mixed
  --fault-rate FLOAT
  --seed INT
  --dry-run-steps INT
  --total-steps INT
  --gpus-per-node INT
  --cuda-visible-devices IDS  Default: 0,1,2,3
  --nnodes INT
  --max-turns INT
  --topk INT
  --trust-logging true|false
  --skip-smoke true|false      Only meaningful for --mode core

Prepare retriever assets first if they are missing:
  SEARCH_DATA_ROOT=/root/autodl-fs
  python scripts/download.py --data-root "$SEARCH_DATA_ROOT"

Examples:
  bash scripts/autodl_one_click_experiment.sh --mode smoke

  bash scripts/autodl_one_click_experiment.sh \
    --mode core \
    --model /root/autodl-tmp/models/Qwen2.5-3B \
    --data-dir /root/autodl-tmp/data/nq_search \
    --search-data-root /root/autodl-fs \
    --retriever-index /root/autodl-fs/indexes/wiki-18/e5_Flat.index \
    --retriever-corpus /root/autodl-fs/data/wiki-18-extracted.jsonl \
    --gpus-per-node 4 \
    --cuda-visible-devices 0,1,2,3 \
    --total-steps 100
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --algo) ALGO="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --search-data-root)
      SEARCH_DATA_ROOT="$2"
      RETRIEVER_INDEX="$SEARCH_DATA_ROOT/indexes/wiki-18/e5_Flat.index"
      RETRIEVER_CORPUS="$SEARCH_DATA_ROOT/data/wiki-18-extracted.jsonl"
      shift 2
      ;;
    --retriever-url) RETRIEVER_URL="$2"; shift 2 ;;
    --retriever-index) RETRIEVER_INDEX="$2"; shift 2 ;;
    --retriever-corpus) RETRIEVER_CORPUS="$2"; shift 2 ;;
    --retriever-name) RETRIEVER_NAME="$2"; shift 2 ;;
    --retriever-model) RETRIEVER_MODEL="$2"; shift 2 ;;
    --faiss-gpu) FAISS_GPU="$2"; shift 2 ;;
    --auto-start-retriever) AUTO_START_RETRIEVER="$2"; shift 2 ;;
    --prepare-data) PREPARE_DATA="$2"; shift 2 ;;
    --experiments) EXPERIMENTS="$2"; shift 2 ;;
    --fault-mode) FAULT_MODE="$2"; shift 2 ;;
    --fault-rate) FAULT_RATE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --dry-run-steps) DRY_RUN_STEPS="$2"; shift 2 ;;
    --total-steps) TOTAL_STEPS="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --nnodes) NNODES="$2"; shift 2 ;;
    --max-turns) MAX_TURNS="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --trust-logging) TRUST_LOGGING="$2"; shift 2 ;;
    --skip-smoke) SKIP_SMOKE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

is_true() {
  [[ "$1" == "true" || "$1" == "1" || "$1" == "yes" ]]
}

require_autodl() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "Refusing to launch experiments on local macOS. Run this script on AutoDL." >&2
    exit 3
  fi
  if [[ ! -d /root/autodl-tmp ]]; then
    echo "Expected /root/autodl-tmp. This does not look like the configured AutoDL environment." >&2
    exit 3
  fi
}

validate_mode() {
  case "$MODE" in
    smoke|core|eval) ;;
    *) echo "Unknown mode: $MODE" >&2; usage; exit 2 ;;
  esac
}

check_git_state() {
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log "git commit: $(git rev-parse --short HEAD)"
    if [[ -n "$(git status --porcelain)" ]]; then
      log "WARNING: git working tree has uncommitted changes. Do not use this run for final reported numbers unless the diff is recorded."
      git status --short || true
    fi
  else
    log "WARNING: no git repository detected; commit hash will not be reliable."
  fi
}

run_experiment_script() {
  bash scripts/run_trust_r1_experiments.sh \
    "$@" \
    --algo "$ALGO" \
    --model "$MODEL" \
    --data-dir "$DATA_DIR" \
    --run-root "$RUN_ROOT" \
    --retriever-url "$RETRIEVER_URL" \
    --fault-mode "$FAULT_MODE" \
    --fault-rate "$FAULT_RATE" \
    --seed "$SEED" \
    --dry-run-steps "$DRY_RUN_STEPS" \
    --total-steps "$TOTAL_STEPS" \
    --gpus-per-node "$GPUS_PER_NODE" \
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
    --nnodes "$NNODES" \
    --max-turns "$MAX_TURNS" \
    --topk "$TOPK" \
    --trust-logging "$TRUST_LOGGING"
}

check_paths() {
  mkdir -p "$RUN_ROOT"

  if [[ "$MODEL" == /* && ! -e "$MODEL" ]]; then
    echo "Model path does not exist: $MODEL" >&2
    echo "Use --model with an existing AutoDL path or a Hugging Face model id." >&2
    exit 5
  fi
}

data_ready() {
  [[ -f "$DATA_DIR/train.parquet" && -f "$DATA_DIR/test.parquet" ]]
}

ensure_data() {
  case "$PREPARE_DATA" in
    false)
      if ! data_ready; then
        echo "Data missing under $DATA_DIR and --prepare-data false was set." >&2
        exit 5
      fi
      log "Data exists: $DATA_DIR"
      ;;
    true)
      log "Preparing data under $DATA_DIR"
      run_experiment_script --stage prepare-data
      ;;
    auto)
      if data_ready; then
        log "Data exists: $DATA_DIR"
      else
        log "Data missing under $DATA_DIR; preparing it now"
        run_experiment_script --stage prepare-data
      fi
      ;;
    *) echo "Unknown --prepare-data value: $PREPARE_DATA" >&2; exit 2 ;;
  esac
}

retriever_healthy() {
  curl -fsS \
    -X POST "$RETRIEVER_URL" \
    -H 'Content-Type: application/json' \
    -d '{"queries":["health check"],"topk":1,"return_scores":false}' \
    >/dev/null 2>&1
}

start_retriever() {
  if [[ -z "$RETRIEVER_INDEX" || -z "$RETRIEVER_CORPUS" ]]; then
    echo "Retriever is not healthy at $RETRIEVER_URL." >&2
    echo "Provide --retriever-index and --retriever-corpus to auto-start it, or start the retriever manually." >&2
    exit 6
  fi
  if [[ ! -f "$RETRIEVER_INDEX" ]]; then
    echo "Retriever index not found: $RETRIEVER_INDEX" >&2
    exit 6
  fi
  if [[ ! -f "$RETRIEVER_CORPUS" ]]; then
    echo "Retriever corpus not found: $RETRIEVER_CORPUS" >&2
    exit 6
  fi

  local retriever_log="$RUN_ROOT/retriever_$(date +%Y%m%d_%H%M%S).log"
  local faiss_flag=()
  if is_true "$FAISS_GPU"; then
    faiss_flag=(--faiss_gpu)
  fi

  log "Starting retriever; log: $retriever_log"
  nohup python3 search_r1/search/retrieval_server.py \
    --index_path "$RETRIEVER_INDEX" \
    --corpus_path "$RETRIEVER_CORPUS" \
    --topk "$TOPK" \
    --retriever_name "$RETRIEVER_NAME" \
    --retriever_model "$RETRIEVER_MODEL" \
    "${faiss_flag[@]}" \
    > "$retriever_log" 2>&1 &

  for _ in {1..30}; do
    sleep 2
    if retriever_healthy; then
      log "Retriever is healthy: $RETRIEVER_URL"
      return 0
    fi
  done

  echo "Retriever did not become healthy after startup. Check log: $retriever_log" >&2
  exit 6
}

ensure_retriever() {
  if retriever_healthy; then
    log "Retriever is healthy: $RETRIEVER_URL"
    return 0
  fi

  if is_true "$AUTO_START_RETRIEVER"; then
    start_retriever
  else
    echo "Retriever is not healthy at $RETRIEVER_URL and auto-start is disabled." >&2
    exit 6
  fi
}

run_smoke() {
  log "Running B0 dry-run"
  run_experiment_script --stage dry-run --experiment B0

  log "Running M2 dry-run"
  run_experiment_script --stage dry-run --experiment M2
}

run_core() {
  log "Running core suite: $EXPERIMENTS"
  run_experiment_script --suite core --experiments "$EXPERIMENTS"
}

run_eval() {
  log "Running eval suite: $EXPERIMENTS"
  run_experiment_script --suite eval --experiments "$EXPERIMENTS"
}

validate_mode
require_autodl
check_git_state
check_paths

log "Running environment check"
run_experiment_script --stage check

if [[ "$MODE" != "eval" ]]; then
  ensure_data
fi
ensure_retriever

case "$MODE" in
  smoke)
    run_smoke
    ;;
  core)
    if ! is_true "$SKIP_SMOKE"; then
      run_smoke
    fi
    run_core
    ;;
  eval)
    run_eval
    ;;
esac

log "Done. Runs are under: $RUN_ROOT"
