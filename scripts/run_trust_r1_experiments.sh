#!/usr/bin/env bash
set -euo pipefail

STAGE="check"
SUITE=""
EXPERIMENT="B0"
EXPERIMENTS="B0"
ALGO="grpo"
MODEL="/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
DATA_DIR="/root/autodl-tmp/data/nq_search"
RUN_ROOT="/root/autodl-tmp/runs"
RETRIEVER_URL="http://127.0.0.1:8000/retrieve"
FAULT_MODE="mixed"
FAULT_RATE="0.2"
SEED="42"
DRY_RUN_STEPS="2"
TOTAL_STEPS="100"
WANDB_PROJECT="TRUST-R1"
GPUS_PER_NODE="4"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NNODES="1"
MAX_TURNS="2"
TOPK="3"
FULL_DATA_MIN_GB="150"
TRUST_LOGGING="false"

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_trust_r1_experiments.sh [options]

Stages:
  --stage check|prepare-data|launch-retriever|dry-run|train|eval|suite

Common options:
  --suite smoke|core|eval
  --experiment B0|B1|M1|M2
  --experiments B0,B1,M1,M2
  --algo grpo|ppo
  --model PATH_OR_HF_ID
  --data-dir PATH
  --run-root PATH
  --retriever-url URL
  --fault-mode clean|empty|drop_top|duplicate|mixed
  --fault-rate FLOAT
  --seed INT
  --dry-run-steps INT
  --total-steps INT
  --gpus-per-node INT
  --cuda-visible-devices IDS
  --nnodes INT
  --max-turns INT
  --topk INT
  --trust-logging true|false

Examples:
  bash scripts/run_trust_r1_experiments.sh --stage check
  bash scripts/run_trust_r1_experiments.sh --stage dry-run --experiment B0
  bash scripts/run_trust_r1_experiments.sh --suite core --experiments B0,B1,M1,M2
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --suite) SUITE="$2"; STAGE="suite"; shift 2 ;;
    --experiment) EXPERIMENT="$2"; shift 2 ;;
    --experiments) EXPERIMENTS="$2"; shift 2 ;;
    --algo) ALGO="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --retriever-url) RETRIEVER_URL="$2"; shift 2 ;;
    --fault-mode) FAULT_MODE="$2"; shift 2 ;;
    --fault-rate) FAULT_RATE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --dry-run-steps) DRY_RUN_STEPS="$2"; shift 2 ;;
    --total-steps) TOTAL_STEPS="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --nnodes) NNODES="$2"; shift 2 ;;
    --max-turns) MAX_TURNS="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --trust-logging) TRUST_LOGGING="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

is_macos() { [[ "$(uname -s)" == "Darwin" ]]; }
is_heavy_stage() { [[ "$STAGE" != "check" ]]; }

require_autodl_for_heavy() {
  if is_macos && is_heavy_stage; then
    echo "Refusing to run heavy stage '$STAGE' on local macOS. Use AutoDL for data/model/retriever/training/eval." >&2
    exit 3
  fi
}

available_gb() {
  local path="$1"
  df -Pk "$path" | awk 'NR==2 {printf "%d", $4/1024/1024}'
}

check_full_data_disk() {
  local base="/root/autodl-tmp"
  [[ -d "$base" ]] || return 0
  local free_gb
  free_gb="$(available_gb "$base")"
  if (( free_gb < FULL_DATA_MIN_GB )); then
    echo "Full wiki-18 index/corpus needs much more disk. Available ${free_gb}GB < required ${FULL_DATA_MIN_GB}GB." >&2
    exit 4
  fi
}

check_stage() {
  echo "== TRUST-R1 check =="
  echo "root: $ROOT_DIR"
  echo "os: $(uname -a)"
  echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo no-git)"
  echo "python: $(command -v python3 || true)"
  python3 - <<'PY'
import importlib.util
mods = ["torch", "transformers", "ray", "datasets", "trust_r1"]
for mod in mods:
    spec = importlib.util.find_spec(mod)
    print(f"{mod}: {'ok' if spec else 'missing'}")
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  else
    echo "nvidia-smi: not found"
  fi
}

prepare_data_stage() {
  require_autodl_for_heavy
  check_full_data_disk
  mkdir -p "$DATA_DIR"
  python3 scripts/data_process/nq_search.py --local_dir "$DATA_DIR"
}

launch_retriever_stage() {
  require_autodl_for_heavy
  echo "Use retrieval_launch.sh after setting corpus/index paths, or start an external retriever at $RETRIEVER_URL."
  echo "This script currently performs a health check instead of guessing your index paths."
  curl -fsS "$RETRIEVER_URL" >/dev/null || true
}

experiment_overrides() {
  local exp="$1"
  case "$exp" in
    B0)
      echo "retrieval_fault.enabled=false retrieval_fault.mode=clean retrieval_fault.fault_rate=0.0 trust_reward.enabled=false"
      ;;
    B1)
      echo "retrieval_fault.enabled=true retrieval_fault.mode=$FAULT_MODE retrieval_fault.fault_rate=$FAULT_RATE retrieval_fault.seed=$SEED trust_reward.enabled=false"
      ;;
    M1)
      echo "retrieval_fault.enabled=true retrieval_fault.mode=$FAULT_MODE retrieval_fault.fault_rate=$FAULT_RATE retrieval_fault.seed=$SEED trust_reward.enabled=true trust_reward.duplicate_penalty_weight=0.0"
      ;;
    M2)
      echo "retrieval_fault.enabled=true retrieval_fault.mode=$FAULT_MODE retrieval_fault.fault_rate=$FAULT_RATE retrieval_fault.seed=$SEED trust_reward.enabled=true"
      ;;
    *) echo "Unknown experiment: $exp" >&2; exit 2 ;;
  esac
}

algo_overrides() {
  case "$ALGO" in
    grpo)
      echo "algorithm.adv_estimator=grpo actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.rollout.n_agent=5"
      ;;
    ppo)
      echo "algorithm.adv_estimator=gae actor_rollout_ref.rollout.n_agent=1 actor_rollout_ref.actor.use_kl_loss=false"
      ;;
    *) echo "Unknown algo: $ALGO" >&2; exit 2 ;;
  esac
}

build_common_overrides() {
  local run_dir="$1"
  local steps="$2"
  cat <<EOF
data.train_files=$DATA_DIR/train.parquet
data.val_files=$DATA_DIR/test.parquet
data.train_batch_size=32
data.val_batch_size=32
data.max_prompt_length=4096
data.max_response_length=500
data.max_start_length=2048
data.max_obs_length=500
actor_rollout_ref.model.path=$MODEL
critic.model.path=$MODEL
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
actor_rollout_ref.rollout.temperature=1
actor_rollout_ref.actor.state_masking=true
trainer.logger=['console']
trainer.n_gpus_per_node=$GPUS_PER_NODE
trainer.nnodes=$NNODES
trainer.save_freq=-1
trainer.test_freq=50
trainer.project_name=$WANDB_PROJECT
trainer.experiment_name=$(basename "$run_dir")
trainer.total_epochs=1
trainer.total_training_steps=$steps
trainer.default_hdfs_dir=null
trainer.default_local_dir=$run_dir/checkpoints
max_turns=$MAX_TURNS
retriever.url=$RETRIEVER_URL
retriever.topk=$TOPK
trust_r1_logging.enabled=$TRUST_LOGGING
trust_r1_logging.output_dir=$run_dir
trust_r1_logging.write_trajectories=$TRUST_LOGGING
EOF
}

run_one() {
  local exp="$1"
  local mode="$2"
  require_autodl_for_heavy

  if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
    echo "Missing parquet data under $DATA_DIR. Expected train.parquet and test.parquet." >&2
    exit 5
  fi

  local steps="$TOTAL_STEPS"
  if [[ "$mode" == "dry-run" ]]; then
    steps="$DRY_RUN_STEPS"
  fi

  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)_${exp}_${mode}_${ALGO}_seed${SEED}"
  local run_dir="$RUN_ROOT/$run_id"
  mkdir -p "$run_dir"

  git rev-parse HEAD > "$run_dir/git_commit.txt" 2>/dev/null || true
  env | sort > "$run_dir/env.txt"

  local overrides_file="$run_dir/overrides.txt"
  {
    build_common_overrides "$run_dir" "$steps"
    algo_overrides
    experiment_overrides "$exp"
    if [[ "$mode" == "eval" ]]; then
      echo "+trainer.val_only=true"
      echo "+trainer.val_before_train=true"
    else
      echo "+trainer.val_only=false"
      echo "+trainer.val_before_train=true"
    fi
  } | tr ' ' '\n' | awk 'NF' > "$overrides_file"

  local cmd_file="$run_dir/command.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd '$ROOT_DIR'"
    echo "export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
    echo "PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \\"
    sed "s/^/  /; s/$/ \\/" "$overrides_file"
    echo "  2>&1 | tee '$run_dir/${mode}.log'"
  } > "$cmd_file"
  chmod +x "$cmd_file"

  echo "Running $exp $mode. Run dir: $run_dir"
  bash "$cmd_file"
}

suite_stage() {
  local suite_name="${SUITE:-core}"
  case "$suite_name" in
    smoke)
      run_one B0 dry-run
      ;;
    core)
      IFS=',' read -ra exps <<< "$EXPERIMENTS"
      for exp in "${exps[@]}"; do
        run_one "$exp" train
      done
      ;;
    eval)
      IFS=',' read -ra exps <<< "$EXPERIMENTS"
      for exp in "${exps[@]}"; do
        run_one "$exp" eval
      done
      ;;
    *) echo "Unknown suite: $suite_name" >&2; exit 2 ;;
  esac
}

case "$STAGE" in
  check) check_stage ;;
  prepare-data) prepare_data_stage ;;
  launch-retriever) launch_retriever_stage ;;
  dry-run) run_one "$EXPERIMENT" dry-run ;;
  train) run_one "$EXPERIMENT" train ;;
  eval) run_one "$EXPERIMENT" eval ;;
  suite) suite_stage ;;
  *) echo "Unknown stage: $STAGE" >&2; usage; exit 2 ;;
esac
