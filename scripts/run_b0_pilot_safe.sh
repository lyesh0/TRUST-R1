#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-help}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-3B}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/data/nq_search}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-fs/runs/b0_pilot_safe}"
RETRIEVER_URL="${RETRIEVER_URL:-${SEARCH_URL:-http://127.0.0.1:8000/retrieve}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${NNODES:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-TRUST-R1}"
SEED="${SEED:-42}"
TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-10000}"
VAL_DATA_NUM="${VAL_DATA_NUM:-100}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-20}"
N_AGENT="${N_AGENT:-5}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-4}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-8}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-8}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.1}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
TOPK="${TOPK:-3}"
MAX_START_LENGTH="${MAX_START_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-256}"
MAX_TURNS="${MAX_TURNS:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-10}"
TEST_FREQ="${TEST_FREQ:-2}"
SAVE_FREQ="${SAVE_FREQ:-2}"
VAL_NUM_EXAMINE="${VAL_NUM_EXAMINE:-20}"
TRUST_LOGGING="${TRUST_LOGGING:-false}"
ALLOW_INSTRUCT="${ALLOW_INSTRUCT:-false}"

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_b0_pilot_safe.sh MODE

Clean B0 pilot-safe launcher for diagnosing policy degradation. It always uses
retrieval_fault.enabled=false and trust_reward.enabled=false.

Modes:
  check      Print environment/model/data/batch checks only.
  step0      Run zero-update validation from the base model.
  dry2       Run a 2-step dry run from the base model.
  smoke10    Run the 10-step pilot-safe smoke run from the base model.
  sequence   Run step0, dry2, then smoke10.

Default pilot-safe config:
  MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-3B
  TRAIN_DATA_NUM=10000 VAL_DATA_NUM=100
  TRAIN_BATCH_SIZE=32 VAL_BATCH_SIZE=20 N_AGENT=5
  PPO_MINI_BATCH_SIZE=16 PPO_MICRO_BATCH_SIZE=4
  ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE=8 REF_LOG_PROB_MICRO_BATCH_SIZE=8
  ACTOR_LR=5e-7 LR_WARMUP_RATIO=0.1
  TEMPERATURE=0.8 TOP_P=0.95
  MAX_START_LENGTH=1024 MAX_PROMPT_LENGTH=2560
  MAX_RESPONSE_LENGTH=256 MAX_OBS_LENGTH=384 MAX_TURNS=2
  TOTAL_STEPS=10 SAVE_FREQ=2 TEST_FREQ=2

The script refuses Instruct and step-100 checkpoint paths unless explicitly
overridden for debugging with ALLOW_INSTRUCT=true. Do not use this launcher for
M1/M2 or 100/300-step formal experiments.
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

require_autodl() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "Refusing to run B0 pilot-safe mode on local macOS. Use AutoDL for retriever, rollout, training, and evaluation." >&2
    exit 3
  fi
  if [[ ! -d /root/autodl-tmp ]]; then
    echo "Expected /root/autodl-tmp; this does not look like the configured AutoDL environment." >&2
    exit 3
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found. Run pilot-safe experiments on an AutoDL GPU instance." >&2
    exit 3
  fi
}

require_base_model() {
  if [[ "$MODEL_PATH" == *step_100* || "$MODEL_PATH" == *global_step_100* ]]; then
    echo "Refusing to use degraded step-100 checkpoint: $MODEL_PATH" >&2
    exit 4
  fi
  if [[ "$ALLOW_INSTRUCT" != "true" && "$MODEL_PATH" == *Instruct* ]]; then
    echo "MODEL_PATH appears to be an Instruct checkpoint: $MODEL_PATH" >&2
    echo "For B0 pilot-safe, use the base model, e.g. /root/autodl-tmp/models/Qwen2.5-3B." >&2
    exit 4
  fi
}

assert_divisible() {
  local lhs="$1"
  local rhs="$2"
  local name="$3"
  if (( rhs == 0 || lhs % rhs != 0 )); then
    echo "Config check failed: $name ($lhs % $rhs != 0)" >&2
    exit 5
  fi
}

trainer_stop_step() {
  local target_steps="$1"
  echo $((target_steps + 1))
}

check_batch_config() {
  local world_size=$((GPUS_PER_NODE * NNODES))
  local expanded_rollout_batch=$((TRAIN_BATCH_SIZE * N_AGENT))

  assert_divisible "$TRAIN_BATCH_SIZE" "$world_size" "train_batch_size % world_size"
  assert_divisible "$expanded_rollout_batch" "$world_size" "expanded_rollout_batch % world_size"
  assert_divisible "$PPO_MINI_BATCH_SIZE" "$world_size" "ppo_mini_batch_size % world_size"
  assert_divisible "$PPO_MICRO_BATCH_SIZE" "$world_size" "ppo_micro_batch_size % world_size"
  assert_divisible "$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE" "$world_size" "rollout_log_prob_micro_batch_size % world_size"
  assert_divisible "$REF_LOG_PROB_MICRO_BATCH_SIZE" "$world_size" "ref_log_prob_micro_batch_size % world_size"
  if (( PPO_MINI_BATCH_SIZE < PPO_MICRO_BATCH_SIZE )); then
    echo "Config check failed: ppo_mini_batch_size must be >= ppo_micro_batch_size" >&2
    exit 5
  fi
  assert_divisible "$PPO_MINI_BATCH_SIZE" "$PPO_MICRO_BATCH_SIZE" "ppo_mini_batch_size % ppo_micro_batch_size"
  assert_divisible "$expanded_rollout_batch" "$PPO_MINI_BATCH_SIZE" "expanded_rollout_batch % ppo_mini_batch_size"

  cat <<EOF
world_size=$world_size
train_batch_size=$TRAIN_BATCH_SIZE
n_agent=$N_AGENT
expanded_rollout_batch=$expanded_rollout_batch
val_batch_size=$VAL_BATCH_SIZE
ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE
ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE
rollout_log_prob_micro_batch_size=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE
ref_log_prob_micro_batch_size=$REF_LOG_PROB_MICRO_BATCH_SIZE
per_rank_train_batch_size=$((TRAIN_BATCH_SIZE / world_size))
per_rank_expanded_rollout_batch=$((expanded_rollout_batch / world_size))
expected_fsdp_actor_ppo_mini_batch_size_per_rank=$((PPO_MINI_BATCH_SIZE / world_size))
expected_fsdp_actor_ppo_micro_batch_size_per_rank=$((PPO_MICRO_BATCH_SIZE / world_size))
actor_lr=$ACTOR_LR
lr_warmup_ratio=$LR_WARMUP_RATIO
temperature=$TEMPERATURE
top_p=$TOP_P
max_start_length=$MAX_START_LENGTH
max_prompt_length=$MAX_PROMPT_LENGTH
max_response_length=$MAX_RESPONSE_LENGTH
max_obs_length=$MAX_OBS_LENGTH
max_turns=$MAX_TURNS
target_training_updates=$TOTAL_STEPS
trainer_total_training_steps=$(trainer_stop_step "$TOTAL_STEPS")
save_freq=$SAVE_FREQ
test_freq=$TEST_FREQ
trust_reward.enabled=false
retrieval_fault.enabled=false
EOF
}

print_model_info() {
  MODEL_PATH="$MODEL_PATH" python3 - <<'PY' || true
import os
from pathlib import Path
model_path = Path(os.environ["MODEL_PATH"])
print(f"model_path={model_path}")
try:
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    print(f"model_type={getattr(cfg, 'model_type', None)}")
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    print(f"tokenizer_path={model_path}")
    print(f"chat_template={getattr(tok, 'chat_template', None)}")
except Exception as exc:
    print(f"model/tokenizer info unavailable: {exc}")
PY
}

check_inputs() {
  require_autodl
  require_base_model
  mkdir -p "$OUTPUT_ROOT"
  if [[ "$MODEL_PATH" == /* && ! -e "$MODEL_PATH" ]]; then
    echo "MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 5
  fi
  if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
    echo "Missing train.parquet/test.parquet under DATA_DIR=$DATA_DIR" >&2
    exit 5
  fi
  curl -fsS -X POST "$RETRIEVER_URL" -H 'Content-Type: application/json' \
    -d '{"queries":["health check"],"topk":1,"return_scores":false}' >/dev/null

  log "git commit: $(git rev-parse HEAD 2>/dev/null || echo no-git)"
  if [[ -n "$(git status --porcelain 2>/dev/null || true)" ]]; then
    log "WARNING: working tree has uncommitted changes. Record the diff before reporting final numbers."
    git status --short || true
  fi
  log "MODEL_PATH=$MODEL_PATH"
  log "DATA_DIR=$DATA_DIR"
  log "RETRIEVER_URL=$RETRIEVER_URL"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  print_model_info
  check_batch_config
}

write_preflight() {
  local run_dir="$1"
  {
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo no-git)"
    git status --short 2>/dev/null || true
    print_model_info
    check_batch_config
  } > "$run_dir/preflight.txt"
}

run_ppo() {
  local name="$1"
  shift
  check_inputs

  local run_dir="$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)_$name"
  mkdir -p "$run_dir"
  write_preflight "$run_dir"

  local cmd_file="$run_dir/command.sh"
  cat > "$cmd_file" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '$ROOT_DIR'
export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'
export PYTHONUNBUFFERED=1

# Memory-saving: offload optimizer states and ref model to CPU (required for 24GB GPU)
python3 -m verl.trainer.main_ppo \\
  data.train_files=$DATA_DIR/train.parquet \\
  data.val_files=$DATA_DIR/test.parquet \\
  data.train_data_num=$TRAIN_DATA_NUM \\
  data.val_data_num=$VAL_DATA_NUM \\
  data.train_batch_size=$TRAIN_BATCH_SIZE \\
  data.val_batch_size=$VAL_BATCH_SIZE \\
  data.max_start_length=$MAX_START_LENGTH \\
  data.max_prompt_length=$MAX_PROMPT_LENGTH \\
  data.max_response_length=$MAX_RESPONSE_LENGTH \\
  data.max_obs_length=$MAX_OBS_LENGTH \\
  actor_rollout_ref.model.path=$MODEL_PATH \\
  actor_rollout_ref.model.enable_gradient_checkpointing=true \\
  actor_rollout_ref.model.use_remove_padding=true \\
  actor_rollout_ref.actor.fsdp_config.param_offload=false \\
  actor_rollout_ref.actor.fsdp_config.grad_offload=false \\
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \\
  actor_rollout_ref.ref.fsdp_config.param_offload=true \\
  critic.model.path=$MODEL_PATH \\
  actor_rollout_ref.actor.optim.lr=$ACTOR_LR \\
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$LR_WARMUP_RATIO \\
  actor_rollout_ref.actor.use_kl_loss=true \\
  actor_rollout_ref.actor.kl_loss_coef=0.001 \\
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \\
  actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \\
  actor_rollout_ref.actor.state_masking=true \\
  actor_rollout_ref.rollout.name=vllm \\
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \\
  actor_rollout_ref.rollout.temperature=$TEMPERATURE \\
  actor_rollout_ref.rollout.top_p=$TOP_P \\
  actor_rollout_ref.rollout.n_agent=$N_AGENT \\
  actor_rollout_ref.rollout.log_prob_micro_batch_size=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE \\
  actor_rollout_ref.ref.log_prob_micro_batch_size=$REF_LOG_PROB_MICRO_BATCH_SIZE \\
  algorithm.adv_estimator=grpo \\
  trainer.logger=['console','wandb'] \\
  trainer.project_name=$WANDB_PROJECT \\
  trainer.experiment_name=$name \\
  trainer.n_gpus_per_node=$GPUS_PER_NODE \\
  trainer.nnodes=$NNODES \\
  trainer.total_epochs=1 \\
  trainer.default_hdfs_dir=null \\
  trainer.default_local_dir=$run_dir/checkpoints \\
  +trainer.val_num_examine=$VAL_NUM_EXAMINE \\
  max_turns=$MAX_TURNS \\
  retriever.url=$RETRIEVER_URL \\
  retriever.topk=$TOPK \\
  retrieval_fault.enabled=false \\
  retrieval_fault.mode=clean \\
  retrieval_fault.fault_rate=0.0 \\
  retrieval_fault.seed=$SEED \\
  trust_reward.enabled=false \\
  trust_r1_logging.enabled=$TRUST_LOGGING \\
  trust_r1_logging.output_dir=$run_dir \\
  trust_r1_logging.write_trajectories=false \\
  $* \\
  2>&1 | tee '$run_dir/run.log'
EOF
  chmod +x "$cmd_file"

  log "Running $name; output: $run_dir"
  bash "$cmd_file"
}

step0() {
  run_ppo "b0_clean_base_step0_seed${SEED}" \
    trainer.total_training_steps=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    +trainer.val_before_train=true \
    +trainer.val_only=true
}

dry2() {
  run_ppo "b0_clean_base_dry2_seed${SEED}" \
    trainer.total_training_steps=$(trainer_stop_step 2) \
    trainer.save_freq=2 \
    trainer.test_freq=2 \
    +trainer.val_before_train=true \
    +trainer.val_only=false
}

smoke10() {
  run_ppo "b0_clean_base_smoke10_seed${SEED}" \
    trainer.total_training_steps=$(trainer_stop_step "$TOTAL_STEPS") \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    +trainer.val_before_train=true \
    +trainer.val_only=false
}

case "$MODE" in
  help|-h|--help) usage ;;
  check) check_inputs ;;
  step0|baseline) step0 ;;
  dry2|dry-run) dry2 ;;
  smoke10) smoke10 ;;
  sequence) step0; dry2; smoke10 ;;
  *) echo "Unknown mode: $MODE" >&2; usage; exit 2 ;;
esac
