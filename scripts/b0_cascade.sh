#!/usr/bin/env bash
# B0 Cascade: pilot → 100-step → 300-step
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Activate trustr1 conda env
source /root/miniconda3/etc/profile.d/conda.sh
conda activate trustr1

# --- config: keep identical to B0 pilot-safe defaults ---
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-3B}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/data/nq_search}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-fs/runs/b0_cascade}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
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
# With four GPUs: 160 Actor trajectories, local mini=8, local micro=2, accumulation=4.
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-16}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-16}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
CLIP_RATIO="${CLIP_RATIO:-0.2}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-bfloat16}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
TOPK="${TOPK:-3}"
MAX_START_LENGTH="${MAX_START_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2560}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-384}"
MAX_TURNS="${MAX_TURNS:-2}"
TEST_FREQ="${TEST_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
VAL_NUM_EXAMINE="${VAL_NUM_EXAMINE:-20}"
TRUST_LOGGING="${TRUST_LOGGING:-false}"
B0_100_SAVE_FREQ="${B0_100_SAVE_FREQ:-20}"
B0_100_TEST_FREQ="${B0_100_TEST_FREQ:-20}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

trainer_stop_step() { echo $(($1 + 1)); }

# --- common PPO invocation ---
run_ppo() {
  local name="$1"
  local total_steps="$2"
  local model_path="$3"
  shift 3

  local run_dir="$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)_$name"
  mkdir -p "$run_dir"

  {
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo no-git)"
    echo "name=$name"
    echo "total_steps=$total_steps"
    echo "model_path=$model_path"
    git status --short 2>/dev/null || true
    df -h "$OUTPUT_ROOT" "$model_path" || true
  } > "$run_dir/preflight.txt"

  log "Launching $name (${total_steps} steps, model=$model_path) → $run_dir"

  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
  export PYTHONUNBUFFERED=1

  python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=$TRAIN_DATA_NUM \
    data.val_data_num=$VAL_DATA_NUM \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_start_length=$MAX_START_LENGTH \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.max_obs_length=$MAX_OBS_LENGTH \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    critic.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$LR_WARMUP_RATIO \
    actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=$ROLLOUT_DTYPE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.n_agent=$N_AGENT \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$REF_LOG_PROB_MICRO_BATCH_SIZE \
    algorithm.adv_estimator=grpo \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$name \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$(trainer_stop_step "$total_steps") \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$run_dir/checkpoints \
    +trainer.val_before_train=true \
    +trainer.val_only=false \
    +trainer.val_num_examine=$VAL_NUM_EXAMINE \
    max_turns=$MAX_TURNS \
    retriever.url=$RETRIEVER_URL \
    retriever.topk=$TOPK \
    retrieval_fault.enabled=false \
    retrieval_fault.mode=clean \
    retrieval_fault.fault_rate=0.0 \
    retrieval_fault.seed=$SEED \
    trust_reward.enabled=false \
    trust_r1_logging.enabled=$TRUST_LOGGING \
    trust_r1_logging.output_dir=$run_dir \
    trust_r1_logging.write_trajectories=false \
    "$@" \
    2>&1 | tee "$run_dir/run.log"

  local exit_code=${PIPESTATUS[0]}
  if [[ $exit_code -eq 0 ]]; then
    log "SUCCESS: $name completed"
    echo "SUCCESS" > "$run_dir/COMPLETED"
  else
    log "FAILED: $name (exit code $exit_code)"
    echo "FAILED exit=$exit_code" > "$run_dir/COMPLETED"
    exit $exit_code
  fi
}

# --- find latest step-N checkpoint ---
find_checkpoint() {
  local dir="$1"
  local step="$2"
  # Look for global_step_<N> directory in the checkpoint tree
  find "$dir" -type d -name "global_step_${step}" 2>/dev/null | head -1
}

# ==============================
# Phase 2: B0 100-step
# ==============================
run_100() {
  log "=== B0 100-step training ==="
  run_ppo "b0_clean_100step_seed${SEED}" 100 "$MODEL_PATH" \
    trainer.save_freq=$B0_100_SAVE_FREQ \
    trainer.test_freq=$B0_100_TEST_FREQ
}

# ==============================
# Phase 3: B0 300-step (from step-100 ckpt)
# ==============================
run_300() {
  local ckpt100_dir
  # Find the most recent B0 100-step run
  ckpt100_dir=$(ls -dt "$OUTPUT_ROOT"/*b0_clean_100step*/checkpoints 2>/dev/null | head -1)
  if [[ -z "$ckpt100_dir" ]]; then
    log "ERROR: no 100-step checkpoint found under $OUTPUT_ROOT"
    exit 6
  fi

  local ckpt_path
  ckpt_path=$(find_checkpoint "$ckpt100_dir" 100)
  if [[ -z "$ckpt_path" ]]; then
    log "ERROR: global_step_100 not found in $ckpt100_dir"
    log "Available checkpoints:"
    find "$ckpt100_dir" -type d -name "global_step_*" 2>/dev/null || echo "(none)"
    exit 6
  fi

  log "=== B0 300-step from checkpoint: $ckpt_path ==="
  run_ppo "b0_clean_300step_seed${SEED}" 300 "$ckpt_path" \
    trainer.save_freq=100 \
    trainer.test_freq=20
}

# ==============================
# Entry point
# ==============================
MODE="${1:-help}"

case "$MODE" in
  100|100step) run_100 ;;
  300|300step) run_300 ;;
  cascade) run_100 && run_300 ;;
  *) echo "Usage: $0 {100|300|cascade}" >&2; exit 2 ;;
esac
