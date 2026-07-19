#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
EXPERIMENT="${2:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE1_ROOT="${STAGE1_ROOT:-/root/autodl-tmp/TRUST-R1-stage1}"
DATA_DIR="${DATA_DIR:-$STAGE1_ROOT/data}"
C0_PATH="${C0_PATH:-$STAGE1_ROOT/checkpoints/C0}"
MANIFEST="${MANIFEST:-$ROOT_DIR/artifacts/stage1/data_manifest.json}"
C0_SELECTION="${C0_SELECTION:-$ROOT_DIR/artifacts/stage1/c0_selection.json}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
SEED=42

usage() {
  echo "Usage: bash scripts/run_stage1_local_reward.sh smoke|train|eval S1-B0|S1-B1" >&2
  exit 2
}

[[ "$MODE" == "smoke" || "$MODE" == "train" || "$MODE" == "eval" ]] || usage
[[ "$EXPERIMENT" == "S1-B0" || "$EXPERIMENT" == "S1-B1" ]] || usage

cd "$ROOT_DIR"
if [[ "$(uname -s)" == "Darwin" || ! -d /root/autodl-tmp ]]; then
  echo "Stage1 rollout, training, and evaluation must run on AutoDL, not the local Mac." >&2
  exit 3
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing formal Stage1 execution from a dirty worktree." >&2
  exit 4
fi
if [[ "$C0_PATH" == *Instruct* ]]; then
  echo "Stage1 requires the merged Qwen2.5-3B Base cold-start model, not Instruct." >&2
  exit 4
fi
[[ -f "$MANIFEST" ]] || { echo "Missing Stage1 manifest: $MANIFEST" >&2; exit 4; }
[[ -f "$C0_SELECTION" ]] || { echo "Missing committed C0 selection artifact: $C0_SELECTION" >&2; exit 4; }
[[ -f "$C0_PATH/config.json" ]] || { echo "C0 is not a complete Hugging Face model: $C0_PATH" >&2; exit 4; }
if ! compgen -G "$C0_PATH/model*.safetensors*" >/dev/null && \
   ! compgen -G "$C0_PATH/pytorch_model*.bin*" >/dev/null; then
  echo "C0 merged model weights are missing: $C0_PATH" >&2
  exit 4
fi
if grep -Rqi "instruct" "$C0_PATH/config.json" "$C0_PATH/tokenizer_config.json" 2>/dev/null; then
  echo "C0 metadata contains an Instruct model marker." >&2
  exit 4
fi
python3 scripts/data_process/build_stage1_data.py verify --manifest "$MANIFEST" --data-dir "$DATA_DIR"

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
[[ "$GPU_COUNT" == "4" ]] || { echo "Stage1 requires exactly 4 visible GPUs; found $GPU_COUNT." >&2; exit 5; }

if [[ "$MODE" == "eval" ]]; then
  MODEL_PATH="${EVAL_MODEL_PATH:?Set EVAL_MODEL_PATH explicitly to C0 or a step25/50/75/100 checkpoint}"
  TOTAL_STEPS=1
  SAVE_FREQ=-1
  TEST_FREQ=-1
  VAL_ONLY=true
elif [[ "$MODE" == "smoke" ]]; then
  MODEL_PATH="$C0_PATH"
  TOTAL_STEPS=3
  SAVE_FREQ=-1
  TEST_FREQ=1
  VAL_ONLY=false
else
  MODEL_PATH="$C0_PATH"
  TOTAL_STEPS=101
  SAVE_FREQ=25
  TEST_FREQ=25
  VAL_ONLY=false
fi

PROCESS_ENABLED=false
[[ "$EXPERIMENT" == "S1-B1" ]] && PROCESS_ENABLED=true
COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="$(git rev-parse --short HEAD)"
RUN_ID="${EXPERIMENT}-seed${SEED}-${SHORT_COMMIT}"
RUN_DIR="$STAGE1_ROOT/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

{
  echo "run_id=$RUN_ID"
  echo "git_commit=$COMMIT"
  echo "mode=$MODE"
  echo "experiment=$EXPERIMENT"
  echo "model_path=$MODEL_PATH"
  echo "train_batch_size=32"
  echo "actor_trajectory_batch=128"
  echo "global_mini_batch=32"
  echo "global_micro_batch=8"
  echo "per_gpu_mini_batch=8"
  echo "per_gpu_micro_batch=2"
  echo "gradient_accumulation=4"
  echo "optimizer_steps_per_trainer_step=4"
  echo "process_reward_enabled=$PROCESS_ENABLED"
  echo "total_training_steps=$TOTAL_STEPS"
  echo "data.train_files=$DATA_DIR/rl_train.parquet"
  echo "data.val_files=$DATA_DIR/rl_validation.parquet"
  echo "data.shuffle_train_dataloader=false"
  echo "actor_rollout_ref.actor.optim.lr=2e-7"
  echo "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03"
  echo "actor_rollout_ref.actor.grad_clip=1.0"
  echo "actor_rollout_ref.actor.clip_ratio=0.2"
  echo "actor_rollout_ref.actor.entropy_coeff=0.001"
  echo "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  echo "actor_rollout_ref.actor.kl_loss_coef=0.001"
  echo "actor_rollout_ref.actor.ppo_epochs=1"
  echo "actor_rollout_ref.actor.abort_on_non_finite=true"
  echo "actor_rollout_ref.rollout.seed=42"
  echo "actor_rollout_ref.rollout.temperature=1.0"
  echo "actor_rollout_ref.rollout.top_p=0.95"
  echo "actor_rollout_ref.rollout.dtype=bfloat16"
  echo "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  echo "actor_rollout_ref.rollout.gpu_memory_utilization=0.45"
  echo "process_reward.weight=0.2"
  echo "process_reward.z_clip=2.0"
  echo "process_reward.max_search_steps=2"
  echo "process_reward.abort_on_alignment_error=true"
  echo "max_turns=2"
  echo "retriever.url=$RETRIEVER_URL"
  echo "retriever.topk=3"
  echo "trainer.save_freq=$SAVE_FREQ"
  echo "trainer.test_freq=$TEST_FREQ"
  python3 -m pip show torch transformers peft ray vllm 2>/dev/null || true
  df -h
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} > "$RUN_DIR/preflight.txt"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export PYTHONUNBUFFERED=1

python3 -m verl.trainer.main_ppo \
  data.train_files="$DATA_DIR/rl_train.parquet" \
  data.val_files="$DATA_DIR/rl_validation.parquet" \
  data.train_data_num=null \
  data.val_data_num=null \
  data.train_batch_size=32 \
  data.val_batch_size=100 \
  data.max_prompt_length=2560 \
  data.max_start_length=1024 \
  data.max_response_length=512 \
  data.max_obs_length=384 \
  data.shuffle_train_dataloader=false \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=2e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.state_masking=true \
  actor_rollout_ref.actor.abort_on_non_finite=true \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.grad_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.n_agent=4 \
  actor_rollout_ref.rollout.seed=42 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
  algorithm.adv_estimator=grpo \
  algorithm.no_think_rl=false \
  retriever.url="$RETRIEVER_URL" \
  retriever.topk=3 \
  retrieval_fault.enabled=false \
  trust_reward.enabled=false \
  process_reward.enabled="$PROCESS_ENABLED" \
  process_reward.compute_diagnostics=true \
  process_reward.weight=0.2 \
  process_reward.z_clip=2.0 \
  process_reward.max_search_steps=2 \
  process_reward.abort_on_alignment_error=true \
  trust_r1_logging.enabled=true \
  trust_r1_logging.write_trajectories=true \
  trust_r1_logging.sample_limit_per_call=32 \
  trust_r1_logging.output_dir="$RUN_DIR" \
  max_turns=2 \
  trainer.seed=42 \
  trainer.logger="['console','wandb']" \
  trainer.project_name=TRUST-R1 \
  trainer.experiment_name="$RUN_ID" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="$RUN_DIR/checkpoints" \
  +trainer.val_before_train=true \
  +trainer.val_only="$VAL_ONLY" \
  2>&1 | tee "$RUN_DIR/run.log"
