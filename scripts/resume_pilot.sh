#!/usr/bin/env bash
set -euo pipefail
cd '/root/autodl-tmp/TRUST-R1'

# ============================================================
# Resume B0 pilot from global_step_100
# 原实验: /root/autodl-tmp/runs/20260712_191857_B0_pilot_grpo_seed42
# 原训练崩溃于 step ~150 (磁盘满)，最新完整 checkpoint 为 step_100
# 剩余步数: 300 - 100 = 200
# ============================================================

ORIGINAL_RUN="/root/autodl-tmp/runs/20260712_191857_B0_pilot_grpo_seed42"
RESUME_CKPT="${ORIGINAL_RUN}/checkpoints/actor/global_step_100"
RESUME_RUN_DIR="${ORIGINAL_RUN}_resume"

mkdir -p "$RESUME_RUN_DIR"

export CUDA_VISIBLE_DEVICES='0,1,2,3'

echo "== Resume B0 Pilot =="
echo "  Checkpoint: $RESUME_CKPT"
echo "  Remaining steps: 200"
echo "  Run dir: $RESUME_RUN_DIR"
echo ""

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files=/root/autodl-tmp/data/nq_search/train.parquet \
  data.val_files=/root/autodl-tmp/data/nq_search/test.parquet \
  data.train_data_num=10000 \
  data.val_data_num=300 \
  data.train_batch_size=32 \
  data.val_batch_size=20 \
  data.max_prompt_length=2560 \
  data.max_response_length=512 \
  data.max_start_length=1024 \
  data.max_obs_length=384 \
  actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen2.5-3B-Instruct \
  +actor_rollout_ref.model.resume_path="$RESUME_CKPT" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  critic.model.path=/root/autodl-tmp/models/Qwen2.5-3B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size=4 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.grad_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
  trainer.logger=['console','wandb'] \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.total_epochs=10 \
  trainer.total_training_steps=200 \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${RESUME_RUN_DIR}/checkpoints" \
  trainer.project_name=TRUST-R1-4090 \
  trainer.experiment_name="$(basename "$RESUME_RUN_DIR")" \
  max_turns=2 \
  retriever.url=http://127.0.0.1:8000/retrieve \
  retriever.topk=3 \
  trust_r1_logging.enabled=true \
  trust_r1_logging.output_dir="$RESUME_RUN_DIR" \
  trust_r1_logging.write_trajectories=true \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.n_agent=5 \
  retrieval_fault.enabled=false \
  retrieval_fault.mode=clean \
  retrieval_fault.fault_rate=0.0 \
  trust_reward.enabled=false \
  2>&1 | tee "${RESUME_RUN_DIR}/pilot_resume.log"
