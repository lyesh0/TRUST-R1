#!/usr/bin/env bash
# TRUST-R1 4×RTX 4090 专用训练脚本
#
# 基于 4×24GB RTX 4090 配置优化，解决以下问题：
# 1. batch/micro batch 不匹配
# 2. save_freq=-1 导致无法保存 checkpoint
# 3. total_epochs=1 可能提前结束
# 4. 长度配置浪费显存
#
# 分阶段训练流程：
# - 阶段 1: 2-step dry run（验证配置、显存、retriever）
# - 阶段 2: 20-step smoke test（B0, M2）
# - 阶段 3: 100-step pilot（B0, M2）
# - 阶段 4: 正式 300-step 核心矩阵（B0, B1, M1, M2）

set -euo pipefail

# ==================== 默认配置（针对 4×RTX 4090 优化） ====================

# 训练配置
STAGE="check"                        # check | dry-run | smoke | pilot | formal
EXPERIMENT="B0"                      # B0 | B1 | M1 | M2
EXPERIMENTS="B0,B1,M1,M2"            # 核心矩阵
ALGO="grpo"                          # grpo | ppo

# 模型与数据
MODEL="/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
DATA_DIR="/root/autodl-tmp/data/nq_search"
RUN_ROOT="/root/autodl-tmp/runs"
RETRIEVER_URL="http://127.0.0.1:8000/retrieve"

# 4×RTX 4090 GPU 配置
GPUS_PER_NODE="4"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NNODES="1"

# 训练参数（基于 79K 数据，batch=64，0.5 epoch ≈ 617 steps）
DRY_RUN_STEPS="${DRY_RUN_STEPS:-2}"
SMOKE_STEPS="${SMOKE_STEPS:-20}"
PILOT_STEPS="${PILOT_STEPS:-300}"      # 0.25 epoch
FORMAL_STEPS="${FORMAL_STEPS:-600}"    # 0.5 epoch
EXTENDED_STEPS="${EXTENDED_STEPS:-1200}"  # 1 epoch

# 数据规模
TRAIN_DATA_NUM="10000"
VAL_DATA_NUM="300"

# 批大小配置（先测试 batch=32 是否 OOM）
# 79K 数据，batch=32 → 1 epoch ≈ 2,474 steps
# 79K 数据，batch=64 → 1 epoch ≈ 1,237 steps
# 79K 数据，batch=128 → 1 epoch ≈ 617 steps
# 默认先用 batch=32 测试，通过后再试 batch=64
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-20}"

# PPO 批大小配置（保持 mini:micro ≈ 4:1，根据 train_batch 自动调整）
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-4}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"

# 长度配置（降低显存压力）
MAX_PROMPT_LENGTH="2560"
MAX_START_LENGTH="1024"
MAX_RESPONSE_LENGTH="512"
MAX_OBS_LENGTH="384"
MAX_TURNS="2"

# 检索与故障配置
TOPK="3"
FAULT_MODE="mixed"
FAULT_RATE="0.2"
SEED="42"

# Reward 配置
RECOVERY_WEIGHT="0.2"
DUPLICATE_PENALTY_WEIGHT="0.1"

# vLLM 配置（为 4090 降低显存占用）
VLLM_GPU_MEMORY_UTILIZATION="0.45"

# 学习率配置
ACTOR_LR="1e-6"
LR_WARMUP_RATIO="0.03"

# Checkpoint 与评测频率
SAVE_FREQ="50"
TEST_FREQ="50"
TOTAL_EPOCHS="10"  # 防止提前结束

# 日志配置
WANDB_PROJECT="TRUST-R1-4090"
TRUST_LOGGING="true"  # 默认开启轨迹日志

# ==================== 帮助信息 ====================

usage() {
  cat <<'USAGE'
Usage: bash scripts/train_4090.sh [options]

Stages (按顺序运行):
  --stage check         环境检查（GPU、数据、retriever）
  --stage dry-run        2-step 验证（检查配置、显存、是否 OOM）
  --stage smoke          20-step 快速测试（B0, M2）
  --stage pilot          300-step 预实验（B0, M2）
  --stage formal         正式训练（B0, B1, M1, M2）
  --stage batch-scaling  自动 batch scaling 测试（32→64→128）

Batch 配置 (基于 79K 数据):
  batch=32  → 1 epoch ≈ 2,474 steps
  batch=64  → 1 epoch ≈ 1,237 steps
  batch=128 → 1 epoch ≈ 617 steps

Options:
  --train-batch-size 32|64|128   覆盖默认 batch size (默认 32)
  --steps INT                    覆盖默认步数

Experiment variants:
  B0  - Clean baseline（无故障，无 trust reward）
  B1  - Fault augmentation（有故障，无 trust reward）
  M1  - Recovery reward only（有故障，recovery reward，无 penalty）
  M2  - Full TRUST-R1（有故障，recovery reward + duplicate penalty）

Options:
  --experiment B0|B1|M1|M2         单个实验
  --experiments B0,B1,M1,M2       多个实验（用于 suite）
  --steps INT                     覆盖默认步数
  --model PATH                    模型路径
  --data-dir PATH                 数据目录
  --run-root PATH                 运行根目录
  --retriever-url URL             Retriever URL
  --fault-mode clean|empty|drop_top|duplicate|mixed
  --fault-rate FLOAT              故障率
  --seed INT                      随机种子
  --trust-logging true|false      开启轨迹日志
  --save-freq INT                 Checkpoint 保存频率
  --test-freq INT                 评测频率
  --gpu-mem-util FLOAT            vLLM 显存利用率 (0.0-1.0)

Examples:
  # 环境检查
  bash scripts/train_4090.sh --stage check

  # 2-step dry run
  bash scripts/train_4090.sh --stage dry-run --experiment B0

  # 20-step smoke test
  bash scripts/train_4090.sh --stage smoke

  # 100-step pilot
  bash scripts/train_4090.sh --stage pilot

  # 300-step 正式训练
  bash scripts/train_4090.sh --stage formal

  # 自动 batch scaling 测试（32→64→128）
  bash scripts/train_4090.sh --stage batch-scaling --experiment B0

  # 自定义步数
  bash scripts/train_4090.sh --stage dry-run --experiment M2 --steps 5

Configuration (内置 4090 优化):
  - train_batch_size: 32
  - ppo_mini_batch_size: 16 (修复 256 问题)
  - ppo_micro_batch_size: 4  (修复 64 问题)
  - log_prob_micro_batch_size: 8
  - max_prompt_length: 2560   (降低显存)
  - max_start_length: 1024
  - max_response_length: 512
  - max_obs_length: 384
  - gpu_memory_utilization: 0.45
  - save_freq: 50             (修复 -1 问题)
  - total_epochs: 10          (修复 1 问题)
USAGE
}

# ==================== 参数解析 ====================

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
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
    --steps) CUSTOM_STEPS="$2"; shift 2 ;;
    --train-batch-size)
      TRAIN_BATCH_SIZE="$2"
      VAL_BATCH_SIZE="$((TRAIN_BATCH_SIZE * 3 / 4))"
      PPO_MINI_BATCH_SIZE="$((TRAIN_BATCH_SIZE / 2))"
      PPO_MICRO_BATCH_SIZE="$((PPO_MINI_BATCH_SIZE / 4))"
      LOG_PROB_MICRO_BATCH_SIZE="$((PPO_MINI_BATCH_SIZE / 2))"
      shift 2
      ;;
    --gpu-mem-util) VLLM_GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --save-freq) SAVE_FREQ="$2"; shift 2 ;;
    --test-freq) TEST_FREQ="$2"; shift 2 ;;
    --trust-logging) TRUST_LOGGING="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ==================== 工具函数 ====================

# 检查是否在 AutoDL
is_autodl() { [[ -d "/root/autodl-tmp" ]]; }

# 检查 GPU 可用性
check_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "错误: nvidia-smi 未找到，GPU 不可用" >&2
    return 1
  fi

  echo "== GPU 状态 =="
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | nl

  local gpu_count
  gpu_count=$(nvidia-smi --list-gpus | wc -l)
  if (( gpu_count < 4 )); then
    echo "警告: 检测到 ${gpu_count} 个 GPU，但建议使用 4×4090" >&2
  fi
}

# 检查数据文件
check_data() {
  echo "== 数据检查 =="
  local missing=0

  if [[ ! -f "$DATA_DIR/train.parquet" ]]; then
    echo "错误: 缺少 $DATA_DIR/train.parquet" >&2
    missing=1
  fi

  if [[ ! -f "$DATA_DIR/test.parquet" ]]; then
    echo "错误: 缺少 $DATA_DIR/test.parquet" >&2
    missing=1
  fi

  if (( missing )); then
    echo "" >&2
    echo "准备数据:" >&2
    echo "  python3 scripts/data_process/nq_search.py --local_dir $DATA_DIR" >&2
    exit 1
  fi

  # 检查实际数据量
  python3 - <<PY
import pandas as pd

train_df = pd.read_parquet("$DATA_DIR/train.parquet")
test_df = pd.read_parquet("$DATA_DIR/test.parquet")

print(f"  train.parquet: {len(train_df)} rows")
print(f"  test.parquet: {len(test_df)} rows")

if len(train_df) < $TRAIN_DATA_NUM:
    print(f"  警告: 训练数据量 {len(train_df)} < 配置的 $TRAIN_DATA_NUM")
PY
}

# 检查 Retriever
check_retriever() {
  echo "== Retriever 检查 =="
  if curl -fsS "$RETRIEVER_URL" >/dev/null 2>&1; then
    echo "  ✓ Retriever 可用: $RETRIEVER_URL"
  else
    echo "  ✗ Retriever 不可用: $RETRIEVER_URL" >&2
    echo "" >&2
    echo "启动 Retriever (需要在 retriever conda 环境中):" >&2
    echo "  conda activate retriever" >&2
    echo "  bash scripts/launch_retriever.sh" >&2
    exit 1
  fi
}

# 检查磁盘空间
check_disk() {
  echo "== 磁盘空间检查 =="
  df -h "$RUN_ROOT" | tail -1 | awk '{print "  可用: " $4}'
}

# 计算阶段对应的步数
get_steps_for_stage() {
  case "$STAGE" in
    dry-run) echo "${CUSTOM_STEPS:-$DRY_RUN_STEPS}" ;;
    smoke) echo "${CUSTOM_STEPS:-$SMOKE_STEPS}" ;;
    pilot) echo "${CUSTOM_STEPS:-$PILOT_STEPS}" ;;
    formal) echo "${CUSTOM_STEPS:-$FORMAL_STEPS}" ;;
    *) echo "${CUSTOM_STEPS:-$TOTAL_STEPS}" ;;
  esac
}

# 获取阶段对应的实验列表
get_experiments_for_stage() {
  case "$STAGE" in
    dry-run) echo "$EXPERIMENT" ;;
    smoke) echo "B0,M2" ;;
    pilot) echo "B0,M2" ;;
    formal) echo "$EXPERIMENTS" ;;
    *) echo "$EXPERIMENT" ;;
  esac
}

# ==================== 实验配置生成 ====================

# 生成实验特定的 override
experiment_overrides() {
  local exp="$1"
  case "$exp" in
    B0)
      cat <<EOF
retrieval_fault.enabled=false
retrieval_fault.mode=clean
retrieval_fault.fault_rate=0.0
trust_reward.enabled=false
EOF
      ;;
    B1)
      cat <<EOF
retrieval_fault.enabled=true
retrieval_fault.mode=$FAULT_MODE
retrieval_fault.fault_rate=$FAULT_RATE
retrieval_fault.seed=$SEED
trust_reward.enabled=false
EOF
      ;;
    M1)
      cat <<EOF
retrieval_fault.enabled=true
retrieval_fault.mode=$FAULT_MODE
retrieval_fault.fault_rate=$FAULT_RATE
retrieval_fault.seed=$SEED
trust_reward.enabled=true
trust_reward.recovery_weight=$RECOVERY_WEIGHT
trust_reward.duplicate_penalty_weight=0.0
EOF
      ;;
    M2)
      cat <<EOF
retrieval_fault.enabled=true
retrieval_fault.mode=$FAULT_MODE
retrieval_fault.fault_rate=$FAULT_RATE
retrieval_fault.seed=$SEED
trust_reward.enabled=true
trust_reward.recovery_weight=$RECOVERY_WEIGHT
trust_reward.duplicate_penalty_weight=$DUPLICATE_PENALTY_WEIGHT
EOF
      ;;
    *) echo "Unknown experiment: $exp" >&2; exit 2 ;;
  esac
}

# 生成算法特定 override
algo_overrides() {
  case "$ALGO" in
    grpo)
      cat <<EOF
algorithm.adv_estimator=grpo
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
actor_rollout_ref.rollout.n_agent=5
EOF
      ;;
    ppo)
      cat <<EOF
algorithm.adv_estimator=gae
actor_rollout_ref.rollout.n_agent=1
actor_rollout_ref.actor.use_kl_loss=false
EOF
      ;;
    *) echo "Unknown algo: $ALGO" >&2; exit 2 ;;
  esac
}

# 构建完整的 override 配置（针对 4090 优化）
build_overrides_4090() {
  local run_dir="$1"
  local steps="$2"

  # 计算 prompt 数量和 GRPO 轨迹数
  local prompt_count=$((TRAIN_BATCH_SIZE * steps))
  local grpo_trajectories=$((prompt_count * 5))

  cat <<EOF
# ==================== 数据配置 ====================
data.train_files=$DATA_DIR/train.parquet
data.val_files=$DATA_DIR/test.parquet
data.train_data_num=$TRAIN_DATA_NUM
data.val_data_num=$VAL_DATA_NUM
data.train_batch_size=$TRAIN_BATCH_SIZE
data.val_batch_size=$VAL_BATCH_SIZE

# 长度配置（针对 4090 优化，降低显存压力）
data.max_prompt_length=$MAX_PROMPT_LENGTH
data.max_response_length=$MAX_RESPONSE_LENGTH
data.max_start_length=$MAX_START_LENGTH
data.max_obs_length=$MAX_OBS_LENGTH

# ==================== 模型配置 ====================
actor_rollout_ref.model.path=$MODEL
actor_rollout_ref.model.enable_gradient_checkpointing=true
actor_rollout_ref.model.use_remove_padding=true
critic.model.path=$MODEL

# ==================== Actor 优化配置 ====================
actor_rollout_ref.actor.optim.lr=$ACTOR_LR
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$LR_WARMUP_RATIO
actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE
actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE
actor_rollout_ref.actor.ppo_epochs=1

# FSDP 配置（针对 4090 优化）
actor_rollout_ref.actor.fsdp_config.param_offload=false
actor_rollout_ref.actor.fsdp_config.grad_offload=false
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
actor_rollout_ref.ref.fsdp_config.param_offload=true

# ==================== Rollout 配置 ====================
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION
actor_rollout_ref.rollout.temperature=1
actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE
actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE

# ==================== 训练配置 ====================
trainer.logger=['console','wandb']
trainer.n_gpus_per_node=$GPUS_PER_NODE
trainer.nnodes=$NNODES
trainer.save_freq=$SAVE_FREQ
trainer.test_freq=$TEST_FREQ
trainer.total_epochs=$TOTAL_EPOCHS
trainer.total_training_steps=$steps
trainer.default_hdfs_dir=null
trainer.default_local_dir=$run_dir/checkpoints
trainer.project_name=$WANDB_PROJECT
trainer.experiment_name=\$(basename "$run_dir")

# ==================== 实验配置 ====================
max_turns=$MAX_TURNS
retriever.url=$RETRIEVER_URL
retriever.topk=$TOPK

# ==================== 日志配置 ====================
trust_r1_logging.enabled=$TRUST_LOGGING
trust_r1_logging.output_dir=$run_dir
trust_r1_logging.write_trajectories=$TRUST_LOGGING
# trust_r1_logging.sample_rate=0.01  # Not in config struct
EOF
}

# ==================== 单个实验运行 ====================

run_one() {
  local exp="$1"
  local mode="$2"

  if ! is_autodl; then
    echo "错误: 非 AutoDL 环境，请使用 AutoDL 运行训练" >&2
    exit 1
  fi

  # 检查数据
  if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
    echo "错误: 缺少数据文件: $DATA_DIR/train.parquet 或 test.parquet" >&2
    exit 1
  fi

  # 获取步数
  local steps
  steps="$(get_steps_for_stage)"

  # 生成运行 ID
  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)_${exp}_${mode}_${ALGO}_seed${SEED}"
  local run_dir="$RUN_ROOT/$run_id"
  mkdir -p "$run_dir"

  # 记录环境信息
  git rev-parse HEAD > "$run_dir/git_commit.txt" 2>/dev/null || echo "no-git" > "$run_dir/git_commit.txt"
  env | sort > "$run_dir/env.txt"

  # 生成 override 文件
  local overrides_file="$run_dir/overrides.txt"
  {
    build_overrides_4090 "$run_dir" "$steps"
    echo ""
    algo_overrides
    echo ""
    experiment_overrides "$exp"
  } > "$overrides_file"

  # 生成运行脚本
  local cmd_file="$run_dir/run.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd '$ROOT_DIR'"
    echo "export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
    echo ""
    echo "echo \"== 4090 优化配置摘要 ==\""
    echo "echo \"  Experiment: $exp\""
    echo "echo \"  Steps: $steps\""
    echo "echo \"  Batch size: $TRAIN_BATCH_SIZE\""
    echo "echo \"  Mini batch: $PPO_MINI_BATCH_SIZE\""
    echo "echo \"  Micro batch: $PPO_MICRO_BATCH_SIZE\""
    echo "echo \"  GRPO n_agent: 5\""
    echo "echo \"  Expected prompts: $((TRAIN_BATCH_SIZE * steps))\""
    echo "echo \"  Expected GRPO trajectories: $((TRAIN_BATCH_SIZE * steps * 5))\""
    echo "echo \"  Max length: ~$((MAX_START_LENGTH + MAX_RESPONSE_LENGTH + MAX_OBS_LENGTH * MAX_TURNS)) tokens\""
    echo "echo \"\""
    echo ""
    echo "PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \\"
    grep -v '^[[:space:]]*#' "$overrides_file" | grep -v '^$' | sed 's/^/  /; s/$/ \\/'
    echo "  2>&1 | tee '$run_dir/${mode}.log'"
  } > "$cmd_file"
  chmod +x "$cmd_file"

  echo ""
  echo "=========================================="
  echo "  运行实验: $exp ($mode)"
  echo "  运行目录: $run_dir"
  echo "  步数: $steps"
  echo "=========================================="
  echo ""

  bash "$cmd_file"
}

# ==================== 阶段运行 ====================

run_suite() {
  local experiments
  experiments="$(get_experiments_for_stage)"

  echo "== $STAGE 阶段 =="
  echo "  实验: $experiments"
  echo "  步数: $(get_steps_for_stage)"
  echo ""

  IFS=',' read -ra exps <<< "$experiments"
  for exp in "${exps[@]}"; do
    run_one "$exp" "$STAGE"
  done

  echo ""
  echo "== $STAGE 阶段完成 =="
  echo "  已运行实验: $experiments"
  echo "  结果目录: $RUN_ROOT"
}

# ==================== 主入口 ====================

case "$STAGE" in
  check)
    echo "=== TRUST-R1 4090 环境检查 ==="
    check_gpu
    check_data
    check_retriever
    check_disk
    echo ""
    echo "=== 检查通过，可以开始训练 ==="
    echo ""
    echo "建议的训练顺序："
    echo "  1. bash scripts/train_4090.sh --stage dry-run --experiment B0"
    echo "  2. bash scripts/train_4090.sh --stage smoke"
    echo "  3. bash scripts/train_4090.sh --stage pilot"
    echo "  4. bash scripts/train_4090.sh --stage formal"
    ;;

  batch-scaling)
    # 自动 batch scaling 测试（32 → 64 → 128）
    local batches=(32 64 128)
    local max_ok_batch=0
    local current_stage_orig="$STAGE"
    local exp="${EXPERIMENT:-B0}"

    echo "=== Batch Scaling 测试 ==="
    echo "测试配置: 2-step dry-run, 实验: $exp"
    echo ""

    for batch in "${batches[@]}"; do
      echo "----------------------------------------"
      echo "测试 batch size: $batch"
      echo "----------------------------------------"

      # 临时设置 batch size
      TRAIN_BATCH_SIZE="$batch"
      VAL_BATCH_SIZE="$((batch * 3 / 4))"
      PPO_MINI_BATCH_SIZE="$((batch / 2))"
      PPO_MICRO_BATCH_SIZE="$((PPO_MINI_BATCH_SIZE / 4))"
      LOG_PROB_MICRO_BATCH_SIZE="$((PPO_MINI_BATCH_SIZE / 2))"

      # 临时设置 STAGE 为 dry-run
      STAGE="dry-run"

      # 运行测试
      if run_one "$exp" "batch-scaling_${batch}"; then
        max_ok_batch="$batch"
        echo "✓ Batch size $batch 测试通过"
      else
        echo "✗ Batch size $batch 测试失败（可能是 OOM）"
        break
      fi

      echo ""
    done

    echo "=========================================="
    echo "Batch Scaling 测试完成"
    echo "=========================================="
    echo "最大可用 batch size: $max_ok_batch"
    echo ""

    if [[ "$max_ok_batch" -gt 0 ]]; then
      echo "建议配置："
      echo "  --train-batch-size $max_ok_batch"
      echo ""
      echo "预期训练步数（基于 10K 数据）："
      echo "  batch=$max_ok_batch → 1 epoch ≈ $((10000 / max_ok_batch)) steps"
      echo ""
    else
      echo "警告: 所有 batch size 测试均失败，请检查配置"
      exit 1
    fi
    ;;

  dry-run|smoke|pilot|formal)
    run_suite
    ;;

  *)
    echo "Unknown stage: $STAGE" >&2
    usage
    exit 2
    ;;
esac