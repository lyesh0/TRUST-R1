#!/usr/bin/env bash
# TRUST-R1 4090 训练辅助工具
#
# 提供以下功能：
# - 检查数据规模
# - 检查训练进度
# - 比较 checkpoint 性能
# - 提取关键指标

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="/root/autodl-tmp/data/nq_search"
RUN_ROOT="/root/autodl-tmp/runs"

usage() {
  cat <<'USAGE'
Usage: bash scripts/4090_utils.sh <command> [args]

Commands:
  check-data          检查数据规模和格式
  check-progress      检查训练进度
  extract-metrics     从日志提取关键指标
  compare-ckpts       比较 checkpoint 性能
  analyze-trajectories 分析轨迹日志
  verify-4090-config  验证 4090 配置参数

Examples:
  bash scripts/4090_utils.sh check-data
  bash scripts/4090_utils.sh check-progress --run-dir runs/20241212_120000_B0_dry-run_grpo_seed42
  bash scripts/4090_utils.sh extract-metrics --run-dir runs/20241212_120000_B0_formal_grpo_seed42
  bash scripts/4090_utils.sh compare-ckpts --run-dir runs/20241212_120000_B0_formal_grpo_seed42
  bash scripts/4090_utils.sh verify-4090-config
USAGE
}

# ==================== 检查数据规模 ====================

check_data() {
  echo "=== 数据规模检查 ==="
  echo ""

  local train_path="$DATA_DIR/train.parquet"
  local test_path="$DATA_DIR/test.parquet"

  if [[ ! -f "$train_path" ]]; then
    echo "错误: 训练数据不存在: $train_path" >&2
    exit 1
  fi

  if [[ ! -f "$test_path" ]]; then
    echo "错误: 测试数据不存在: $test_path" >&2
    exit 1
  fi

  python3 - <<'PY'
import pandas as pd
import os

train_path = os.path.expanduser("/root/autodl-tmp/data/nq_search/train.parquet")
test_path = os.path.expanduser("/root/autodl-tmp/data/nq_search/test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

print("Train 数据:")
print(f"  行数: {len(train_df)}")
print(f"  列: {list(train_df.columns)}")

print("\nTest 数据:")
print(f"  行数: {len(test_df)}")
print(f"  列: {list(test_df.columns)}")

# 检查 prompt 格式
if 'prompt' in train_df.columns:
    sample_prompt = train_df.iloc[0]['prompt']
    print(f"\nPrompt 格式示例: {type(sample_prompt)}")

# 训练预算计算
batch_size = 32
steps_100 = 100
steps_300 = 300

print(f"\n训练预算 (batch={batch_size}):")
print(f"  100 steps: ~{batch_size * steps_100:,} prompts, ~{batch_size * steps_100 * 5:,} GRPO trajectories")
print(f"  300 steps: ~{batch_size * steps_300:,} prompts, ~{batch_size * steps_300 * 5:,} GRPO trajectories")
print(f"  完整数据 epoch (10K): {(len(train_df) / (batch_size * steps_100)):.2f} (100 steps), {(len(train_df) / (batch_size * steps_300)):.2f} (300 steps)")
PY
}

# ==================== 检查训练进度 ====================

check_progress() {
  local run_dir="${1:-}"

  if [[ -z "$run_dir" ]]; then
    # 列出最近的运行
    echo "=== 最近的运行 ==="
    echo ""
    for dir in "$RUN_ROOT"/*; do
      if [[ -d "$dir" ]]; then
        local name
        name="$(basename "$dir")"
        local log_file="$dir/train.log"
        if [[ -f "$log_file" ]]; then
            local last_line
            last_line="$(tail -1 "$log_file" 2>/dev/null || true)"
            echo "  $name"
            echo "    最新: ${last_line:0:80}..."
            echo ""
        fi
      fi
    done
    return
  fi

  if [[ ! -d "$run_dir" ]]; then
    echo "错误: 运行目录不存在: $run_dir" >&2
    exit 1
  fi

  echo "=== 训练进度: $run_dir ==="
  echo ""

  local log_file="$run_dir/train.log"

  if [[ ! -f "$log_file" ]]; then
    echo "日志文件不存在: $log_file" >&2
    exit 1
  fi

  # 提取训练信息
  echo "训练信息:"
  grep -E "(Step|Epoch|loss|reward|EM|F1)" "$log_file" | tail -20

  echo ""
  echo "Checkpoints:"
  find "$run_dir/checkpoints" -type f -name "*.pt" -o -name "*.safetensors" 2>/dev/null | head -10 || echo "  未找到 checkpoint"
}

# ==================== 提取关键指标 ====================

extract_metrics() {
  local run_dir="${1:-}"

  if [[ -z "$run_dir" ]]; then
    echo "错误: 需要指定 --run-dir" >&2
    usage
    exit 1
  fi

  if [[ ! -d "$run_dir" ]]; then
    echo "错误: 运行目录不存在: $run_dir" >&2
    exit 1
  fi

  python3 - <<PY
import re
import os
from pathlib import Path

run_dir = Path("$run_dir")
log_file = run_dir / "train.log"

if not log_file.exists():
    print(f"错误: 日志文件不存在: {log_file}")
    exit(1)

with open(log_file, 'r') as f:
    content = f.read()

# 提取关键指标
patterns = {
    'Step:': r'Step:\s*(\d+)',
    'Train loss:': r'Train loss:\s*([\d.]+)',
    'Val EM:': r'Val EM:\s*([\d.]+)',
    'Val F1:': r'Val F1:\s*([\d.]+)',
    'Answer reward:': r'answer_reward:\s*([\d.]+)',
    'Recovery reward:': r'recovery_reward:\s*([\d.]+)',
    'Duplicate penalty:': r'duplicate_penalty:\s*([\d.]+)',
}

results = {}
for name, pattern in patterns.items():
    matches = re.findall(pattern, content)
    if matches:
        results[name] = [float(m) for m in matches]

print("=== 关键指标提取 ===")
print()
print(f"运行目录: {run_dir}")
print()

for name, values in results.items():
    if values:
        print(f"{name:20s}: 最小={min(values):.4f}, 最大={max(values):.4f}, 最后={values[-1]:.4f}")

# 检查是否有 recovery reward
if 'Recovery reward:' in results:
    recovery_vals = results['Recovery reward:']
    non_zero = sum(1 for v in recovery_vals if v > 0)
    print(f"\nRecovery reward 分析:")
    print(f"  非零值数量: {non_zero}/{len(recovery_vals)}")
    print(f"  非零率: {non_zero/len(recovery_vals)*100:.1f}%")
    if non_zero == 0:
        print(f"  警告: Recovery reward 始终为 0，可能存在逻辑问题！")
PY
}

# ==================== 比较 checkpoint 性能 ====================

compare_ckpts() {
  local run_dir="${1:-}"

  if [[ -z "$run_dir" ]]; then
    echo "错误: 需要指定 --run-dir" >&2
    usage
    exit 1
  fi

  echo "=== Checkpoint 性能对比 ==="
  echo "运行目录: $run_dir"
  echo ""

  # 这里假设每个 checkpoint 有对应的评测结果
  # 实际实现需要根据日志格式调整
  python3 - <<PY
import re
import os
from pathlib import Path

run_dir = Path("$run_dir")
log_file = run_dir / "train.log"

if not log_file.exists():
    print(f"错误: 日志文件不存在: {log_file}")
    exit(1)

with open(log_file, 'r') as f:
    content = f.read()

# 提取每个 step 的评测结果
checkpoint_pattern = r'checkpoint:\s*step_(\d+)'
em_pattern = r'Val EM:\s*([\d.]+)'
f1_pattern = r'Val F1:\s*([\d.]+)'

# 简单提取最近的评测结果
recent_evals = re.findall(r'Step:\s*(\d+).*?Val EM:\s*([\d.]+).*?Val F1:\s*([\d.]+)', content, re.DOTALL)

if recent_evals:
    print(f"{'Step':>6}  {'EM':>6}  {'F1':>6}  {'Trend'}")
    print("-" * 30)
    prev_em = None
    for step, em, f1 in recent_evals[-10:]:  # 显示最近 10 个
        trend = ""
        if prev_em:
            change = float(em) - prev_em
            if change > 0.01:
                trend = "↑"
            elif change < -0.01:
                trend = "↓"
            else:
                trend = "→"
        print(f"{int(step):>6}  {float(em):>6.2%}  {float(f1):>6.2%}  {trend}")
        prev_em = float(em)

    # 分析趋势
    if len(recent_evals) >= 2:
        first_em = float(recent_evals[0][1])
        last_em = float(recent_evals[-1][1])
        change = last_em - first_em
        print()
        print(f"整体趋势: {first_em:.2%} → {last_em:.2%} ({change:+.2%})")
else:
    print("未找到评测结果")
PY
}

# ==================== 分析轨迹日志 ====================

analyze_trajectories() {
  local run_dir="${1:-}"

  if [[ -z "$run_dir" ]]; then
    echo "错误: 需要指定 --run-dir" >&2
    usage
    exit 1
  fi

  echo "=== 轨迹分析 ==="
  echo "运行目录: $run_dir"
  echo ""

  local trajectory_dir="$run_dir/trajectories"

  if [[ ! -d "$trajectory_dir" ]]; then
    echo "轨迹目录不存在: $trajectory_dir"
    echo "可能原因:"
    echo "  - trust_logging.enabled=false"
    echo "  - 训练尚未生成轨迹"
    exit 1
  fi

  local count
  count="$(find "$trajectory_dir" -name "*.jsonl" | wc -l)"
  echo "轨迹文件数量: $count"

  if (( count > 0 )); then
    echo ""
    echo "轨迹统计:"
    python3 - <<PY
import json
from pathlib import Path

traj_dir = Path("$trajectory_dir")
total_samples = 0
fault_samples = 0
recovery_samples = 0
duplicate_samples = 0

for jsonl_file in traj_dir.glob("*.jsonl"):
    try:
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                total_samples += 1

                # 统计各种模式
                if data.get('has_fault'):
                    fault_samples += 1
                if data.get('has_recovery'):
                    recovery_samples += 1
                if data.get('has_duplicate'):
                    duplicate_samples += 1
    except:
        pass

print(f"  总样本数: {total_samples:,}")
if total_samples > 0:
    print(f"  故障样本: {fault_samples:,} ({fault_samples/total_samples*100:.1f}%)")
    print(f"  恢复样本: {recovery_samples:,} ({recovery_samples/total_samples*100:.1f}%)")
    print(f"  重复样本: {duplicate_samples:,} ({duplicate_samples/total_samples*100:.1f}%)")
PY
  fi
}

# ==================== 验证 4090 配置 ====================

verify_4090_config() {
  echo "=== 4090 配置验证 ==="
  echo ""

  echo "推荐配置 vs 当前配置:"
  echo ""
  echo "参数              推荐值      说明"
  echo "----------------------------------------"
  echo "train_batch_size  32          ←"
  echo "val_batch_size    20          ←"
  echo "ppo_mini_batch    16          ← (修复 256 问题)"
  echo "ppo_micro_batch   4           ← (修复 64 问题)"
  echo "log_prob_micro    8           ←"
  echo "max_prompt_len    2560        ← (降低显存)"
  echo "max_start_len     1024        ← (降低显存)"
  echo "max_response_len  512         ← (降低显存)"
  echo "max_obs_len       384         ← (降低显存)"
  echo "gpu_mem_util      0.45        ← (4090)"
  echo "save_freq         50          ← (修复 -1 问题)"
  echo "total_epochs      10          ← (修复 1 问题)"
  echo ""
  echo "显存估算 (max_length = max_start + max_response + max_obs * max_turns):"
  local max_len=$((1024 + 512 + 384 * 2))
  echo "  预估 max_length: $max_len tokens"
  echo "  对比旧配置: 3548 tokens → $max_len tokens (降低 $((3548 - max_len)) tokens)"
  echo ""
  echo "训练预算 (batch=32, n_agent=5):"
  echo "  100 steps: 3,200 prompts, 16,000 GRPO trajectories"
  echo "  300 steps: 9,600 prompts, 48,000 GRPO trajectories"
  echo "  400 steps: 12,800 prompts, 64,000 GRPO trajectories"
  echo ""
  echo "分阶段训练流程:"
  echo "  1. check      - 环境检查"
  echo "  2. dry-run    - 2-step 验证"
  echo "  3. smoke      - 20-step 快速测试"
  echo "  4. pilot      - 100-step 预实验"
  echo "  5. formal     - 300-step 正式训练"
}

# ==================== 主入口 ====================

COMMAND="${1:-}"
shift || true

case "$COMMAND" in
  check-data)
    check_data
    ;;
  check-progress)
    check_progress "$@"
    ;;
  extract-metrics)
    extract_metrics "$@"
    ;;
  compare-ckpts)
    compare_ckpts "$@"
    ;;
  analyze-trajectories)
    analyze_trajectories "$@"
    ;;
  verify-4090-config)
    verify_4090_config
    ;;
  "")
    usage
    exit 1
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage
    exit 2
    ;;
esac