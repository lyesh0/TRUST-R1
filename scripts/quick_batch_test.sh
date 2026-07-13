#!/usr/bin/env bash
# 快速批大小测试（非交互式）

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 从命令行获取要测试的 batch size
BATCH_SIZE="${1:-32}"

echo "=== 测试 batch_size = $BATCH_SIZE ==="
echo ""

# 计算 mini/micro batch
local mini=$((BATCH_SIZE / 2))
local micro=$((mini / 4))

echo "配置:"
echo "  train_batch_size: $BATCH_SIZE"
echo "  val_batch_size: $((BATCH_SIZE * 3 / 4))"
echo "  ppo_mini_batch: $((BATCH_SIZE / 2))"
echo "  ppo_micro_batch: $((BATCH_SIZE / 8))"
echo ""

log_file="/tmp/batch_${BATCH_SIZE}_dryrun_$(date +%Y%m%d_%H%M%S).log"

echo "运行 2-step dry-run..."
echo ""

if bash scripts/train_4090.sh \
    --stage dry-run \
    --experiment B0 \
    --train-batch-size "$BATCH_SIZE" \
    --steps 2 \
    2>&1 | tee "$log_file"; then

  echo ""
  echo "=== 检查结果 ==="

  if grep -qi "out of memory\|CUDA out of memory\|OOM" "$log_file"; then
    echo "✗ batch=$BATCH_SIZE: OOM!"
    echo "  建议尝试更小的 batch_size"
    exit 1
  else
    echo "✓ batch=$BATCH_SIZE: 通过！"
    echo ""
    echo "1 epoch 步数计算 (79K 数据):"
    steps_per_epoch=$((79168 / BATCH_SIZE))
    echo "  batch=$BATCH_SIZE → 1 epoch ≈ $steps_per_epoch steps"
    echo ""
    echo "建议配置:"
    echo "  --train-batch-size $BATCH_SIZE"
    echo "  --steps $steps_per_epoch  # 1 epoch"
    echo "  --steps $((steps_per_epoch * 2))  # 2 epochs"
    exit 0
  fi
else
  echo ""
  echo "✗ batch=$BATCH_SIZE: 运行失败（非 OOM 错误）"
  echo "  检查日志: $log_file"
  exit 1
fi