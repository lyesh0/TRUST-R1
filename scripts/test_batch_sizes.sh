#!/usr/bin/env bash
# 批大小测试脚本
# 用于测试不同 batch_size 是否会导致 OOM

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== 批大小测试 ==="
echo ""
echo "测试流程："
echo "  1. batch=32, 2-step dry-run"
echo "  2. batch=64, 2-step dry-run (如果 32 通过)"
echo "  3. batch=128, 2-step dry-run (如果 64 通过)"
echo ""
echo "按 Enter 继续，或 Ctrl+C 取消"
read -r

BATCH_SIZES=(32 64 128)
RESULT_FILE="/tmp/batch_test_results_$(date +%Y%m%d_%H%M%S).txt"

echo "批次大小测试结果" > "$RESULT_FILE"
echo "时间: $(date)" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

for batch in "${BATCH_SIZES[@]}"; do
  echo ""
  echo "=========================================="
  echo "测试 batch_size = $batch"
  echo "=========================================="
  echo ""

  # 计算 mini/micro batch（train_4090.sh 会自动计算）
  # mini=$((batch / 2))
  # micro=$((mini / 4))

  log_file="/tmp/batch_${batch}_dryrun.log"

  if bash scripts/train_4090.sh \
      --stage dry-run \
      --experiment B0 \
      --train-batch-size "$batch" \
      --steps 2 \
      2>&1 | tee "$log_file"; then

    # 检查是否有 OOM
    if grep -qi "out of memory\|CUDA out of memory\|OOM" "$log_file"; then
      echo "✗ batch=$batch: OOM!" | tee -a "$RESULT_FILE"
      echo "  最大可用 batch_size: $((batch / 2)) 或更小"
      break
    else
      echo "✓ batch=$batch: 通过" | tee -a "$RESULT_FILE"

      # 计算 1 epoch 需要的步数
      local steps_per_epoch=$((79168 / batch))
      echo "  1 epoch ≈ $steps_per_epoch steps" | tee -a "$RESULT_FILE"
    fi
  else
    echo "✗ batch=$batch: 运行失败（非 OOM 错误）" | tee -a "$RESULT_FILE"
    break
  fi
done

echo ""
echo "=========================================="
echo "测试完成，结果保存到: $RESULT_FILE"
echo "=========================================="

# 显示结果摘要
echo ""
echo "=== 结果摘要 ==="
cat "$RESULT_FILE"

# 给出建议
echo ""
echo "=== 建议 ==="

if grep -q "batch=128: 通过" "$RESULT_FILE"; then
  echo "✓ 可以使用 batch=128"
  echo "  formal steps 建议: 600-1200 (1-2 epochs)"
elif grep -q "batch=64: 通过" "$RESULT_FILE"; then
  echo "✓ 可以使用 batch=64"
  echo "  formal steps 建议: 1200-2400 (1-2 epochs)"
elif grep -q "batch=32: 通过" "$RESULT_FILE"; then
  echo "✓ 只能使用 batch=32"
  echo "  formal steps 建议: 2400+ (至少 1 epoch)"
  echo "  或者考虑减小模型/启用更多 offload"
else
  echo "✗ 所有 batch 都测试失败，需要进一步调优配置"
fi