#!/usr/bin/env python3
"""
检查数据规模和训练预算
"""

import pandas as pd
import os

DATA_DIR = "/root/autodl-tmp/data/nq_search"

def main():
    print("=== 数据规模检查 ===\n")

    train_path = os.path.join(DATA_DIR, "train.parquet")
    test_path = os.path.join(DATA_DIR, "test.parquet")

    # 检查文件存在
    if not os.path.exists(train_path):
        print(f"错误: 训练数据不存在: {train_path}")
        return

    if not os.path.exists(test_path):
        print(f"错误: 测试数据不存在: {test_path}")
        return

    # 读取数据
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    print(f"Train 数据:")
    print(f"  行数: {len(train_df)}")
    print(f"  列: {list(train_df.columns)}")

    print(f"\nTest 数据:")
    print(f"  行数: {len(test_df)}")
    print(f"  列: {list(test_df.columns)}")

    # 训练预算计算
    batch_size = 32
    n_agent = 5  # GRPO agent 数

    print(f"\n训练预算 (batch_size={batch_size}, n_agent={n_agent}):")
    print(f"  2 steps (dry-run):    {batch_size * 2:,} prompts, {batch_size * 2 * n_agent:,} GRPO trajectories")
    print(f"  20 steps (smoke):     {batch_size * 20:,} prompts, {batch_size * 20 * n_agent:,} GRPO trajectories")
    print(f"  100 steps (pilot):    {batch_size * 100:,} prompts, {batch_size * 100 * n_agent:,} GRPO trajectories")
    print(f"  300 steps (formal):   {batch_size * 300:,} prompts, {batch_size * 300 * n_agent:,} GRPO trajectories")
    print(f"  400 steps (extended): {batch_size * 400:,} prompts, {batch_size * 400 * n_agent:,} GRPO trajectories")

    # Epoch 计算
    print(f"\nEpoch 计算 (训练数据: {len(train_df)}):")
    for steps in [100, 300, 400]:
        prompts_seen = batch_size * steps
        epochs = prompts_seen / len(train_df)
        print(f"  {steps:3d} steps: {prompts_seen:6,} prompts = {epochs:.2f} epochs")

    # 检查是否足够
    print(f"\n数据充足性检查:")
    if len(train_df) >= 10000:
        print(f"  ✓ 训练数据 >= 10K (当前: {len(train_df)})")
    else:
        print(f"  ⚠ 训练数据 < 10K (当前: {len(train_df)})")

    if len(test_df) >= 300:
        print(f"  ✓ 测试数据 >= 300 (当前: {len(test_df)})")
    else:
        print(f"  ⚠ 测试数据 < 300 (当前: {len(test_df)})")

if __name__ == "__main__":
    main()