# TRUST-R1 4×RTX 4090 训练指南

本文档说明如何使用 `scripts/train_4090.sh` 进行针对 4×RTX 4090 的训练。

## 概述

`train_4090.sh` 是专门为 4×RTX 4090 (24GB VRAM) 优化的训练脚本，解决了原 `run_trust_r1_experiments.sh` 中的配置问题：

| 问题 | 原配置 | 修复后 |
|------|--------|--------|
| batch/micro batch 不匹配 | mini=256, micro=64 | mini=16, micro=4 |
| 无法保存 checkpoint | save_freq=-1 | save_freq=50 |
| 可能提前结束 | total_epochs=1 | total_epochs=10 |
| 显存浪费 | max_prompt=4096, max_start=2048 | max_prompt=2560, max_start=1024 |

## 快速开始

### 1. 环境检查

```bash
bash scripts/train_4090.sh --stage check
```

这将检查：
- GPU 状态（需要 4×4090）
- 数据文件是否存在
- Retriever 是否可用
- 磁盘空间

### 2. 2-step Dry Run

```bash
bash scripts/train_4090.sh --stage dry-run --experiment B0
```

**目的：** 验证配置、检查是否 OOM、确保所有组件正常工作。

**验收标准：**
- 四张 GPU 都被使用
- 不出现 NCCL、Ray、OOM 错误
- Retriever 正常返回
- Reward 能计算
- State masking 没有报错

### 3. 20-step Smoke Test

```bash
bash scripts/train_4090.sh --stage smoke
```

这将运行 B0 和 M2 各 20 steps。

**验收标准：**
- Parser error 低于 ~5%
- 搜索调用率没有坍缩到 0
- B0 的 answer reward 不是全部相同
- M2 确实出现 fault
- M2 的 recovery reward 至少偶尔非零
- Duplicate penalty 没有完全压制搜索

### 4. 100-step Pilot

```bash
bash scripts/train_4090.sh --stage pilot
```

这将运行 B0 和 M2 各 100 steps。

**目的：** 回答三个问题：
1. B0 是否确实在学习？
2. M2 是否没有学成"少搜索/不搜索"？
3. M2 在 noisy dev 上是否出现初步改善趋势？

### 5. 300-step Formal Training

```bash
bash scripts/train_4090.sh --stage formal
```

这将运行核心矩阵 B0, B1, M1, M2 各 300 steps。

**延长条件：** 如果 200→300 step 的 noisy EM/F1、recovery rate 仍明显提高，可以考虑续跑到 400 steps。

## 配置参数详解

### 批大小配置

```bash
TRAIN_BATCH_SIZE=32          # 全局训练批大小
VAL_BATCH_SIZE=20            # 验证批大小
PPO_MINI_BATCH_SIZE=16       # PPO mini batch (修复原 256)
PPO_MICRO_BATCH_SIZE=4       # PPO micro batch (修复原 64)
LOG_PROB_MICRO_BATCH_SIZE=8  # Log prob 计算
```

**理由：**
- 原 `mini_batch=256` 与 `train_batch=32` 不匹配
- 新配置保持 `mini:micro ≈ 4:1` 比例
- `micro=4` 确保每张卡瞬时负载足够小

### 长度配置

```bash
MAX_PROMPT_LENGTH=2560       # 最大 prompt 长度 (原 4096)
MAX_START_LENGTH=1024        # 最大 start 长度 (原 2048)
MAX_RESPONSE_LENGTH=512      # 最大响应长度 (原 500)
MAX_OBS_LENGTH=384           # 最大观察长度 (原 500)
MAX_TURNS=2                  # 最大轮次
```

**显存估算：**
```
原配置: 2048 + 500 + 500×2 = 3548 tokens
新配置: 1024 + 512 + 384×2 = 2304 tokens (降低 35%)
```

### vLLM 配置

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.45  # vLLM 显存利用率
```

**显存不足时的调整顺序：**
1. `0.45 → 0.40`
2. `ppo_micro_batch_size: 4 → 2`
3. `log_prob_micro_batch_size: 8 → 4`
4. `max_obs_length: 384 → 256`
5. `max_response_length: 512 → 384`
6. 最后才开启 `param_offload=true`

### Checkpoint 配置

```bash
SAVE_FREQ=50                 # 每 50 steps 保存
TEST_FREQ=50                 # 每 50 steps 评测
TOTAL_EPOCHS=10              # 防止提前结束
```

**保留的 checkpoint：**
- step 50
- step 100
- step 150
- step 200
- step 250
- step 300

**目的：** 防止训练中断，并比较模型在哪一步达到最好结果。

### 学习率配置

```bash
ACTOR_LR=1e-6                # 学习率
LR_WARMUP_RATIO=0.03         # Warmup 比例
```

这些参数与 Search-R1 原始实验一致。

## 实验矩阵

| 实验 | 故障注入 | Trust Reward | 描述 |
|------|----------|--------------|------|
| B0 | ❌ | ❌ | Clean baseline |
| B1 | ✅ | ❌ | Fault augmentation |
| M1 | ✅ Recovery only | 验证 recovery reward 效果 |
| M2 | ✅ Recovery + Penalty | 完整 TRUST-R1 |

## 辅助工具

### 检查数据规模

```bash
bash scripts/4090_utils.sh check-data
```

### 检查训练进度

```bash
# 列出最近的运行
bash scripts/4090_utils.sh check-progress

# 查看特定运行
bash scripts/4090_utils.sh check-progress --run-dir runs/20241212_120000_B0_formal_grpo_seed42
```

### 提取关键指标

```bash
bash scripts/4090_utils.sh extract-metrics --run-dir runs/...
```

输出包括：
- Train loss
- Val EM/F1
- Answer reward
- Recovery reward
- Duplicate penalty

### 比较 checkpoint 性能

```bash
bash scripts/4090_utils.sh compare-ckpts --run-dir runs/...
```

### 分析轨迹日志

```bash
bash scripts/4090_utils.sh analyze-trajectories --run-dir runs/...
```

### 验证 4090 配置

```bash
bash scripts/4090_utils.sh verify-4090-config
```

## 训练预算

| 配置 | Steps | Prompts | GRPO 轨迹 (n=5) |
|------|-------|---------|-----------------|
| dry-run | 2 | 64 | 320 |
| smoke | 20 | 640 | 3,200 |
| pilot | 100 | 3,200 | 16,000 |
| formal | 300 | 9,600 | 48,000 |
| 延长 | 400 | 12,800 | 64,000 |

**对 10K 数据的 epoch 数：**
- 100 steps: ~0.32 epoch
- 300 steps: ~0.96 epoch
- 400 steps: ~1.28 epoch

## 常见问题

### Q: 为什么不直接跑 1005 steps？

**A:**
1. 显存限制：4×4090 难以支持大 batch 的大步数训练
2. 预算合理：300 steps 已经接近 1 epoch，足够观察趋势
3. 可控性：分阶段训练可以及时发现和修复问题

### Q: 为什么不跑 7B 模型？

**A:**
4×4090 难以支持 7B full-weight GRPO。需要切换到 7B LoRA 或继续使用 3B full-weight。

### Q: Recovery reward 始终为 0 怎么办？

**A:**
当前 recovery reward 非常严格：必须同时满足"发生故障、改变 query、恢复证据、最终答对"。如果 20–50 steps 后仍始终为 0，应先排查轨迹逻辑。

### Q: Duplicate penalty 只是 exact match 吗？

**A:**
是的，当前只识别完全重复的 query（大小写不敏感）。第一版可以照常跑，但报告应写成"exact duplicate query penalty"，而不是"semantic redundancy penalty"。

### Q: 如何判断是否应该延长到 400 steps？

**A:**
检查 200→300 step 的趋势：
- Noisy EM/F1 是否持续提升
- Recovery rate 是否持续提升
- 如果是，可以统一补到 400（注意保持对照公平性）

## 预期输出

### Dry Run (2 steps)

```
== GPU 状态 ==
    1  NVIDIA GeForce RTX 4090, 24564 MiB
    2  NVIDIA GeForce RTX 4090, 24564 MiB
    3  NVIDIA GeForce RTX 4090, 24564 MiB
    4  NVIDIA GeForce RTX 4090, 24564 MiB

== 4090 优化配置摘要 ==
  Experiment: B0
  Steps: 2
  Batch size: 32
  Mini batch: 16
  Micro batch: 4
  GRPO n_agent: 5
  Expected prompts: 64
  Expected GRPO trajectories: 320
  Max length: ~2304 tokens
```

### Formal Training (300 steps)

预计生成以下 checkpoint：
- `checkpoints/global_step50/`
- `checkpoints/global_step100/`
- `checkpoints/global_step150/`
- `checkpoints/global_step200/`
- `checkpoints/global_step250/`
- `checkpoints/global_step300/`

以及对应的评测结果。

## 参考文献

- Search-R1 论文: https://ar5iv.org/html/2503.09516v5
- TRUST-R1 实验方案: `TRUST-R1_完整实验方案_v1.0_资源冻结版.md`