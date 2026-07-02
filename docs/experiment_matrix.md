# TRUST-R1 实验矩阵

本文件记录 TRUST-R1 首版实验设计。所有实验数字必须来自真实日志。

## 1. 实验阶段

```text
Layer 0: 项目初始化与本地准备
Layer 1: 原 Search-R1 clean baseline 跑通
Layer 2: noisy retrieval 脆弱性诊断
Layer 3: fault augmentation baseline
Layer 4: recovery reward 方法
Layer 5: duplicate penalty / TRUST-R1 full
Layer 6: noisy suite + report
```

## 2. 核心实验组

| ID | 名称 | 检索环境 | Recovery Reward | Duplicate Penalty | 目的 |
|---|---|---|---:|---:|---|
| B0 | Search-R1 Clean Baseline | clean | 否 | 否 | 原始能力基线 |
| B1 | Fault Augmentation | noisy training | 否 | 否 | 判断仅故障增强是否足够 |
| M1 | Recovery Reward | noisy training | 是 | 否 | 验证恢复奖励贡献 |
| M2 | TRUST-R1 Full | noisy training | 是 | 是 | 完整方法 |

## 3. 故障类型

首版训练故障：

```text
empty
duplicate
drop_top / answer_drop
mixed
```

评测可扩展：

```text
clean
seen-mixed 10%
seen-mixed 20%
seen-mixed 30%
unseen hard-negative
unseen timeout
```

## 4. 对照原则

B0/B1/M1/M2 尽量保持一致：

- 模型；
- 数据；
- 检索器；
- 训练步数；
- batch；
- rollout_n；
- max_tool_calls；
- seed 策略；
- 评测 split。

除目标变量外，不随意改变其他设置。

## 5. 指标

任务指标：

- Exact Match / Accuracy；
- token F1（如适用）；
- clean-to-noisy performance drop。

工具行为指标：

- legal tool-call rate；
- search trigger rate；
- average search calls；
- query rewrite rate；
- duplicate query rate；
- premature answer rate；
- invalid action rate。

鲁棒性指标：

- first-failure recovery rate；
- fault-hit accuracy；
- robustness drop；
- answer-bearing passage recall。

## 6. 最低可交付标准

- Search-R1 baseline 跑通；
- 三类故障环境可控；
- 证明 baseline 在 noisy retrieval 下退化；
- TRUST-R1 至少在一个 noisy setting 上提升；
- 有失败分类和真实轨迹案例。

## 7. 预算优先级

预算不足时优先保留：

1. B0 clean baseline；
2. M2 TRUST-R1 full；
3. B0/M2 第二 seed；
4. B1/M1 消融；
5. unseen fault 扩展。

不要为了实验数量牺牲核心对照质量。
