# TRUST-R1：面向不可靠检索工具的鲁棒搜索智能体强化学习

> **版本**：v1.0（资源与路线冻结版）  
> **项目周期**：10～14 天  
> **总预算上限**：1000 元  
> **Baseline**：Search-R1  
> **训练框架**：verl  
> **首版范围**：单检索工具，不做 multi-tool  
> **训练方式**：full-weight GRPO  

---

# 0. 已冻结的关键决策

## 0.1 硬件选择

**首选 4×A800 80GB，不选 4×RTX 4090 作为正式 full-weight 实验平台。**

本方案默认 A800 为 80GB 版本。如果实际是 40GB 版本，需要重新评估训练配置。

原因：

- 7B/8B full-weight GRPO 同时包含 actor、reference、optimizer state、rollout KV cache 和多条采样轨迹，显存压力远高于普通 SFT；
- Search-R1 还是多轮工具交互任务，response 和 observation 会进一步增加 rollout 显存；
- 4090 单卡只有 24GB，虽然 4 卡总显存为 96GB，但 full-weight GRPO 不能简单按总显存相加理解；
- 4090 多卡通信、参数分片、CPU offload 和长序列 rollout 的工程成本会明显增加，不符合 1～2 周交付目标；
- A800 80GB 更适合 FSDP full-shard 与 vLLM/SGLang rollout。

## 0.2 主模型选择

### 正式主模型

**`Qwen/Qwen2.5-7B`（Base）**

不是 3B，也不在第一版追求更新的 Qwen3/Qwen3.5。

选择理由：

1. Search-R1 原始代码、训练模板和公开结果直接覆盖 Qwen2.5-7B，复现风险最低；
2. 7B Base 已被 Search-R1 验证可以通过 RL 学会多轮搜索与推理；
3. 7B 容量足以承担 query rewrite、失败恢复和证据整合；
4. Base 模型保留更明显的训练前后行为差异，有利于证明能力来自本项目，而不是原有 instruction/tool-use 对齐；
5. 在 1000 元和 10～14 天约束下，同时引入新模型架构、框架迁移、检索索引重建和新算法，风险过高。

### 开发模型

**`Qwen/Qwen2.5-3B` 或 `Qwen/Qwen2.5-3B-Instruct`**

只用于：

- parser 与 reward 单测；
- fault wrapper 调试；
- 10～20 step smoke test；
- 检查 loss mask、checkpoint 和检索服务；
- 不用于最终方法结论。

### 兜底模型

**`Qwen/Qwen2.5-7B-Instruct`**

仅在 7B Base 经过以下处理后仍不能稳定进入工具调用格式时使用：

1. 增加低权重 format reward；
2. 做 500～1000 条 format-only SFT；
3. 20～30 step 训练后合法工具调用率仍低于 70%。

### 可选扩展模型

**`Qwen/Qwen3-8B-Base`** 只作为第二阶段跨模型验证，不阻塞第一版。

只有在以下条件全部满足时才尝试：

- Search-R1 baseline 和 TRUST-R1 主结果已经完成；
- 剩余预算足够；
- 最新 verl + rollout engine 的 Qwen3 链路已通过 20-step smoke test；
- 不需要再修改检索器或数据格式。

## 0.3 代码路线

不建议长期依赖 Search-R1 仓库内嵌的旧版 verl 代码。

推荐路线：

- **算法与评测定义**：严格参考 Search-R1；
- **训练底座**：使用一个固定 commit 的最新版 verl；
- **多轮交互**：使用 verl 的 AgentLoop / Search Tool Integration；
- **检索服务**：复用 Search-R1 的 E5 + FAISS 输入输出格式；
- **baseline reward**：保持 Search-R1 的 outcome-only EM reward；
- **新方法**：在 baseline 上添加可靠性环境和恢复奖励。

这样得到的是：

> 基于现代 verl 实现的 Search-R1 baseline，而不是对旧仓库进行大规模版本修补。

如果最新版 verl 的 Search Tool 示例无法在 1 天内跑通，则立即回退原 Search-R1 仓库支持良好的 Qwen2.5 路线，不继续折腾 Qwen3。

## 0.4 检索部署

**默认采用 CPU FAISS，把 4 张 A800 全部留给 full-weight GRPO。**

原因：原 61GB GPU Flat index 已丢失，需要重新下载或重建；如果再让检索器独占一张 A800，只剩 3 张卡训练，吞吐和并行配置都会更麻烦。

建议机器最低条件：

- 内存：128GB 起，推荐 256GB；
- 数据盘：至少 200GB 可用，推荐 300GB；
- 本地 NVMe；
- CPU 核数尽可能多。

CPU 检索能否接受，不凭感觉决定，必须先做基准测试。

---

# 1. 项目题目与核心贡献

## 1.1 项目名称

# TRUST-R1

**Tool-Reliability-aware Utility and Self-recovery Training for Search Agents**

中文：

> 面向检索工具可靠性感知与自主恢复的搜索智能体强化学习框架

## 1.2 研究问题

Search-R1 默认在相对稳定的检索环境中学习“何时搜索、搜索什么、何时回答”。真实工具可能出现：

- 空返回；
- 返回上一轮重复文档；
- 返回相关但缺少答案的文档；
- 返回高相似但错误的 hard negative；
- 超时；
- 检索结果质量随时间波动。

本项目研究：

> 当搜索工具出现随机故障时，模型能否识别搜索没有带来新信息，并通过改写 query、继续检索或及时停止完成任务？

## 1.3 首版方法只保留三个核心模块

为了确保两周内可交付，第一版不堆叠复杂模块。

### 模块 A：Tool Fault Injection

训练和测试中可控注入：

1. `empty`：返回空结果；
2. `duplicate`：返回上一轮文档；
3. `answer_drop`：移除包含 gold answer alias 的结果；
4. `hard_negative`：只用于 unseen-fault 测试；
5. `timeout`：只用于 unseen-fault 测试。

### 模块 B：Recovery-Conditioned Reward

不是奖励“多搜索一次”，而是奖励：

> 遭遇故障后，模型进行了非机械重复的 query reformulation，后续取得有效证据，并最终答对。

### 模块 C：Redundancy Regularization

轻量惩罚：

- 高相似重复 query；
- 连续返回高度重叠文档；
- 已获得 answer-bearing evidence 后继续无效搜索。

课程训练作为可选增强，不作为必须完成项。

---

# 2. 假设

## H1：7B Base 能学会工具交互

Qwen2.5-7B-Base 在正确 format prompt 和 response mask 下，应能通过 GRPO 学会合法 `<search>` / `<answer>` 输出和多轮搜索。

## H2：Search-R1 在 noisy retrieval 下会系统性退化

随着 fault rate 从 0% 增至 30%，预计出现：

- EM/F1 降低；
- 重复查询上升；
- 首次故障后直接猜答案；
- 错误证据采纳率上升。

## H3：简单 fault augmentation 不能完全解决恢复问题

只在训练中加入故障可能提高容错，但模型仍可能：

- 无脑多搜；
- 原样重复 query；
- 遇到失败后直接回答。

## H4：Recovery reward 能提高恢复成功率

TRUST-R1 应在不明显损害 clean performance 的前提下：

- 提升 first-failure recovery rate；
- 降低 duplicate-query rate；
- 改善 noisy EM/F1；
- 不依赖增加平均搜索次数获得提升。

## H5：恢复能力能迁移到未见故障

训练只见 `empty + duplicate + answer_drop`，测试加入 `hard_negative + timeout`。如果仍有增益，说明模型学到一般恢复策略，而不是记住错误字符串。

---

# 3. 系统结构

```text
Question
   │
   ▼
Qwen2.5-7B-Base Policy
   │
   ├── <search>query</search>
   │             │
   │             ▼
   │      Reliability Wrapper
   │        ├── clean
   │        ├── empty
   │        ├── duplicate
   │        ├── answer_drop
   │        └── hard_negative / timeout (eval only)
   │             │
   │             ▼
   │       CPU FAISS Retriever
   │       E5 + wiki-18 index
   │
   ├── continued reasoning
   ├── query rewrite / retry
   └── <answer>...</answer>
                 │
                 ▼
       EM Reward + Recovery Reward
                 │
                 ▼
             verl GRPO
```

---

# 4. 检索数据与索引恢复

## 4.1 目标配置

- Corpus：Wikipedia 2018 / wiki-18；
- Encoder：`intfloat/e5-base-v2`；
- top-k：3；
- 首选 index：与 Search-R1 baseline 一致的预构建 index；
- 服务：FastAPI/Uvicorn；
- 默认 CPU FAISS。

## 4.2 数据恢复顺序

1. 优先下载官方/公开的预构建 wiki-18 corpus 与 E5 index；
2. 校验文件大小、行数和 checksum；
3. 如果预构建 Flat index 下载失败，不要立即自行编码 2100 万条语料；
4. 先用小语料或 BM25 跑通训练链路；
5. 正式实验前再恢复全量索引；
6. 如果 CPU Flat 检索太慢，优先转换 IVF/PQ/HNSW，而不是立刻占用一张训练 GPU。

## 4.3 CPU FAISS 基准测试

固定 1000 条真实 query，测试并发：

```text
concurrency = 1, 4, 8
```

记录：

- QPS；
- p50 latency；
- p95 latency；
- 峰值内存；
- 检索服务错误率；
- 训练 step 中等待 retriever 的时间占比。

采用 CPU FAISS 的建议门槛：

- concurrency=4 时 p95 不高于约 300ms；或
- retriever 等待时间低于整个 rollout 时间的 20%。

若不满足：

1. 调低并发；
2. 增加 CPU worker；
3. 使用 ANN 压缩索引；
4. 缩短返回文档；
5. 最后才考虑 GPU FAISS。

---

# 5. 环境与代码目录

```text
trust-r1/
├── README.md
├── environment.lock.md
├── configs/
│   ├── smoke_qwen25_3b.yaml
│   ├── baseline_qwen25_7b.yaml
│   ├── fault_aug_qwen25_7b.yaml
│   └── trust_r1_qwen25_7b.yaml
├── trust_r1/
│   ├── tools/
│   │   ├── search_tool.py
│   │   ├── reliability_wrapper.py
│   │   └── fault_profiles.py
│   ├── rewards/
│   │   ├── answer_reward.py
│   │   ├── format_reward.py
│   │   ├── recovery_reward.py
│   │   └── redundancy_penalty.py
│   ├── trajectory/
│   │   ├── parser.py
│   │   ├── logger.py
│   │   └── metrics.py
│   └── eval/
│       ├── fault_sweep.py
│       ├── evaluate.py
│       └── case_analysis.py
├── scripts/
│   ├── download_data.sh
│   ├── launch_retriever_cpu.sh
│   ├── benchmark_retriever.sh
│   ├── run_smoke.sh
│   ├── run_baseline.sh
│   └── run_trust_r1.sh
├── tests/
│   ├── test_parser.py
│   ├── test_fault_wrapper.py
│   ├── test_reward.py
│   └── test_response_mask.py
└── reports/
    ├── experiment_log.md
    ├── failure_taxonomy.md
    └── final_report.md
```

必须固定：

- verl commit；
- PyTorch/CUDA；
- vLLM 或 SGLang 版本；
- Transformers 版本；
- model revision；
- 数据版本；
- index checksum；
- 训练 seed；
- fault seed。

---

# 6. Reward 设计

## 6.1 Baseline reward

保持 Search-R1 的 outcome-only reward：

\[
R_{answer}=\mathbb{1}[normalize(\hat a)=normalize(a)]
\]

评测报告 EM 和 token F1，但训练第一版使用 EM，减少中间奖励投机。

## 6.2 TRUST-R1 reward

\[
R = R_{answer}
+ \lambda_f R_{format}
+ \lambda_r R_{recovery}
- \lambda_d P_{duplicate}
- \lambda_i P_{invalid}
\]

建议启动权重：

```text
lambda_format    = 0.05 ~ 0.10
lambda_recovery  = 0.10
lambda_duplicate = 0.02 ~ 0.05
lambda_invalid   = 0.05
```

辅助 reward 总幅度不能长期超过 answer reward。

## 6.3 Recovery reward 条件

一次轨迹满足以下全部条件才给分：

1. 环境真实注入故障；
2. 故障后的 query 不是原 query 的机械重复；
3. 后续检索获得 answer-bearing document，或明显增加 supporting evidence；
4. 最终答案正确。

```text
fault happened
AND query novelty > threshold
AND later evidence success
AND final answer correct
```

## 6.4 Duplicate penalty

第一版使用规则：

```text
query cosine similarity > 0.92
OR returned doc Jaccard overlap > 0.80
```

阈值必须用约 200 条人工轨迹校准，不能直接写死后不检查。

---

# 7. 训练配置

以下参数是起始配置，不是未经测试的最终值。

## 7.1 3B smoke test

```yaml
model: Qwen/Qwen2.5-3B-Instruct
algorithm: GRPO
train_samples: 1000-2000
train_batch_size: 32
rollout_n: 4
max_prompt_length: 512
max_response_length: 1024
max_tool_calls: 2
retriever_topk: 3
steps: 10-20
fault_rate: 0.0
```

验收条件：

- 20 step 内无 Ray/NCCL/OOM；
- parser error < 5%；
- observation token 的 response mask 正确；
- checkpoint 能保存与恢复；
- reward 分项日志正确；
- 检索服务无持续超时。

## 7.2 7B Base Gate

```yaml
model: Qwen/Qwen2.5-7B
algorithm: GRPO
train_samples: 2000
train_batch_size: 32-64
rollout_n: 4
max_response_length: 1024
max_tool_calls: 2-3
steps: 20-30
fault_rate: 0.0
```

选择 Base 的条件：

- legal tool-call rate ≥ 70%，并持续上升；
- group reward 不是长期全相同；
- search trigger rate 不坍缩为 0；
- 训练后 clean dev 至少出现可观察改善。

如果合法格式率不足：

1. 增加 format reward；
2. format-only SFT；
3. 最后才切 Instruct。

## 7.3 正式 baseline

```yaml
model: Qwen/Qwen2.5-7B
algorithm: GRPO
training: full-weight
train_samples: 10000-20000
train_batch_size: 64
rollout_n: 4
max_prompt_length: 512
max_response_length: 1024-1536
max_obs_length: 512
max_tool_calls: 3
retriever_topk: 3
steps: 100-150
fault_rate: 0.0
```

并行与显存建议：

- 4×A800 80GB 全部参与训练/rollout；
- FSDP full-shard；
- gradient checkpointing 开启；
- remove padding 开启；
- reference model param offload 可开启；
- actor optimizer offload 默认不开，OOM 再开；
- rollout `gpu_memory_utilization` 从 0.45～0.55 起步；
- response length 不要一开始设为 4096；
- group size 第一版固定 4，不追求 8 或 16。

## 7.4 正式 TRUST-R1

与 baseline 完全保持一致：

- 模型；
- 数据；
- 总 step；
- rollout_n；
-最大长度；
- 检索器；
- seed；
- GPU 配置。

只改变：

- fault wrapper；
- recovery reward；
- redundancy penalty。

---

# 8. 数据规模

为了预算与工期，第一版不追求 169K 全量。

建议训练：

```text
NQ:        5K～10K
HotpotQA:  5K～10K
Total:    10K～20K
```

验证：

```text
NQ dev:       300
HotpotQA dev: 300
```

测试：

```text
PopQA:           300
2WikiMultiHopQA: 300
```

如果时间充足再加：

- TriviaQA；
- MuSiQue；
- Bamboogle。

首版最重要的是完整对照和行为分析，不是数据集数量。

---

# 9. 实验矩阵

## 9.1 必须完成

| ID | 训练环境 | Recovery Reward | Duplicate Penalty | 目的 |
|---|---|---:|---:|---|
| B0 | Clean Search-R1 | × | × | 标准 baseline |
| B1 | Fault augmentation | × | × | 判断仅数据增强是否足够 |
| M1 | Fault augmentation | ✓ | × | 恢复奖励贡献 |
| M2 | Fault augmentation | ✓ | ✓ | TRUST-R1 Full |

## 9.2 可选实验

| ID | 内容 |
|---|---|
| A1 | TRUST-R1 w/o answer_drop |
| A2 | Base vs Instruct 30-step 对照 |
| A3 | Qwen3-8B-Base 小规模迁移 |
| A4 | 固定故障率 vs curriculum |

## 9.3 随机种子

在预算 1000 元内的实际策略：

- B0 与 M2：至少 2 seeds；
- B1 与 M1：1 seed；
- 其他只做 1 seed。

如果预算不足，优先保住 B0/M2 双 seed，不要平均砍掉所有训练步数。

---

# 10. 故障设置

## 10.1 训练故障

```text
empty
duplicate
answer_drop
```

建议混合 fault rate：

```text
10% → 20% → 30%
```

第一版可先固定 20%，如果模型出现“不搜索”再改为课程训练。

## 10.2 测试故障

```text
clean
seen-mixed 10%
seen-mixed 20%
seen-mixed 30%
unseen-hard-negative 20%
unseen-timeout 20%
```

每个模型使用同一问题、同一 fault seed。

---

# 11. 评测指标

## 11.1 任务指标

- Exact Match；
- token F1；
- clean-to-noisy performance drop。

## 11.2 工具行为指标

- legal tool-call rate；
- search trigger rate；
- average search calls；
- query rewrite rate；
- duplicate query rate；
- answer-bearing passage recall；
- premature answer rate；
- invalid action rate。

## 11.3 核心鲁棒性指标

### First-Failure Recovery Rate

\[
FRR=
\frac{
\#(首次故障后获得新证据且最终答对)
}{
\#(至少遭遇一次故障的轨迹)
}
\]

### Robustness Drop

\[
\Delta(p)=Score_{clean}-Score_{fault=p}
\]

### Query Novelty

\[
Novelty(q_t)=1-\max_{i<t}\cos(e(q_t),e(q_i))
\]

---

# 12. 10 天执行计划

## Day 0：租卡前准备

在本地完成：

- clone 仓库；
- 写好环境安装脚本；
- 写好数据下载脚本；
- 定义 trajectory schema；
- 定义 fault wrapper 接口；
- 准备所有配置文件模板；
- 不在 A800 上边想方案边写代码。

## Day 1：环境与索引

- 租 4×A800；
- 安装并 pin verl、PyTorch、rollout engine；
- 下载模型；
- 下载 wiki corpus/index；
- 启动 CPU FAISS；
- 做 1000-query 检索基准。

退出条件：检索服务稳定、版本冻结。

## Day 2：3B smoke

- 10～20 step；
- 检查 response mask；
- 检查 reward；
- 检查 checkpoint；
- 检查多轮 observation 注入。

退出条件：整个 pipeline 真正端到端跑通。

## Day 3：7B Base Gate

- 7B Base 20～30 step；
- 统计合法格式率与搜索率；
- 决定是否 format-only SFT；
- 最晚当天冻结 Base/Instruct。

## Day 4：Baseline 正式训练

- B0：100～150 step；
- 保存关键 checkpoint；
- clean dev 评测。

## Day 5：脆弱性诊断

- fault sweep；
- 分析 empty/duplicate/answer_drop；
- 人工查看 50～100 条轨迹；
- 验证问题是真实现象而不是 parser bug。

## Day 6：Fault Augmentation

- 实现并测试 B1；
- 运行 fault augmentation baseline；
- 对比“仅增强环境”能解决多少。

## Day 7：Recovery Reward

- 实现 M1；
- 分项记录 reward；
- 检查是否出现无限重试或 query 伪改写。

## Day 8：TRUST-R1 Full

- 加 duplicate penalty；
- 运行 M2；
- clean/noisy dev 评测。

## Day 9：关键消融与第二 seed

优先顺序：

1. B0 第二 seed；
2. M2 第二 seed；
3. B1/M1 补齐；
4. unseen fault。

## Day 10：报告与简历

- held-out test；
- 图表；
- 轨迹案例；
- README；
- 复现命令；
- 简历 bullet；
- 录制 3～5 分钟项目演示。

如果扩展到 14 天，多出的 4 天用于：

- 补第二 seed；
- curriculum；
- Qwen3-8B 小规模验证；
- 更多 held-out 数据集。

---

# 13. 预算控制

## 13.1 原则

预算上限 1000 元，不代表把 1000 元全部用于长时间挂机。

建议分配：

- 15%：环境、下载、索引与 smoke；
- 25%：正式 baseline；
- 35%：B1/M1/M2；
- 15%：第二 seed 与 held-out evaluation；
- 10%：OOM、崩溃和紧急重跑。

## 13.2 GPU 小时计算

设 4×A800 整机每小时价格为 `P`，可支配训练小时不要按 `1000/P` 计算，而按：

```text
可计划小时 = 预算的约 80% / P
```

保留约 20% 作为重跑余量。

## 13.3 省钱重点

- 代码先在本地写好；
- 不在 A800 上进行文献调研；
- 数据下载、index 构建与环境安装写成可恢复脚本；
- 每次正式训练前先跑 2-step dry run；
- 只保留关键 checkpoint；
- fault evaluation 用固定 300 条 dev，不频繁全量评测；
- 不在第一版加入 multi-tool；
- 不为“模型更新”迁移 Qwen3.5。

---

# 14. 止损规则

## 14.1 环境止损

最新版 verl Search Tool 链路 1 天内无法跑通：

> 回退 Search-R1 原始 Qwen2.5 支持路线，不继续升级依赖。

## 14.2 Base 模型止损

7B Base 经过 format reward 与短 SFT 后，30 step 仍满足以下任一条件：

- legal tool-call rate < 70%；
- search trigger rate 接近 0；
- reward 长期无区分；
- 输出长度持续失控。

则切换 Qwen2.5-7B-Instruct。

## 14.3 检索止损

全量 E5 index 在 Day 1 不能恢复：

- 使用小规模语料完成 pipeline；
- baseline 和正式方法必须在同一检索器上比较；
- 后续恢复全量索引再补正式结果；
- 不允许因为索引问题连续两三天不训练。

## 14.4 方法止损

M1/M2 没有提升时，不继续添加更多 reward。先检查：

- fault 是否真的触发；
- recovery reward 是否稀疏到几乎为零；
- query novelty 阈值是否错误；
- 辅助 reward 是否压过 answer reward；
- 模型是否学成不搜索；
- baseline 是否训练不足；
- 检索结果是否本身无法覆盖答案。

## 14.5 4090 止损

如果最终只能使用 4×4090：

- 放弃 7B full-weight 作为硬约束；
- 改为 7B LoRA 或 3B full-weight；
- 不通过大量 CPU offload 强行制造“能启动但一步极慢”的实验；
- 简历里诚实写明 GRPO-LoRA。

---

# 15. 最终成功标准

## 最低可交付

- Search-R1 baseline 跑通；
- 完成三类故障环境；
- 证明 baseline 在 noisy retrieval 下系统退化；
- TRUST-R1 至少在一个主要 noisy setting 上提升；
- 有完整失败分类和案例。

## 良好结果

- clean 性能基本不下降；
- 20% mixed fault 下 EM/F1 有稳定提升；
- first-failure recovery rate 提升；
- duplicate-query rate 下降；
- 两个主要实验 seed 趋势一致；
- unseen fault 仍有正向迁移。

## 强结果

- B0/B1/M1/M2 完整；
- 两个 seed；
- 4 个测试数据集；
- 方法增益来自更好的恢复，而不是更多搜索；
- GitHub 可复现；
- 有训练吞吐、检索延迟和失败轨迹分析。

---

# 16. 最终简历模板

## TRUST-R1：面向不可靠检索工具的鲁棒搜索智能体强化学习

- 基于 **verl + GRPO + Qwen2.5-7B + E5/FAISS** 复现 Search-R1 多轮推理—检索训练链路，搭建包含 empty、duplicate、answer-drop、hard-negative 和 timeout 的可控检索故障环境，并实现轨迹级工具调用与恢复行为分析。
- 诊断标准 Search-R1 在 noisy retrieval 下存在重复查询、失败后直接作答和错误证据采纳等问题，提出 **TRUST-R1**，通过工具故障随机化、恢复条件奖励和冗余正则化优化查询改写与失败恢复策略。
- 在 `[数据集]` 的 `[故障率]` 混合故障下，模型 EM/F1 提升 `[X.X]pp`，首次故障恢复率提升 `[X.X]%`，重复查询率下降 `[X.X]%`；消融实验验证环境增强与恢复奖励的独立贡献，并在未见故障类型上获得泛化增益。

所有数字必须来自真实日志。

---

# 17. 当前唯一还需确认的硬条件

开始租卡前只需确认两项：

1. A800 是否为 **80GB**；
2. 机器主存是否至少 **128GB**，数据盘是否至少有 **200GB 可用空间**。

如果这两项满足，按本方案直接开工，不再继续扩展选题。
