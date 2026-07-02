# TRUST-R1 项目初始化设计文档

日期：2026-07-02  
状态：brainstorming 已确认，待用户审阅

## 1. 项目定位与目录策略

TRUST-R1 是一个基于 Search-R1 改造的个人研究项目，研究主题是：

> 面向不可靠检索工具的鲁棒搜索智能体强化学习。

目录策略采用“复制原项目，再在副本中改造”的方式：

```text
/Users/icarus/Documents/Search-R1-main   # 原始 Search-R1 参考副本，尽量不改
/Users/icarus/Documents/TRUST-R1         # TRUST-R1 工作目录，可以直接改
```

核心原则：

1. `Search-R1-main` 保留为干净参考副本。
2. 后续开发主要发生在 `TRUST-R1`。
3. `TRUST-R1` 内部可以直接修改复制来的 Search-R1 / verl 原代码。
4. 不要求所有新逻辑都外挂到新目录。
5. 不为了“看起来像新项目”而大规模重命名包、目录或 import。
6. 不做无关重构、依赖升级或训练框架迁移，除非用户明确要求。
7. 尽量保留原 Search-R1 baseline 可运行，方便做 clean baseline 和回退对照。

本项目是 fork-style 改造：保留原始工程结构，同时允许在副本中对原训练、检索、reward、eval 链路做必要增量修改。

---

## 2. `CLAUDE.md` 的定位与硬规则

`CLAUDE.md` 不是完整实验方案，也不是长篇项目说明。它的作用是给后续 Claude 会话提供必须遵守的项目级硬规则。

建议 `CLAUDE.md` 保持简短、明确、可执行，至少包含以下规则。

### 2.1 项目身份

- 当前项目是 TRUST-R1。
- TRUST-R1 基于 Search-R1 改造。
- 目标是研究 noisy / unreliable retrieval 下搜索智能体的鲁棒恢复能力。
- `Search-R1-main` 是参考副本，`TRUST-R1` 是工作目录。

### 2.2 本地 Mac 与 AutoDL 边界

本地 Mac 只用于：

- 代码编辑；
- 文档整理；
- 配置模板；
- 小型单元测试；
- 静态检查；
- README / 报告草稿。

本地 Mac 禁止：

- 下载大模型；
- 下载大数据集；
- 下载或构建 wiki index；
- 启动完整 retriever / FAISS 服务；
- 启动 Ray / vLLM；
- 跑 RL 训练；
- 跑大规模 evaluation；
- 保存 checkpoint 或大日志。

AutoDL 负责：

- 数据；
- 模型；
- 索引；
- retriever；
- smoke test；
- 训练；
- 评测；
- checkpoint；
- 完整实验日志；
- 远程 debug。

任何可能下载大文件、占用 GPU、启动训练、启动 Ray/vLLM/FAISS、或进行大规模评测的命令，都必须先确认运行环境是 AutoDL，而不是本地 Mac。

### 2.3 GitHub 主版本规则

- GitHub 是代码的唯一主版本。
- 本地 Mac 和 AutoDL 都可以改代码。
- 重要改动必须 commit。
- 正式实验必须记录对应 git commit hash。
- 不允许长期存在“远程改了核心代码但没有进入 git”的状态。
- checkpoint、模型权重、数据集、索引、大日志不能进入 git。

### 2.4 改造方式

- 在 `TRUST-R1` 中可以直接改复制来的 Search-R1 / verl 代码。
- 如果直接改原代码能降低复杂度，就优先直接改。
- 不为保持原文件“纯净”而写复杂绕路适配层。
- 但仍要避免无关重构、广泛重命名和依赖升级。
- 每个改动都应该服务于 TRUST-R1 的实验目标。

### 2.5 结果纪律

- README、报告、简历中的所有数字必须来自真实日志。
- 不编造 EM/F1、恢复率、提升百分比或训练结果。
- 没有完成实验前，只能写“计划”“待完成”“实验中”，不能写成已完成结论。

---

## 3. TRUST-R1 代码改造架构

TRUST-R1 的代码目标不是重写 Search-R1，而是在 Search-R1 原有推理—检索—训练链路上加入鲁棒性机制。

首版核心能力包括：

1. 不可靠检索工具 Fault Injection；
2. 轨迹级日志 Trajectory Logging；
3. 恢复奖励 Recovery Reward；
4. 重复搜索惩罚 Duplicate / Redundancy Penalty；
5. noisy retrieval 评测体系。

### 3.1 总体原则

```text
优先复用原 Search-R1 链路。
允许直接改原代码。
新增机制必须可配置、可关闭、可记录。
不为了模块纯洁性牺牲简单性。
每个正式实验必须可复现、可对照、可回退。
```

### 3.2 Fault Injection

Fault Injection 用来模拟不可靠检索工具，使检索结果有时出现空返回、重复返回、删除关键证据等情况。

首版建议支持：

```text
clean       # 原始检索结果，无故障
empty       # 返回空结果
drop_top    # 删除 top-ranked 证据
duplicate   # 返回重复文档
mixed       # 按概率混合多种故障
```

后续可选扩展：

```text
hard_negative
stale_result
timeout
shuffle
truncate
```

要求：

- fault 类型可配置；
- fault rate 可配置；
- fault seed 可配置；
- 每次故障事件可记录；
- clean 模式必须保留，方便跑原始 baseline。

### 3.3 Trajectory Logging

TRUST-R1 的价值不只体现在最终准确率，还要解释模型是否真的学会了故障恢复。因此需要记录轨迹。

建议日志至少包含：

```json
{
  "run_id": "20260702_M2_full_seed42",
  "sample_id": "...",
  "question": "...",
  "gold_answer": "...",
  "final_answer": "...",
  "is_correct": true,
  "turns": [
    {
      "step": 1,
      "query": "...",
      "fault_enabled": true,
      "fault_type": "drop_top",
      "retrieved_doc_ids": ["..."],
      "model_action": "search"
    }
  ],
  "recovery": {
    "had_fault": true,
    "searched_again_after_fault": true,
    "changed_query_after_fault": true,
    "answered_correctly_after_fault": true
  }
}
```

首版可以不一次性实现全部字段，但必须能支持后续分析：

- 搜了几次；
- 每次搜什么 query；
- 是否遭遇故障；
- 遭遇哪种故障；
- 是否改写 query；
- 是否最终答对；
- 是否出现重复搜索。

完整轨迹日志留在 AutoDL。GitHub 只保存小型抽样轨迹、指标摘要和图表。

### 3.4 Reward 修改

Reward 改造以最小可控为原则。

基础形式：

```text
总 reward = 原 Search-R1 answer reward
          + 小权重 recovery bonus
          - 小权重 duplicate / invalid penalty
```

约束：

- 保留原 answer correctness reward 作为主信号；
- recovery reward 只能作为辅助项，不能长期压过 answer reward；
- duplicate penalty 只惩罚明显重复或无效搜索；
- reward 分项必须单独记录，便于排查；
- 所有新增 reward 都必须可通过配置关闭。

### 3.5 Evaluation Metrics

基础指标：

- Exact Match / Accuracy；
- token F1（如数据集适合）；
- average search turns；
- average reward。

鲁棒性指标：

- clean accuracy；
- noisy accuracy；
- robustness drop；
- first-failure recovery rate；
- duplicate query rate；
- query rewrite rate；
- fault-hit accuracy；
- premature answer rate；
- invalid action rate。

核心报告应包含：

1. clean baseline 对比；
2. noisy retrieval 下性能对比；
3. 不同故障类型下的结果；
4. 行为指标；
5. 失败案例与恢复案例。

---

## 4. 实验路线与资源冻结计划

实验顺序采用“先 baseline，后 noisy 诊断，再方法增强”的路线。

### 4.1 总路线

```text
Layer 0: 项目初始化与本地准备
Layer 1: 原 Search-R1 clean baseline 跑通
Layer 2: noisy retrieval 脆弱性诊断
Layer 3: fault augmentation baseline
Layer 4: recovery reward 方法
Layer 5: duplicate penalty / TRUST-R1 full
Layer 6: noisy suite + report + resume
```

每一层都需要明确退出条件，不能只凭感觉进入下一步。

### 4.2 Day 0：本地准备

本地只做：

- 复制 `Search-R1-main` 到 `TRUST-R1`；
- 初始化 git；
- 写 `CLAUDE.md`；
- 整理 README；
- 准备 docs；
- 准备 configs；
- 准备 tests；
- 准备 AutoDL 操作文档。

本地不做：

- 模型下载；
- 数据下载；
- 全量索引；
- retriever；
- Ray/vLLM；
- RL 训练；
- 大规模评测。

Day 0 成功标准：

- 目录清楚；
- `CLAUDE.md` 写好；
- AutoDL 工作流写好；
- 实验配置有模板；
- 单测能覆盖关键纯逻辑；
- GitHub 仓库准备好。

### 4.3 AutoDL Day 1：环境、数据、索引

目标：确认环境可用，而不是马上训练。

步骤：

1. clone GitHub 项目；
2. 创建 conda 环境；
3. 固定 Python / PyTorch / CUDA / vLLM / transformers / verl 版本；
4. 下载模型；
5. 下载 corpus / index；
6. 启动 retriever；
7. 做检索服务 benchmark；
8. 记录环境版本和路径。

退出条件：

- retriever 稳定；
- 模型能加载；
- 路径固定；
- 版本记录完成；
- 没有明显磁盘或显存风险。

### 4.4 AutoDL Day 2：3B Smoke Test

目标：确认训练链路没坏。

建议：

```text
Qwen/Qwen2.5-3B 或 Qwen/Qwen2.5-3B-Instruct
10-20 steps
小 batch
max_tool_calls = 2
top-k = 3
fault_rate = 0.0
```

必须检查：

- OOM；
- Ray / NCCL；
- search tool 调用；
- response mask；
- reward 分项日志；
- checkpoint save / load；
- observation 注入；
- parser error。

Smoke test 不通过，不进入 7B。

### 4.5 AutoDL Day 3：7B Base Gate

目标：判断 `Qwen/Qwen2.5-7B` Base 是否能作为主模型。

观察：

- legal tool-call rate；
- search trigger rate；
- reward 是否有区分；
- 输出长度是否失控；
- clean dev 是否有改善迹象。

止损规则：

```text
30 step 后 legal tool-call rate < 70%
或 search trigger rate 接近 0
或 reward 长期无区分
或输出长度持续失控
=> 不继续硬训 Base，切换策略
```

### 4.6 正式实验矩阵

| ID | 名称 | 检索环境 | Recovery Reward | Duplicate Penalty | 目的 |
|---|---|---|---:|---:|---|
| B0 | Search-R1 Clean Baseline | clean | 否 | 否 | 原始能力基线 |
| B1 | Fault Augmentation | noisy training | 否 | 否 | 判断仅环境增强是否足够 |
| M1 | Recovery Reward | noisy training | 是 | 否 | 验证恢复奖励贡献 |
| M2 | TRUST-R1 Full | noisy training | 是 | 是 | 完整方法 |

对照原则：

- 同一模型；
- 同一数据；
- 同一检索器；
- 同一训练步数；
- 同一 batch / rollout_n；
- 同一 seed 策略；
- 除目标变量外不随意改变其他设置。

---

## 5. GitHub / SSH / AutoDL 双开发机工作流

最终原则：

> GitHub 是唯一主版本。本地 Mac 和 AutoDL 都可以开发，但大数据、大模型、索引、训练和评测只在 AutoDL。任何正式实验都必须对应 git commit。

### 5.1 本地 Mac 职责

允许：

- 项目初始化；
- 文档；
- README / CLAUDE.md / docs；
- 轻量代码编辑；
- 小单测；
- 配置模板；
- 报告草稿；
- 结果图表整理。

禁止：

- 下载大模型；
- 下载大数据；
- 构建 wiki index；
- 启动 Ray/vLLM/full FAISS；
- RL 训练；
- 大规模 evaluation；
- checkpoint 存储。

### 5.2 AutoDL 职责

负责：

- clone GitHub 仓库；
- 安装环境；
- 下载模型；
- 下载数据；
- 恢复或构建检索索引；
- 启动 retriever；
- smoke test；
- 训练；
- 评测；
- checkpoint；
- 完整实验日志；
- 远程 debug。

AutoDL 可以直接改代码，但重要改动必须 commit / push。

### 5.3 同步规则

本地开发流程：

```bash
git pull
# edit files
git status
git add ...
git commit -m "..."
git push
```

AutoDL 运行流程：

```bash
git pull
# run smoke/train/eval
```

AutoDL debug 流程：

```bash
git pull
# edit files on remote
# run small debug / smoke
git status
git add ...
git commit -m "..."
git push
```

本地同步远程改动：

```bash
git pull
```

规则：

- 谁改代码，谁提交；
- 谁跑正式实验，谁记录 commit hash；
- 不从未提交代码状态跑正式实验。

### 5.4 AutoDL 路径约定

推荐：

```text
/root/autodl-tmp/TRUST-R1/       # code repo
/root/autodl-tmp/data/           # datasets/corpus
/root/autodl-tmp/models/         # model checkpoints
/root/autodl-tmp/indexes/        # FAISS/BM25 indexes
/root/autodl-tmp/runs/           # training outputs/checkpoints/logs
/root/autodl-tmp/reports/        # small exported results
```

实际路径以 AutoDL 机器为准，最终写入 `docs/autodl_workflow.md` 或 `reports/environment_lock.md`。

### 5.5 训练前 checklist

每次正式训练前必须确认：

```text
[ ] 当前在 AutoDL，不在本地 Mac
[ ] git status 干净，或明确记录 uncommitted diff
[ ] git commit hash 已记录
[ ] config 文件路径正确
[ ] 数据路径存在
[ ] 模型路径存在
[ ] index / retriever 可用
[ ] fault seed / train seed 已写入配置
[ ] 先跑过 2-step dry run
[ ] 输出目录是新的 run_id
[ ] checkpoint / log 不会进 git
```

---

## 6. 文档结构与落地顺序

目标文档结构：

```text
TRUST-R1/
├── CLAUDE.md
├── README.md
├── TRUST-R1_完整实验方案_v1.0_资源冻结版.md
├── docs/
│   ├── autodl_workflow.md
│   ├── experiment_matrix.md
│   ├── logging_schema.md
│   └── superpowers/
│       └── specs/
│           └── 2026-07-02-trust-r1-project-bootstrap-design.md
└── reports/
    └── README.md
```

各文档职责：

- `CLAUDE.md`：Claude 必须遵守的硬规则；
- `README.md`：GitHub 外部读者看到的项目介绍；
- `docs/autodl_workflow.md`：AutoDL 操作流程；
- `docs/experiment_matrix.md`：实验矩阵与成功标准；
- `docs/logging_schema.md`：日志、轨迹和指标格式；
- `reports/README.md`：说明 reports 只放真实实验的小型摘要、图表和抽样案例；
- `docs/superpowers/specs/...design.md`：本设计文档。

### 6.1 首次落地顺序

1. 复制 `Search-R1-main` 到 `TRUST-R1`；
2. 在 `TRUST-R1` 初始化 git；
3. 添加 `.gitignore`；
4. 添加 `CLAUDE.md`；
5. 添加 docs 骨架；
6. 写入本设计文档；
7. 更新 README 为 TRUST-R1 项目说明；
8. 首次 commit：`Initialize TRUST-R1 project from Search-R1`；
9. 用户手动创建 GitHub 仓库；
10. 用户提供 remote URL 后，再连接 remote 并 push。

### 6.2 第一轮不做的事情

第一轮初始化不做：

- 不实现 fault wrapper；
- 不改 reward；
- 不跑训练；
- 不下载模型；
- 不下载数据；
- 不连接 AutoDL 执行实验；
- 不大规模修改 Search-R1 训练代码；
- 不写假的实验结果；
- 不迁移新版 verl；
- 不切换 Qwen3；
- 不把所有 Search-R1 包名改成 TRUST-R1。

---

## 7. 后续实施阶段

设计通过后，后续 implementation plan 可以拆成：

1. Phase 0：项目初始化、`CLAUDE.md`、README、docs、git；
2. Phase 1：baseline 保护与 AutoDL smoke checklist；
3. Phase 2：fault injection；
4. Phase 3：trajectory logging；
5. Phase 4：recovery reward 与 duplicate penalty；
6. Phase 5：noisy evaluation 与 report。

每个阶段都应该先写 implementation plan，再改代码。

---

## 8. 最终设计摘要

TRUST-R1 将作为 Search-R1 的 fork-style 改造项目存在。我们保留 `Search-R1-main` 作为原始参考副本，在 `TRUST-R1` 中允许直接修改复制来的 Search-R1 / verl 原代码，但避免无关重构、依赖升级和大规模重命名。`CLAUDE.md` 只写硬规则：本地 Mac 禁止大实验，AutoDL 负责数据、模型、索引、训练和评测，GitHub 是唯一主版本，所有正式实验必须对应 commit，不能伪造结果。代码路线围绕 fault injection、trajectory logging、recovery reward、duplicate penalty 和 noisy evaluation 展开。实验路线先 baseline，再 noisy 诊断，再 fault augmentation，最后 TRUST-R1 full。第一轮只做项目初始化和文档骨架，不实现训练逻辑。
