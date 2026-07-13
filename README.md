# TRUST-R1

**TRUST-R1: Tool-Reliability-aware Utility and Self-recovery Training for Search Agents**

中文：**面向不可靠检索工具的鲁棒搜索智能体强化学习框架**

> 当前状态：项目初始化与实验设计阶段。实验结果尚未完成，本文档不包含任何未验证的性能数字。

## 1. 项目简介

TRUST-R1 是一个基于 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) 的研究项目，目标是研究搜索增强 LLM agent 在 **不可靠检索工具** 下的鲁棒性。

Search-R1 让模型通过强化学习学会在推理过程中调用搜索工具。TRUST-R1 在此基础上关注一个更现实的问题：

> 当搜索工具出现空返回、重复返回、关键证据缺失、错误证据或超时等故障时，模型能否识别检索没有带来有效信息，并通过 query 改写、继续检索或及时停止来完成任务？

首版项目采用最小改造原则：保留 Search-R1 原有工程结构和训练链路，在复制出的 TRUST-R1 项目中直接增量修改必要代码，而不是重写训练框架。

## 2. 研究问题

本项目关注：

1. 标准 Search-R1 在 noisy retrieval 下是否会系统性退化？
2. 仅加入 fault augmentation 是否足以提升鲁棒性？
3. recovery-aware reward 能否提升首次故障后的恢复成功率？
4. duplicate / redundancy penalty 能否降低重复 query 和无效搜索？
5. TRUST-R1 的收益是否来自更好的恢复行为，而不是简单增加搜索次数？

## 3. 计划实现的核心模块

首版围绕五个模块展开：

```text
Fault Injection        # 检索故障注入：empty / duplicate / drop_top / mixed
Trajectory Logging     # 轨迹记录：query、fault、evidence、answer、recovery
Recovery Reward        # 故障后恢复奖励
Duplicate Penalty      # 重复搜索与无效搜索惩罚
Noisy Evaluation       # clean/noisy/fault-type 分组评测
```

## 4. 实验矩阵

首版核心实验组：

| ID | 名称 | 检索环境 | Recovery Reward | Duplicate Penalty | 目的 |
|---|---|---|---:|---:|---|
| B0 | Search-R1 Clean Baseline | clean | 否 | 否 | 原始能力基线 |
| B1 | Fault Augmentation | noisy training | 否 | 否 | 判断仅故障增强是否足够 |
| M1 | Recovery Reward | noisy training | 是 | 否 | 验证恢复奖励贡献 |
| M2 | TRUST-R1 Full | noisy training | 是 | 是 | 完整方法 |

详细实验矩阵见 [`docs/experiment_matrix.md`](docs/experiment_matrix.md)。

## 5. 本地与 AutoDL 分工

本地 Mac 只用于：

- 代码编辑；
- 文档整理；
- 配置模板；
- 小型单元测试；
- 静态检查；
- 报告草稿。

AutoDL 负责：

- 数据下载；
- 模型下载；
- 索引恢复或构建；
- retriever 服务；
- smoke test；
- RL 训练；
- evaluation；
- checkpoint 和完整日志。

详细工作流见 [`docs/autodl_workflow.md`](docs/autodl_workflow.md)。

## 6. 文档入口

- [`TRUST-R1_完整实验方案_v1.0_资源冻结版.md`](TRUST-R1_完整实验方案_v1.0_资源冻结版.md)：完整实验方案。
- [`CLAUDE.md`](CLAUDE.md)：Claude 项目级工作规则。
- [`docs/superpowers/specs/2026-07-02-trust-r1-project-bootstrap-design.md`](docs/superpowers/specs/2026-07-02-trust-r1-project-bootstrap-design.md)：项目初始化设计文档。
- [`docs/autodl_workflow.md`](docs/autodl_workflow.md)：AutoDL 远程实验工作流。
- [`docs/experiment_launch_guide.md`](docs/experiment_launch_guide.md)：AutoDL 实验启动命令与分阶段流程。
- [`docs/experiment_matrix.md`](docs/experiment_matrix.md)：实验矩阵与成功标准。
- [`docs/logging_schema.md`](docs/logging_schema.md)：日志、轨迹和指标 schema。
- [`reports/README.md`](reports/README.md)：实验结果摘要目录说明。

## 7. 当前状态

已完成：

- 从 Search-R1 复制出 TRUST-R1 工作目录；
- 初始化项目设计文档；
- 明确本地 / AutoDL / GitHub 工作边界；
- 规划实验矩阵、日志 schema 和文档结构。

待完成：

- AutoDL 环境准备；
- Search-R1 clean baseline smoke test；
- fault injection 实现；
- trajectory logging 实现；
- recovery reward 与 duplicate penalty；
- noisy evaluation；
- 真实实验结果与报告。

## 8. 结果声明

当前仓库尚未完成正式训练和评测，因此不报告任何 EM、F1、recovery rate 或提升百分比。

后续所有结果必须来自真实日志，并记录：

- git commit hash；
- config；
- run command；
- model / dataset / retriever；
- train seed；
- fault seed；
- metrics；
- trajectory samples。

## 9. 上游致谢

本项目基于 Search-R1 改造。Search-R1 的原始工作包括：

- Paper: [Search-R1 paper 1](https://arxiv.org/pdf/2503.09516), [Search-R1 paper 2](https://arxiv.org/abs/2505.15117)
- Code: [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)
- Framework basis: [veRL](https://github.com/volcengine/verl)

如果使用本项目中的 Search-R1 原始代码或结果，请同时引用 Search-R1 和其相关依赖项目。
