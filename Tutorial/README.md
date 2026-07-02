# Search-R1 新手完整教程

本教程旨在帮助零基础新手系统性地学习强化学习(RL)和 LLM Agent 的基础知识，并通过 Search-R1 项目掌握相关技术。

## 📚 教程结构

| 章节 | 文件名 | 内容概要 |
|------|--------|----------|
| 00 | 教程概览 | 教程目标、学习路径、前置知识 |
| 01 | 强化学习基础 | RL 核心概念、MDP、PPO/GRPO 算法 |
| 02 | LLM Agent 基础 | Agent 架构、ReAct 范式、工具使用 |
| 03 | 项目架构详解 | 整体架构、数据流、模块依赖 |
| 04 | 核心代码解析 | generation.py、ray_trainer.py、core_algos.py |
| 05 | 环境配置与安装 | 依赖安装、配置说明、常见问题 |
| 06 | 快速开始指南 | 完整运行流程、演示脚本 |
| 07 | 数据处理流程 | 数据格式、处理脚本、自定义数据 |
| 08 | 训练流程详解 | PPO/GRPO 完整流程、分布式训练 |
| 09 | 推理与评估 | 模型导出、性能评估、调优策略 |
| 10 | 实战项目 | 实验设计、项目扩展、论文复现 |
| 11 | 面试知识点总结 | RL/Agent 面试题、项目亮点、简历撰写 |
| 14 | 训练数据长什么样 | 结合 `nq_search.py` 理解训练样本结构 |
| 15 | 模型怎么发起搜索 | 结合 `generation.py` 理解 search/answer 动作循环 |
| 16 | 检索服务返回什么 | 结合 `retrieval_server.py` 理解返回结果格式 |
| 17 | 一条样本如何完成闭环 | 从 QA 数据到 rollout 再到 reward 的完整链路 |
| 18 | 核心文件阅读路线 | 结合 `train_grpo.sh`、`generation.py`、`ray_trainer.py`、`qa_em.py` 建立主线 |
| 19 | 四个核心文件逐段讲解 | 分别结合代码讲 `train_grpo.sh`、`generation.py`、`ray_trainer.py`、`qa_em.py` |
| 20 | veRL 框架学习指导 | 结合 Search-R1 代码理解 veRL 的 trainer、worker、DataProto 和 Ray 结构 |
| 21 | 项目主流程代码串讲 | 从数据处理到 PPO/GRPO 更新的完整代码链路 |
| 22 | 检索结果如何进入下一轮上下文 | 结合 `generation.py` 详细讲搜索结果的格式化、包装和拼接 |
| 23 | PPO 更新代码与数学 | 从采样、reward、GAE、critic/actor update 串起 PPO 路径 |
| 24 | GRPO 更新代码与数学 | 从采样、group-relative advantage 到 actor update 串起 GRPO 路径 |

## 🚀 快速开始

```bash
# 1. 环境配置
conda create -n searchr1 python=3.10
conda activate searchr1
pip install -r requirements.txt

# 2. 启动检索服务器
bash example/retriever/retrieval_launch_bm25.sh

# 3. 开始训练
bash train_ppo.sh
```

## 📖 学习路径

```
第1天     → 快速浏览全部内容，理解项目全貌
第2-3天   → 学习 01-02 章（强化学习 + Agent 基础）
第4-5天   → 学习 03-04 章（项目架构 + 核心代码）
第6天     → 环境配置 + 运行 demo
第7-10天  → 深入研究 + 实战项目
```

## 🎯 学习目标

完成本教程后，你将掌握：

1. **强化学习基础**：理解 RL 基本概念，掌握 PPO/GRPO 算法原理
2. **LLM Agent 设计**：了解 Agent 架构和 ReAct 范式
3. **项目实战能力**：能够运行、训练和评估 Search-R1
4. **面试竞争力**：掌握 RL + LLM 相关的核心知识点

## 📝 简历亮点

基于 Search-R1 项目，可以在简历中突出：

- **RL + LLM 融合能力**：熟悉 PPO/GRPO 算法原理和实现
- **系统设计能力**：分布式训练系统、检索系统架构
- **工程实践能力**：大规模模型训练调优、端到端项目经验

## 📚 前置知识

- Python 编程基础
- 基本的机器学习概念
- 高中数学基础（概率、矩阵）

## 🔗 参考资料

- [DeepSeek-R1 论文](https://arxiv.org/abs/2501.12599)
- [PPO 算法原论文](https://arxiv.org/abs/1707.06347)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [veRL 框架文档](VERL_README.md)

---

**祝学习愉快！**
