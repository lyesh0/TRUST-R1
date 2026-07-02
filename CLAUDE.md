# CLAUDE.md

本文件是 TRUST-R1 项目的 Claude 工作规则。后续 Claude 会话必须优先遵守这里的约束。

## 项目身份

- 本项目是 **TRUST-R1**，基于 Search-R1 改造。
- 研究目标：面向不可靠检索工具的鲁棒搜索智能体强化学习。
- `/Users/icarus/Documents/Search-R1-main` 是原始参考副本，尽量不改。
- `/Users/icarus/Documents/TRUST-R1` 是工作目录，可以直接改。

## 改造原则

- 允许直接修改 `TRUST-R1` 内复制来的 Search-R1 / verl 原代码。
- 如果直接改原代码更简单、更少 bug，就不要为了“外挂”写复杂适配层。
- 不做无关重构、广泛重命名、依赖升级或训练框架迁移，除非用户明确要求。
- 不把所有 `search_r1` 包名强行改成 `trust_r1`。
- 尽量保留原 Search-R1 clean baseline 可运行，方便对照和回退。
- 每个改动都应该服务于 TRUST-R1 的实验目标：fault injection、trajectory logging、recovery reward、duplicate penalty、noisy evaluation。

## 本地 Mac 与 AutoDL 边界

本地 Mac 只用于：

- 代码编辑；
- 文档整理；
- 配置模板；
- 小型单元测试；
- 静态检查；
- README / 报告草稿。

本地 Mac 禁止：

- 下载大模型或大数据集；
- 下载或构建 wiki index；
- 启动完整 retriever / FAISS 服务；
- 启动 Ray / vLLM；
- 跑 RL 训练；
- 跑大规模 evaluation；
- 保存 checkpoint 或大日志。

AutoDL 负责：

- 数据、模型、索引；
- retriever；
- smoke test；
- 训练和评测；
- checkpoint；
- 完整实验日志；
- 远程 debug。

在建议或执行任何可能下载大文件、占用 GPU、启动训练、启动 Ray/vLLM/FAISS、或进行大规模评测的命令前，必须确认当前环境是 AutoDL，而不是本地 Mac。

## GitHub 与版本纪律

- GitHub 是代码的唯一主版本。
- 本地 Mac 和 AutoDL 都可以改代码。
- 重要改动必须 commit。
- 正式实验必须记录对应 git commit hash。
- 不允许长期存在“远程改了核心代码但没有进入 git”的状态。
- 不要从未提交代码状态跑正式实验；如必须这样做，先明确记录 uncommitted diff。

## 大文件与结果纪律

不要提交：

- 模型权重；
- 数据集；
- 检索索引；
- checkpoint；
- wandb 目录；
- 大型日志；
- 全量 trajectory dump。

可以提交：

- 小型配置；
- 小型 fixtures；
- 指标摘要；
- 抽样轨迹；
- 图表；
- 报告文档。

README、报告、简历中的所有数字必须来自真实日志。没有完成实验前，只能写“计划”“待完成”“实验中”，不能写成已完成结论。

## 参考文档

- 完整研究大纲：`TRUST-R1_完整实验方案_v1.0_资源冻结版.md`
- 项目初始化设计：`docs/superpowers/specs/2026-07-02-trust-r1-project-bootstrap-design.md`
- AutoDL 工作流：`docs/autodl_workflow.md`
- 实验矩阵：`docs/experiment_matrix.md`
- 日志 schema：`docs/logging_schema.md`
