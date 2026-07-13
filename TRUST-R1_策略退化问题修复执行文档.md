# TRUST-R1 Pilot B0 策略退化诊断与修复执行文档

## 0. 文档定位

本执行文档只处理 **pilot B0（Clean baseline）** 在训练后发生策略退化的问题。

B0 的实际配置为：

```text
retrieval_fault.enabled=false
retrieval_fault.mode=clean
retrieval_fault.fault_rate=0.0
trust_reward.enabled=false
algorithm.adv_estimator=grpo
```

因此，本轮不得把根因归因于 TRUST reward、recovery reward、duplicate penalty 或 retrieval fault。当前任务是修复基础 Search-R1/GRPO 训练链路与 pilot 配置。

目标仓库：<https://github.com/lyesh0/TRUST-R1>

## 1. 已知故障

step 100 checkpoint 对不同问题持续输出类似内容：

```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
My previous action is invalid. If I want to search, I should put the query
between <search> and </search>...
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

主要表现：

1. 不再生成完整的 `<search>...</search>` 或 `<answer>...</answer>`。
2. 复制环境注入的 invalid-action 纠错文本。
3. 出现连续 `!` 等单 token 重复。
4. 无效动作会继续进入下一轮，增加 rollout 时间与生成长度。
5. step 100 checkpoint 已不可作为后续训练起点。

## 2. 已确认事实

### 2.1 B0 没有启用 TRUST 模块

当前 `train_4090.sh` 的 B0 override 已正确关闭 fault 和 trust reward。因此此次退化发生在基础 QA EM reward + Search-R1 agent loop 中。

### 2.2 invalid 重试机制来自上游 Search-R1

`search_r1/llm_agent/generation.py` 在动作无法解析时会把固定纠错文本放入下一轮 observation，并设置 `done=0`。该行为不是 TRUST-R1 新增代码，而是继承自上游。

这段机制只有在模型频繁输出 invalid 时才会成为复制闭环。因此它是**退化放大器**，不一定是最初触发器。

### 2.3 `qa_em.extract_solution()` 暂时不得修改

原始 prompt 中包含格式示例：

```text
For example, <answer> Beijing </answer>.
```

RewardManager 解码的字符串包含 prompt 与 response。上游要求至少出现两个 `<answer>`，是为了排除 prompt 中自带的示例。不得把逻辑简单改为“只要出现一个 `<answer>` 就接受”，否则 prompt 示例可能被误认为模型答案。

### 2.4 当前 mini/micro batch 已显式缩放

当前 4090 脚本实际设置：

```text
train_batch_size=32
n_agent=5
ppo_mini_batch_size=16
ppo_micro_batch_size=4
log_prob_micro_batch_size=8
```

训练 batch 经 `n_agent=5` 展开后为 160 条轨迹。`ppo_mini_batch_size=16` 并非大于 rollout batch，因此此前“仍使用默认 256”的判断不适用于当前脚本。

但与上游 `train_batch_size=512` 相比，当前每步只有 32 个独立问题组，而上游有 512 个。GRPO 在稀疏奖励下会因此获得更少的成功组和更高的梯度方差。这是参数缩放风险，不是简单的 shape 错误。

### 2.5 当前模型偏离上游默认 B0

当前脚本默认：

```text
Qwen2.5-3B-Instruct
```

上游 GRPO 默认：

```text
Qwen2.5-3B
```

数据处理代码也把现有 prompt 模板标记为适用于 base model。Instruct 模型拥有已有聊天对齐行为，可能自然语言作答、提前 EOS，或不遵守严格 XML 动作协议。B0 首先应恢复为上游默认 Base 模型。

## 3. 当前根因判断

### P0：B0 使用了未经该 prompt 验证的 Instruct 模型

严格 XML 工具协议与 Qwen2.5-3B-Instruct 的既有聊天行为可能冲突。模型一旦没有输出闭合标签，就会触发 invalid observation；随后开始复制固定纠错文本。

### P0：缺少 step-0 与短程行为验证

当前 pilot 直接运行较长训练，验证与保存间隔为 50 step。无法区分：

- 原始模型在 step 0 就不会输出合法动作；
- 训练在前 5～10 step 破坏了动作格式；
- 退化直到 step 50～100 才发生。

在未确认这三者之前，不能继续调正式参数。

### P1：独立 prompt group 从 512 缩至 32

上游每次更新使用 512 个问题组，每组 5 条轨迹；当前只有 32 个问题组。若大量组内 5 条轨迹都获得相同 reward，GRPO advantage 为零，真正提供学习信号的 group 可能很少。

当前 mini/micro batch 比例可以运行，但 `train_batch_size=32` 可能不足以稳定支撑稀疏 QA reward。

### P1：学习率未随 small-batch 稳定性调整

当前仍使用上游量级的：

```text
actor_lr=1e-6
```

在独立问题组缩小 16 倍后，梯度估计方差更大。`1e-6` 不一定错误，但 pilot 应先使用 `5e-7` 作为保守值，避免少数高优势轨迹快速改变 token 分布。

### P1：invalid observation 会放大已有失败

长纠错文本进入下一轮上下文，模型可能复制它；同时未闭合标签的输出会生成至 `max_response_length=512`。这会放大策略退化并浪费 rollout 时间。

第一轮应先修复模型与 pilot 流程；如果 Base 模型仍出现相同复制，再加入“连续 invalid 终止”保护。

## 4. 本轮修改范围

### 必须修改

1. 新增独立的 B0 pilot-safe 入口，或为 `train_4090.sh` 增加 `--pilot-safe` 模式。
2. B0 pilot-safe 默认模型改为 `Qwen2.5-3B` Base。
3. pilot-safe 先只运行 10 step。
4. 每 2 step 验证并保存 checkpoint。
5. 显式设置所有 batch、LR、采样和长度参数。
6. 记录 step-0、2、4、6、8、10 的生成样例与关键指标。
7. 增加配置自检，启动前打印有效配置并验证整除关系。

### 本轮禁止修改

1. 不修改 TRUST reward。
2. 不启用 retrieval fault。
3. 不修改 `qa_em.extract_solution()` 的双 answer 规则。
4. 不做格式 SFT。
5. 不直接删除上游 invalid 重试机制。
6. 不从 step 100 checkpoint 续训。
7. 不自动启动 M1、M2 或正式 100/300-step 训练。

## 5. Pilot-safe 推荐配置

第一轮使用以下配置：

```text
model=Qwen2.5-3B Base
train_data_num=10000
val_data_num=100
train_batch_size=32
val_batch_size=20
n_agent=5
ppo_mini_batch_size=16
ppo_micro_batch_size=4
log_prob_micro_batch_size=8
actor_lr=5e-7
lr_warmup_ratio=0.1
temperature=0.8
top_p=0.95
max_start_length=1024
max_prompt_length=2560
max_response_length=256
max_obs_length=384
max_turns=2
total_training_steps=10
save_freq=2
test_freq=2
trust_reward.enabled=false
retrieval_fault.enabled=false
```

说明：

- 保留当前已能运行的 `16/4/8` batch 配置，不再声称它存在 shape 错误。
- 将 Base 模型作为 B0 复现基线。
- LR 降为 `5e-7`，减少 small-batch 更新破坏。
- response length 降为 256，避免 invalid 轨迹持续生成 512 tokens。
- temperature 降为 0.8，但不得设为 0；GRPO 仍需要组内探索。

## 6. 配置自检要求

Agent 必须在训练启动前执行并保存以下检查结果：

```text
expanded_rollout_batch = train_batch_size * n_agent
```

按推荐配置：

```text
expanded_rollout_batch = 32 * 5 = 160
```

至少验证：

```text
train_batch_size % world_size == 0
expanded_rollout_batch % world_size == 0
ppo_mini_batch_size % world_size == 0
ppo_mini_batch_size >= ppo_micro_batch_size
ppo_mini_batch_size % ppo_micro_batch_size == 0
expanded_rollout_batch % ppo_mini_batch_size == 0
```

如果框架内部对 mini-batch 的定义是 per-rank 而不是 global，Agent 必须通过运行时打印或更新函数代码确认，不得仅凭变量名猜测。

同时打印：

- 最终模型路径和 `config.json` 中的模型类型；
- tokenizer 路径与 chat template；
- Hydra 完整 resolved config；
- Git commit SHA；
- 数据条数；
- world size；
- 每卡归一化后的 mini/micro batch；
- optimizer 实际 learning rate；
- `trust_reward.enabled` 与 `retrieval_fault.enabled`。

## 7. Step-0 基线检查

正式更新参数前，必须对原始 Base 模型运行验证或零更新 rollout。

至少保存 20 条完整轨迹，统计：

- `env/ratio_of_valid_action`；
- `env/number_of_valid_search`；
- `env/finish_ratio`；
- `response_length/mean`；
- `response_length/clip_ratio`；
- `val/test_score/nq`；
- 首轮 `<search>` 数量；
- 首轮 `<answer>` 数量；
- invalid 数量；
- 是否出现连续 `!`；
- 是否出现 `My previous action is invalid` 复制。

必须回答：

1. 退化文本在 step 0 是否已经出现？
2. 原始 Base 模型能否生成至少一部分完整动作？
3. step-0 的 5 个 rollout 是否有 reward 差异？

若 step 0 已经大面积退化，停止 RL，优先检查模型文件、tokenizer/chat template、加载路径和生成参数。

## 8. 10-step 执行流程

### 阶段 A：静态检查

1. 确认不使用 `scripts/resume_pilot.sh`。
2. 确认模型路径指向 Base，而不是 Instruct 或 step 100。
3. 输出 resolved config。
4. 运行配置整除断言。
5. 检查 retriever health。
6. 确认 B0 override 关闭 fault 与 trust reward。

### 阶段 B：step-0 验证

1. 不进行 optimizer update。
2. 保存固定 20 条样例。
3. 输出第 7 节指标。
4. 人工检查动作标签与重复文本。

### 阶段 C：2-step dry run

1. 运行 2 step。
2. 确认四卡没有 OOM、NaN 或 worker 退出。
3. 确认 step 2 checkpoint 能加载。
4. 对比 step 0 与 step 2 的有效动作率。

### 阶段 D：10-step smoke run

1. 从原始 Base 模型重新开始，不从 dry-run 或 step 100 续训。
2. 每 2 step 验证并保存。
3. 保存固定问题的响应变化。
4. 满足退化条件时立即停止。

## 9. 退化熔断条件

出现任一情况立即停止：

1. 连续两次验证 `env/ratio_of_valid_action < 0.05`；
2. `response_length/clip_ratio > 0.8`；
3. 固定样例中超过 80% 复制 invalid 纠错文本；
4. 固定样例中超过 50% 出现明显单 token 长重复；
5. reward max、mean、min 连续多次全部相同且有效动作率下降；
6. actor loss、KL、grad norm 出现 NaN/Inf；
7. step 2 相比 step 0 的有效动作率明显下降且 step 4 继续下降。

熔断后保存：

- 当前 checkpoint；
- resolved config；
- W&B run URL；
- 最近 20 条轨迹；
- reward 与 advantage 摘要；
- optimizer LR、grad norm、KL、entropy；
- `ACTIVE_TRAJ_NUM`。

## 10. 第一轮验收标准

10-step smoke run 通过需同时满足：

1. 没有大面积复制 invalid 纠错话术。
2. 没有大面积 `!` 单 token 重复。
3. `env/ratio_of_valid_action` 未持续下降到接近 0。
4. `response_length/clip_ratio < 0.5`。
5. 至少部分样例能生成完整 `<search>` 或 `<answer>`。
6. step 10 checkpoint 可正常加载和推理。
7. 无 NaN/Inf。
8. step 0、2、4、6、8、10 的样例与指标均已保存。

只有通过上述标准，才能运行 30-step pilot。30 step 通过后，才允许讨论 100 step。

## 11. Batch 扩展策略

如果 Base + 10-step 配置不退化，但有效奖励过于稀疏，再测试：

```text
train_batch_size=64
n_agent=5
expanded_rollout_batch=320
ppo_mini_batch_size=32
ppo_micro_batch_size=4
```

保持：

```text
expanded_rollout_batch / ppo_mini_batch_size = 10
```

与 batch 32 配置一致。Batch scaling 只用于增加独立 prompt group 数量，不应同时更换模型、LR 或 reward。

如果 batch 64 OOM，Agent 应区分：

- rollout KV cache OOM；
- FSDP actor update OOM；
- log-prob/ref 阶段 OOM。

再分别调整 vLLM memory utilization、micro-batch 或 offload，不得直接把所有参数同时缩小。

## 12. invalid 环境保护的后置方案

只有当 Base 模型、step-0 检查和 small-batch 稳定化后仍出现复制闭环，才修改 invalid 分支。

推荐最小保护：同一轨迹连续两次 invalid 后终止。第一次仍允许上游纠错：

```python
if action is None:
    invalid_action_stats[i] += 1
    if invalid_action_stats[i] >= 2:
        next_obs.append("")
        dones.append(1)
    else:
        next_obs.append("<invalid_action/>")
        dones.append(0)
```

该修改属于 B0 环境语义变更，必须单独 commit，并在实验报告中说明。不能把修改后的结果直接称为严格原版 Search-R1 baseline。

## 13. Agent 需要修改的文件

优先范围：

```text
scripts/train_4090.sh
新增 scripts/run_b0_pilot_safe.sh（推荐）
必要时新增配置自检脚本或测试
```

暂不修改：

```text
trust_r1/reward_adapter.py
trust_r1/rewards.py
verl/utils/reward_score/qa_em.py
search_r1/llm_agent/generation.py
```

只有在第一轮 10-step 仍触发 invalid 复制后，才进入第 12 节的环境保护修改。

## 14. Agent 交付要求

Agent 完成后必须返回：

1. 实际使用的原始 pilot 命令和 resolved config。
2. 退化发生前后模型路径确认。
3. B0 pilot-safe 脚本。
4. 配置整除与运行时 batch 打印结果。
5. step-0 轨迹与指标。
6. 2-step dry-run 结果。
7. 10-step smoke run 命令；未获人工确认前不得自动运行长实验。
8. 修改文件和关键 diff。
9. 尚未解决的风险。

Agent 不得：

- 继续运行 `resume_pilot.sh`；
- 从 global_step_100 续训；
- 自动启动 M1/M2；
- 自动跑 100/300/600 step；
- 修改双 answer 提取规则；
- 把当前问题归因于 TRUST reward。

## 15. Agent 最终汇报模板

```markdown
## B0 修复状态
- [ ] 确认原 pilot 实际配置
- [ ] Base 模型路径确认
- [ ] resolved config 保存
- [ ] batch 运行时检查
- [ ] step-0 基线
- [ ] 2-step dry run
- [ ] 10-step 命令准备

## 关键配置
- model:
- train_batch_size:
- expanded_rollout_batch:
- ppo_mini_batch_size:
- ppo_micro_batch_size:
- actor_lr:
- temperature:
- max_response_length:

## Step-0 指标
- valid_action_rate:
- valid_search_mean:
- finish_ratio:
- response_length_mean:
- response_length_clip_ratio:
- val_score:

## 修改文件
- path: purpose

## 测试与结果
- command:
- result:

## 风险与下一步
- ...
```

## 16. 最终判断

当前 B0 退化不能归因于 TRUST reward。当前 4090 脚本的 mini/micro batch 已经过缩放，不能再把“默认 256 未调整”作为已确认根因。

现阶段最需要修复的是：**B0 使用了偏离上游默认配置的 Qwen2.5-3B-Instruct；训练前没有 step-0 行为基线；small-batch 条件下仍以较长 pilot、较稀疏验证运行，导致动作格式退化没有被及时发现和停止。**

本轮先恢复 Base 模型并完成 10-step 可观测 pilot。只有短程运行稳定后，才能继续扩大 batch 或训练步数。
