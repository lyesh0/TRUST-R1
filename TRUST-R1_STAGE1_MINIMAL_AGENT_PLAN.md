# TRUST-R1 第一阶段最小实现与 250 步实验执行文档（V2.1）

> 面向执行 Agent；已按仓库 `lyesh0/TRUST-R1` 当前提交
> `dd4ccc9a3d624b5491aefdcff41bf75d9a6db31d` 校准。正式实现必须从该提交或其后继提交创建 `codex/stage1-local-advantage` 独立分支。

## 0. 最终交付目标

在不重构 Search-R1/veRL、不启用检索故障、不训练奖励模型的前提下，完成下面一条唯一实验链：

```text
Qwen2.5-3B Base
  -> 50 step 动作格式 LoRA-SFT
  -> 合并为统一冷启动模型 C0
  -> S1-B0：100 step GRPO，仅最终答案奖励
  -> S1-B1：100 step GRPO，最终答案奖励 + query-token 局部检索进展 advantage
  -> 固定验证集对比
```

本轮只回答：

> 在相同 C0、问题顺序、检索器和训练预算下，把“本次搜索是否首次召回包含 gold answer alias 的证据”经过同问题组内标准化后，仅分配给该次 query 内容 token，是否能改善小模型的检索质量？

本轮是单 seed、低预算的机制预实验。即使结果为正，也只能称为“方向性证据”，不能直接宣称方法普遍有效。

本文后续的 B0/B1 均指 `S1-B0/S1-B1`，不得与旧实验矩阵中的 clean baseline / fault augmentation B0/B1 混用；run ID、目录名和图表标签必须使用完整的 `S1-` 前缀。

---

## 1. 仓库现状与必须规避的问题

### 1.1 可直接复用

- RL 主入口：`verl/trainer/main_ppo.py`
- GRPO advantage：`verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`
- 训练控制器：`verl/trainer/ppo/ray_trainer.py`
- actor loss：`verl/workers/actor/dp_actor.py`
- 多轮搜索：`search_r1/llm_agent/generation.py`
- 检索器：现有 E5 + FAISS 服务，`topk=3`
- 信息 token 屏蔽：`info_mask/loss_mask` 已实现，检索文档不会参与 actor loss
- 现有动作、搜索次数和 finish 指标可继续使用

### 1.2 不能直接复用

1. `trust_r1/rewards.py` 和 `reward_adapter.py` 产生的是轨迹级标量总奖励；它会通过 GRPO 扩散到整段 response，不符合“只更新 query token”的假设。
2. 旧脚本中的 `B1` 表示 fault augmentation，与本轮 B1 含义冲突。不要复用旧实验矩阵；新建独立的 Stage1 脚本，并使用 `S1-B0/S1-B1` 标识。
3. 当前分支不包含另一条历史线上的 `scripts/train_4090.sh`；不要为复用旧参数而 cherry-pick 那套脚本，Stage1 使用独立启动器。
4. `generation.py` 仍返回长篇 invalid 提示，正是此前重复输出坍缩的模仿源。
5. 当前 SFT trainer 引用了仓库中不存在的 `SFTDataset`，不要在本轮修复整套旧 SFT trainer。
6. 当前轨迹写入依赖 `trust_reward.enabled=true`，导致 S1-B0 无法得到完整诊断日志；必须把日志与旧 trust reward 解耦。
7. 当前 trainer 从 `global_steps=1` 开始更新，`trainer.total_training_steps=N` 实际只完成 `N-1` 次更新。Stage1 不重写计数主循环；100 次更新统一传 `trainer.total_training_steps=101`，并避免结束时重复记录 step101 验证。
8. 当前提交已经包含 PPO/`low_var_kl` exp clamp、masked NaN 隔离、FSDP 同步跳过、GRPO group-size 校验和非有限梯度保护。Stage1 只增加正式实验专用的 `abort_on_non_finite=true` 与控制器侧诊断落盘，不能重复实现或撤销现有保护。
9. 当前保护默认遇到异常后跳过更新继续训练；Stage1 正式实验要求所有 rank 同步跳过坏 step、控制器写诊断包，然后抛出异常终止 run。
10. prompt 本身含 `<search> query </search>` 和 `<answer>...` 示例。过程奖励与 query span 解析必须只看 `valid_response_ids`，不能解析完整 `prompt + response`。
11. `trust_r1_rollout_traces` 当前只留在 `meta_info`，reward/advantage 路径只能看到摘要；Stage1 必须把完整 search trace 按样本写入 `non_tensor_batch`，用于实际执行 query 与 token span 的交叉校验。

---

## 2. 本轮冻结决策

### 2.1 数据集：使用 NQ，不切 HotpotQA

最低改动版使用当前已准备的 NQ search parquet。原因：现有 NQ 数据、训练入口和 answer reward 已跑通；HotpotQA 的 supporting facts 尚未写入当前 `ground_truth`。本轮证据判定因此限定为：

```text
检索返回的 <information> 块中，是否出现规范化后的 gold answer alias 完整词序列。
```

不能把本轮指标写成 supporting-fact recall。若 S1-B1 有方向性收益，下一阶段再迁移到 HotpotQA 并加入 supporting facts。

### 2.2 固定数据

- RL train：从 NQ train 固定抽取并保存 3200 个问题，顺序固定。
- RL validation：固定 100 个问题。
- 人工诊断集：从 validation 中固定 50 个问题。
- SFT：从 RL train 范围内构造 800 条，不使用 validation。
- 所有 ID、源 parquet 路径/hash、seed、top-k 和 Type B 命中统计写入 `artifacts/stage1/data_manifest.json`；构造完成后先提交该小型 manifest，再进行正式 SFT/RL。
- RL 设置 `data.shuffle_train_dataloader=false`，S1-B0/S1-B1 读取完全相同的 parquet 顺序。

数据脚本在 `/root/autodl-tmp/TRUST-R1-stage1/data/` 写出 `rl_train.parquet`、`rl_validation.parquet`、`diagnostic_50.jsonl` 和 `sft_train.jsonl`。manifest 保存这些文件的 SHA256；启动器校验 hash，不重新抽样。

不需要 10000 条训练数据；100 step × 32 prompt 正好消费 3200 个问题。

### 2.3 模型与硬件

- 起点：`/root/autodl-tmp/models/Qwen2.5-3B`，必须是 Base，不是 Instruct。
- 资源：4 × RTX 4090 24GB。
- retriever：`http://127.0.0.1:8000/retrieve`。
- index/corpus/model/top-k 沿用当前 E5 + `wiki-18` + FAISS Flat 配置。

### 2.4 运行与依赖约束

- Python 代码保持 Python 3.9 兼容。
- rollout/actor 使用 BF16，FSDP reduce 保持 FP32。
- Stage1 新增的唯一训练依赖是 `peft==0.14.0`；不得顺带升级 torch、transformers、vLLM、Ray 或 veRL。
- 所有数据构造、SFT、RL 和评估命令只在 AutoDL 运行；本地 Mac 只做代码、单元测试和静态检查。
- 正式 run 必须来自 clean git commit；生成数据、模型、checkpoint 和全量轨迹不得进入 git。

---

## 3. 最小代码改动范围

### 3.1 新增文件

| 文件 | 唯一职责 |
|---|---|
| `trust_r1/process_reward.py` | answer alias 词边界命中、首次命中、query token span、组内标准化、长度归一化 |
| `scripts/data_process/build_stage1_data.py` | 固定 3200/100/50 ID，并构造 800 条 SFT 数据 |
| `scripts/sft_action_lora.py` | BF16 LoRA SFT、仅 assistant target 计算 loss、保存 step25/50、合并选定 adapter 为 C0 |
| `scripts/eval_action_format.py` | 评估 Base、SFT-step25、SFT-step50 的动作格式能力 |
| `scripts/run_stage1_local_reward.sh` | 只负责 S1-B0/S1-B1 的 smoke、train、eval 命令和冻结配置 |
| `scripts/analyze_stage1.py` | 从验证轨迹计算 EM/F1、召回、利用率、搜索次数和四象限 |
| `tests/test_process_reward.py` | 过程奖励核心单元测试 |
| `tests/test_stage1_data.py` | 固定抽样、split 隔离、SFT 数量和 Type B 命中门槛 |
| `tests/test_stage1_sft.py` | assistant-only label mask、左截断和 C0 目录校验 |

### 3.2 修改文件

| 文件 | 最小修改 |
|---|---|
| `search_r1/llm_agent/generation.py` | invalid observation 改为 `<tool_error code="INVALID_ACTION"/>`；完整 rollout trace、invalid count 和 finish reason 进入 non-tensor batch |
| `verl/trainer/main_ppo.py` | 从 response-only 构造过程诊断 tensor；轨迹写入与旧 trust reward 解耦 |
| `verl/trainer/ppo/ray_trainer.py` | GRPO answer advantage 后叠加 query-local advantage；固定 seed；汇总过程指标；异常时写诊断包并终止 |
| `verl/workers/actor/dp_actor.py` | 复用现有同步熔断，增加可配置的正式实验中止模式 |
| `verl/trainer/config/ppo_trainer.yaml` | 新增 `process_reward`、`abort_on_non_finite` 配置 |
| `requirements.txt` / `pyproject.toml` | 只增加固定版本 `peft==0.14.0`，保持两处依赖一致 |
| 现有 rollout/numerical tests | 增加完整 trace 传播、日志解耦和多 rank abort 行为覆盖 |

不要修改旧 `trust_reward` 的含义，不删除 fault 相关代码，不改 `main_ppo_format.py`，不做无关重构。

`process_reward.py` 的最小公共接口冻结为：

```python
@dataclass
class ProcessFeatures:
    step_rewards: torch.Tensor   # [max_search_steps], 0/1
    query_step_ids: torch.Tensor # [response_length], 0=非 query，1..N=搜索步
    evidence_hits: list[bool]
    queries: list[str]
    information_blocks: list[str]
    alignment_valid: bool

def build_process_features(
    response_token_ids,
    valid_response_length,
    gold_aliases,
    rollout_trace,
    tokenizer,
    max_search_steps=2,
) -> ProcessFeatures: ...

def add_query_local_advantage(
    answer_advantages,
    step_rewards,
    query_step_ids,
    uids,
    loss_mask,
    weight=0.2,
    z_clip=2.0,
): ...
```

纯文本匹配、span 定位和标准化均放在该模块；trainer 只负责数据搬运、调用和指标汇总。

---

## 4. 阶段 A：50 步动作格式 SFT

### 4.1 数据构成

800 条数据恰好对应 effective batch 16 × 50 step：

| 类型 | 数量 | 目标 |
|---|---:|---|
| A：搜索动作 | 400 | 学会 `<think>...</think><search>...</search>` |
| B：证据充分后回答 | 160 | 学会 `<think>...</think><answer>...</answer>` 并正常终止 |
| C：invalid 恢复 | 240 | 看到结构化 tool error 后重新给合法动作 |

SFT 必须匹配当前多轮运行时格式，不能只训练裸 `<search>`：

```text
Type A target:
<think>I need external evidence.</think><search>{query}</search>

Type B prompt suffix:
<think>...</think><search>...</search>
<information>{gold-containing retrieved passage}</information>

Type B target:
<think>The evidence is sufficient.</think><answer>{gold_answer}</answer>

Type C prompt suffix:
{one invalid model action}
<tool_error code="INVALID_ACTION"/>

Type C target:
<think>I should issue a valid action.</think><search>{query}</search>
```

第一版 query 可使用去掉问号的原问题文本。SFT 的任务是动作冷启动，不用 SFT 证明 query 质量。Type B 只保留当前 retriever top-3 中确实存在 gold alias 完整词序列的样本。

构造顺序固定为：先从 3200 个 RL train ID 中选择 Type B 命中样本，再从未使用 ID 中选择 Type A/C；三类 question ID 不重叠。若 top-3 命中样本不足 160 条，数据脚本必须失败并输出实际命中数，不得放宽匹配规则。超过 `max_length` 时只从 prompt 左侧截断，必须完整保留 tool error、最近 information 和全部 target token。

### 4.2 SFT 配置

为避免修复旧 FSDP SFT trainer，使用 Transformers + PEFT：

```yaml
base_model: /root/autodl-tmp/models/Qwen2.5-3B
method: LoRA BF16
max_length: 1024
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
world_size: 4
effective_batch_size: 16
max_steps: 50
learning_rate: 1.0e-4
warmup_steps: 5
lora_rank: 8
lora_alpha: 16
target_modules: [q_proj, v_proj]
gradient_checkpointing: true
gradient_clip: 1.0
save_steps: 25
seed: 42
```

只对 target/assistant token 计算交叉熵；prompt、历史模型动作、检索结果和 tool error 的 label 全部设为 `-100`。

分别评估 Base、step25、step50。C0 选择规则是“最早通过全部门槛”：step25 通过则固定使用 step25；否则检查 step50；step50 通过才使用 step50。不得根据后续 RL 结果反选 C0。合并 LoRA 后保存为完整 Hugging Face 模型：

```text
/root/autodl-tmp/TRUST-R1-stage1/checkpoints/C0
```

C0 必须能被 `AutoModelForCausalLM.from_pretrained(C0)` 和 vLLM 直接加载，不能只是 adapter 目录。

`eval_action_format.py` 必须写 `artifacts/stage1/c0_selection.json`，记录 Base/step25/step50 全部门槛指标、选择理由、adapter checkpoint、合并后 C0 的 `config.json` hash 和 tokenizer hash。该小型结果文件须在正式 RL 前提交；模型权重不提交。

### 4.3 C0 门槛

| 指标 | 门槛 |
|---|---:|
| Valid Action Ratio | ≥95% |
| 非空 query 比例 | ≥98% |
| invalid 后单步恢复率 | ≥80% |
| 长错误提示复述率 | ≤2% |
| evidence-state 正常回答率 | ≥85% |
| 连续重复字符/动作率 | ≤5% |

若两个 checkpoint 都未通过全部门槛，禁止进入正式 RL，先检查 SFT mask、截断策略和数据格式；`valid action <90%` 视为硬失败，不得通过放宽其他门槛继续。

---

## 5. 阶段 B：S1-B0 最终答案奖励基线

S1-B0 使用现有 `qa_em.compute_score_em` 作为最终答案奖励，保持其行为不变：

```text
process_reward.enabled=false
trust_reward.enabled=false
retrieval_fault.enabled=false
```

不加 format reward、搜索奖励、重复惩罚、检索成本或证据利用奖励。

---

## 6. 阶段 C：S1-B1 query-token 局部检索进展信号

### 6.1 证据命中

对 response-only 中第 `t` 个 `<information>` 块，使用 gold aliases 判断 `hit_t`。匹配必须基于规范化后的 token 序列窗口，不允许裸 substring：

```text
normalize: lowercase -> remove punctuation -> remove a/an/the -> split whitespace
hit: answer_tokens 是 information_tokens 的连续完整子序列
```

例如 alias `US` 不能命中 `business`。

`gold aliases` 来自 `ground_truth["target"]`，字符串先转成单元素列表；规范化后为空的 alias 忽略。只解析完整配对的 `<information>...</information>`；缺失闭标签时该步记为未命中并增加 parse-error 指标。

### 6.2 局部奖励

```text
r_t = 1，若第 t 次检索命中且此前所有检索都未命中
r_t = 0，其他情况
```

没有负奖励。已经命中后再次命中为 0；没有新增有效证据为 0。

这里“没有负奖励”只指原始 `r_t`；组内标准化后的 `z_t` 在 informative group 中允许为负，这是 GRPO 相对比较信号，不是额外设计的失败惩罚。

### 6.3 组内标准化

以 `uid = question index` 分组。对同一问题的 4 条 rollout，在相同搜索步分别计算：

```text
z_t = clip((r_t - group_mean_t) / (group_std_t + 1e-6), -2, 2)
```

只让实际执行第 `t` 次搜索且存在完整 query token span 的轨迹参与该步统计；没有执行该步的轨迹不作为 `r_t=0` 参与。参与数少于 2，或参与轨迹全 0/全 1 时，全部 `z_t=0`。标准差使用 sample std（`unbiased=True`）；不得跨问题、跨搜索步标准化。

### 6.4 只作用于 query 内容 token

先运行原有 `compute_grpo_outcome_advantage` 得到 `A_answer`，再执行：

```text
A[token] = A_answer[token] + 0.2 * z_t / max(query_token_count_t, 1)
```

仅当 token 位于第 `t` 个 `<search>` 与 `</search>` 之间时叠加；标签 token、think token、answer token、information token均不得收到局部项。

关键限制：

- 不把 `r_t` 写入 `token_level_scores` 或最终 answer reward；否则它会再次被 GRPO 扩散到全 response。
- query span 从 response token IDs 中按开闭标签 token 子序列定位，不能对完整 prompt 做正则后猜字符偏移。
- 只接受完整配对且内容非空的 query span；按出现顺序与 `trust_r1_rollout_traces` 中实际执行的 query 逐一核对规范化文本。数量或文本不一致时 `alignment_valid=false`，该轨迹局部项清零；smoke 或正式 S1-B1 发现任何 alignment error 都立即中止。
- 局部项还必须与 `loss_mask` 相交，保证 information/padding token 始终为 0。
- `A_answer`、answer score 和 S1-B0 完全一致；S1-B1 只多一次 advantage 局部叠加。

建议在 batch 中暂存：

```text
process_step_rewards: [batch, max_search_calls]
query_step_ids:       [batch, response_length]  # 0=非query，1/2=搜索步
local_advantages:     [batch, response_length]
```

以上三个 tensor 在 S1-B0/S1-B1 都构造，以保证诊断口径相同；S1-B0 强制 `local_advantages.zero_()`，S1-B1 才调用组内标准化并叠加到最终 `advantages`。过程 tensor 必须在 `_balance_batch()` 完成后，基于已经重排且 UID 对齐的 batch 构造。

训练数据流固定为：`_balance_batch()` → `RewardManager` 计算原始 QA score并原地附加 `process_step_rewards/query_step_ids` → 原有 GRPO 生成 `A_answer` → `ray_trainer.compute_advantage` 调用 `add_query_local_advantage` → 写入最终 `advantages/local_advantages` → Actor update。验证只运行特征与指标构造，不计算或叠加 local advantage。

---

## 7. S1-B0/S1-B1 冻结训练配置

```yaml
model_checkpoint: /root/autodl-tmp/TRUST-R1-stage1/checkpoints/C0
dataset: NQ-stage1-fixed
train_questions: 3200
validation_questions: 100
seed: 42
shuffle_train_dataloader: false

trainer_steps: 100
trainer_total_training_steps: 101  # 当前循环语义下得到 100 次实际更新
train_batch_size: 32
group_size_n_agent: 4
ppo_epochs: 1
ppo_mini_batch_size: 32
ppo_micro_batch_size: 8
rollout_log_prob_micro_batch_size: 16
ref_log_prob_micro_batch_size: 16

actor_learning_rate: 2.0e-7
lr_warmup_ratio: 0.03
kl_loss_type: low_var_kl
kl_loss_coef: 0.001
gradient_clip: 1.0
clip_ratio: 0.2
entropy_coeff: 0.001
precision: bfloat16

max_prompt_length: 2560
max_start_length: 1024
max_response_length: 512
max_obs_length: 384
max_turns: 2
retrieval_topk: 3

rollout_temperature: 1.0
rollout_top_p: 0.95
rollout_tensor_model_parallel_size: 1
vllm_gpu_memory_utilization: 0.45

actor_param_offload: false
actor_grad_offload: false
actor_optimizer_offload: true
ref_param_offload: true
state_masking: true

save_freq: 25
test_freq: 25
total_epochs: 1
```

batch 语义固定为：每个 trainer step 有 `32 × 4 = 128` 条 Actor 轨迹；global mini=32、global micro=8；4 卡下每卡 mini=8、micro=2、梯度累积 4 次，每个 trainer step 执行 `128 / 32 = 4` 次 `optimizer.step()`。因此每个 100-step RL run 有 400 次内部 optimizer step；文档标题中的 250 步按 `50 SFT step + 100 S1-B0 trainer step + 100 S1-B1 trainer step` 计。micro 只控制显存和累积开销，更新方差由 global mini=32 决定。

`max_turns=2` 表示最多执行两次真实 retriever 请求，之后只允许模型完成最终回答。

S1-B0/S1-B1 只能有以下差异：

```text
S1-B0: process_reward.enabled=false
S1-B1: process_reward.enabled=true
```

其余 `process_reward` 字段两组完全相同：

```yaml
process_reward:
  compute_diagnostics: true
  weight: 0.2
  z_clip: 2.0
  max_search_steps: 2
  abort_on_alignment_error: true

actor_rollout_ref:
  actor:
    abort_on_non_finite: true
```

Stage1 启动器接口冻结为：

```text
bash scripts/run_stage1_local_reward.sh MODE EXPERIMENT
MODE       = smoke | train | eval
EXPERIMENT = S1-B0 | S1-B1
```

- `smoke` 固定 2 个 trainer step，向 trainer 传 `total_training_steps=3`，禁用 checkpoint，保留完整轨迹和诊断。
- `train` 固定 100 个 trainer step，传 `total_training_steps=101`，保存/验证频率均为 25。
- `eval` 只加载显式指定的 C0 或 step25/50/75/100 checkpoint，不允许自动选择“最新”目录。
- `S1-B0/S1-B1` 都必须从同一个 C0 启动，不允许从彼此 checkpoint resume。
- 启动器必须拒绝其他实验名、Instruct 模型、缺失 manifest、dirty worktree 和非 4 卡环境；运行前写 resolved overrides、commit、依赖版本、`df -h`、GPU 型号和 batch 语义到 preflight 文件。
- 正式 logger 使用 `['console','wandb']`；run ID 包含实验名、seed 和 commit 短 hash。

---

## 8. 数值稳定性与正式训练前置门槛

### 8.1 NaN 防护

保留当前提交已有的 input/logprob/loss/grad norm 检查和 FSDP rank 同步。在 actor 的每个 microbatch backward 前检查 `policy_loss`，在 optimizer step 前检查全局 `grad_norm`。Stage1 开启 `abort_on_non_finite=true` 后，任一 rank 发现非有限值：

1. 不执行 `optimizer.step()`；
2. 清空梯度；
3. 所有 rank 在相同 collective 边界抛出包含阶段名的 `FloatingPointError`；
4. `ray_trainer` 捕获 worker 异常，在 `$RUN_DIR/diagnostics/non_finite_step_<N>/` 写入 `summary.json` 和 `rollouts.jsonl`，随后重新抛出并中止 run；
5. 诊断至少保存 git commit、配置、step、question ID/UID、trace summary、process tensor 摘要、answer score、response-only 解码文本和最后一个健康 checkpoint 路径。

不得在异常后保存 emergency checkpoint，也不得从已经执行 NaN optimizer step 的模型继续训练。只能使用异常前按 `save_freq=25` 保存且指标有限的 checkpoint；正式对照实验默认从共同 C0 重跑。

### 8.2 必跑 smoke test

正式 100 步前分别运行 S1-B0/S1-B1 各 2 步；smoke 不计入 250 步正式预算。必须验证：

- S1-B0/S1-B1 的 base answer score 在同一批静态样例上完全相同；
- S1-B0 `local_advantages` 全 0；
- S1-B1 在人工构造的“同组部分命中”batch 上只有 query 内容 token 非零；
- 同组全 0/全 1 时局部 advantage 全 0；
- 缺失第二次搜索的轨迹不参与 step2 的 mean/std，且没有 query token 收到局部项；
- query 长度归一化后，等价动作不会因 query 更长获得更大局部梯度总量；
- query token span 与完整 rollout trace 数量和规范化文本完全一致；
- 结构化 tool error 能在真实多轮 loop 中出现；
- 人工注入 NaN 时所有 FSDP rank 同步中止、没有执行 optimizer step，并生成诊断目录；
- 所有原有 pytest 与新增测试通过。

如果 S1-B1 正式训练前 10 步的 `process/informative_group_rate` 始终为 0，立即停止。这说明证据检测、span 对齐或检索命中存在错误，继续训练无法验证假设。

### 8.3 CPU 单元测试验收

- answer alias 使用完整规范化 token 窗口，覆盖大小写、标点、冠词、空 alias 和 `US`/`business` 反例。
- first-hit 序列覆盖 `[0,1]→[0,1]`、`[1,1]→[1,0]`、`[0,0]→[0,0]`。
- query span 只标内容 token，prompt 中示例标签、开闭标签、think/information/answer/padding 均为 0。
- span 与 trace 数量/文本不一致时 fail-fast；完整多轮样例能正确对应 step1/step2。
- UID 分组覆盖 mixed、全 0、全 1、仅一个参与者、缺失第二步和 batch 重排。
- 局部 advantage 与 loss mask 相交，长度归一化后每个 query 的局部 advantage 总和等于 `weight × z_t`（浮点容差 `1e-6`）。
- S1-B0 的局部 tensor 全 0，S1-B1 只改变 query 内容 token；answer advantage 和 answer score 逐元素一致。
- masked non-finite padding 不污染 loss；人工坏 loss/grad 在 optimizer step 前中止。
- 数据 manifest 同 seed 字节级可复现，train/validation/diagnostic split 无交集，SFT 三类数量精确为 400/160/240。
- trajectory logging 在 `trust_reward.enabled=false` 时仍写入完整 Stage1 schema。

---

## 9. 日志和评估

### 9.1 每个训练 step 记录

```text
actor/pg_loss
actor/kl_loss
actor/ppo_kl
actor/entropy_loss
actor/grad_norm
actor/update_skipped
actor/ppo_log_ratio_abs_max
actor/ppo_log_ratio_clamp_frac
actor/ref_log_ratio_abs_max
actor/ref_log_ratio_clamp_frac
env/ratio_of_valid_action
env/finish_ratio
env/number_of_valid_search
trust_r1/search_count_mean
trust_r1/duplicate_query_rate
process/raw_hit_rate
process/first_hit_rate
process/informative_group_rate
process/nonzero_query_token_rate
process/local_adv_abs_mean
process/span_alignment_error_count
answer/em
```

训练指标分母冻结如下：`raw_hit_rate` 为所有实际搜索步中的 `hit_t=1` 比例；`first_hit_rate` 为所有实际搜索步中的 `r_t=1` 比例；`informative_group_rate` 为参与数至少 2 的 `(uid, search_step)` 组中同时含 0/1 的组比例；`nonzero_query_token_rate` 为有效 query 内容 token 中局部 advantage 非零的比例；`local_adv_abs_mean` 只在有效 query 内容 token 上取绝对值均值；`span_alignment_error_count` 是轨迹数而非 token 数；`answer/em` 是原始 QA EM score 的 batch mean。

### 9.2 固定验证节点

评估 `step 0/25/50/75/100`。step0 是 C0，不重复保存模型。验证集使用现有 `do_sample=false` greedy 路径，S1-B0/S1-B1 完全相同。step100 已完成周期验证时禁止再以 step101 重复验证；若最终 step 不落在 `test_freq` 上，则以最后完成的 trainer step 编号补一次验证。

每次验证计算：

- Valid Action Ratio
- Finish Ratio
- Exact Match、token F1
- First-search Success
- Any-search Success / Evidence Recall@3
- 第二次搜索首次带来证据的 Incremental Evidence Rate
- Average Search Count
- Repeated Query Rate（第一版只做规范化后精确重复）
- Evidence Utilization：`P(answer correct | evidence hit)`
- Retrieved-but-Wrong：`P(answer wrong | evidence hit)`
- 四象限：无证据错误 / 有证据错误 / 无证据正确 / 有证据正确

验证指标分母冻结为 100 条 validation trajectory：First-search Success=`hit_1` 比例；Any-search Success/Evidence Recall@3=`any(hit_t)` 比例；Average Search Count 按轨迹平均；Repeated Query Rate 为至少一次规范化精确重复的轨迹比例。Incremental Evidence Rate 只在实际执行第二次搜索的轨迹中计算 `not hit_1 and hit_2`；Evidence Utilization 和 Retrieved-but-Wrong 分别在 `any(hit_t)` 子集内计算正确/错误条件概率。四象限同时输出 count 和除以 100 的 rate。

训练轨迹每 step 抽样 32 条即可；固定验证集 100 条全部写 JSONL。轨迹至少包含：

```json
{
  "question_id": "",
  "prompt": "",
  "gold_aliases": [],
  "rollout_id": 0,
  "queries": [],
  "information_blocks": [],
  "evidence_hit_by_step": [],
  "first_hit_reward_by_step": [],
  "local_z_by_step": [],
  "final_answer": "",
  "answer_correct": false,
  "valid_action": true,
  "invalid_action_count": 0,
  "finish_reason": "",
  "search_count": 0
}
```

日志解析必须使用 response-only 文本，避免把 prompt 中的标签示例当成模型动作。

`valid_action` 定义为该轨迹所有已执行动作均合法；同时单独记录 `invalid_action_count`。`finish_reason` 只允许 `answer`（出现合法 answer 动作）或 `max_turns`（最终仍未合法回答）；不得根据生成文本事后猜测其他枚举值。

轨迹写入使用现有 `trust_r1_logging` 配置名，但调用位置移出 `trust_reward.enabled` 分支：只要 `trust_r1_logging.enabled=true` 且 `write_trajectories=true` 就写。训练每个 trainer step 按 batch 顺序写前 32 条到 `train_trajectories.jsonl`；验证 100 条全部写到按 step 分开的 `validation_step_<N>.jsonl`。S1-B0/S1-B1 使用完全相同 schema，不允许 S1-B0 因未启用旧 trust reward 而缺字段。

---

## 10. 执行顺序

```bash
# 0. 记录代码状态并检查 retriever
git rev-parse HEAD
curl -fsS http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["Apollo 11 first person on the Moon"],"topk":3,"return_scores":true}'

# 1. 固定数据与构造 SFT
python scripts/data_process/build_stage1_data.py build \
  --source-dir /root/autodl-tmp/data/nq_search \
  --output-dir /root/autodl-tmp/TRUST-R1-stage1/data \
  --manifest artifacts/stage1/data_manifest.json \
  --retriever-url http://127.0.0.1:8000/retrieve \
  --seed 42

# 1b. 校验 manifest 和数据 hash；将小型 manifest 提交并推送后再继续
python scripts/data_process/build_stage1_data.py verify \
  --manifest artifacts/stage1/data_manifest.json \
  --data-dir /root/autodl-tmp/TRUST-R1-stage1/data

# 2. 50-step SFT，保存 25/50
torchrun --nproc_per_node=4 scripts/sft_action_lora.py train \
  --base-model /root/autodl-tmp/models/Qwen2.5-3B \
  --data /root/autodl-tmp/TRUST-R1-stage1/data/sft_train.jsonl \
  --output /root/autodl-tmp/TRUST-R1-stage1/checkpoints/sft_action

# 3. 评估并合并选定 SFT checkpoint 为 C0
python scripts/eval_action_format.py --all-candidates
python scripts/sft_action_lora.py merge --checkpoint <selected-step> \
  --output /root/autodl-tmp/TRUST-R1-stage1/checkpoints/C0

# 3b. 验证 C0 可由 Transformers/vLLM 加载，并提交 artifacts/stage1/c0_selection.json

# 4. 单元测试与 2-step smoke
pytest -q
bash scripts/run_stage1_local_reward.sh smoke S1-B0
bash scripts/run_stage1_local_reward.sh smoke S1-B1

# 5. 正式实验；两者都从完全相同 C0 启动
bash scripts/run_stage1_local_reward.sh train S1-B0
bash scripts/run_stage1_local_reward.sh train S1-B1

# 6. 汇总
python scripts/analyze_stage1.py \
  --b0 /root/autodl-tmp/TRUST-R1-stage1/runs/S1-B0 \
  --b1 /root/autodl-tmp/TRUST-R1-stage1/runs/S1-B1
```

Agent 应根据最终脚本参数名修正命令，但不得改变冻结实验值。

---

## 11. 停止规则

立即停止当前 run：

- loss、KL、log prob 或 grad norm 出现 NaN/Inf；
- 连续出现大段同一字符；
- search count 从正常值突然归零；
- Valid Action Ratio 相比 C0 下降超过 10 个百分点；
- S1-B1 局部 advantage 出现在非 query token；
- S1-B0 存在非零局部 advantage；
- query span/trace 对齐错误、question group 对齐错误或 S1-B0/S1-B1 数据顺序不一致。

S1-B1 到 step50 时若同时满足以下三项，可提前停止并保留失败结果：

```text
Evidence Recall 不高于 S1-B0-step50
平均搜索次数比 S1-B0 高 > 0.3
EM/F1 不高于 S1-B0
```

---

## 12. 第一阶段成功标准

S1-B1 视为有初步方向性效果，需同时满足：

1. step100 的 Any-search Success / Evidence Recall@3 比 S1-B0 高至少 3 个百分点，或 step25/50/75/100 至少三个节点持续更高；5 个百分点只作为强效果标记，不作为额外门槛；
2. `S1-B1 average_search_count - S1-B0 average_search_count <= 0.3`；
3. Valid Action Ratio 下降不超过 2 个百分点；
4. Finish Ratio 下降不超过 3 个百分点；
5. 全程无 NaN、无重复字符坍缩；
6. 全 run 的 eligible `(uid, step)` 中 `process/informative_group_rate >= 1%` 且 informative group 累计不少于 20 个，证明 S1-B1 实际收到过可区分的局部信号。

若召回提升但 Retrieved-but-Wrong 同时增加，结论写成：

> 局部信号改善了搜索，但瓶颈转移到证据利用。

若只增加搜索次数而召回、EM 不变，则判定为奖励投机，不调大 `weight`，下一阶段改进证据判定或信息增益定义。

---

## 13. 本轮明确不做

- 不使用 Hotpot supporting facts；
- 不训练独立 PRM；
- 不启用 retrieval fault；
- 不使用旧 recovery reward/duplicate penalty；
- 不加 format reward、搜索成本或负奖励；
- 不测试多个 λ、模型、数据集、retriever 或 seed；
- 不迁移框架，不升级核心依赖；
- 不修复与本实验无关的旧 SFT/Fault 实验体系；
- 不把 100 条验证集上的单 seed 改进写成统计显著结论。

---

## 14. Agent 完成定义

Agent 交付时必须提供：

1. 修改文件清单和每处修改目的；
2. `pytest -q` 结果；
3. 数据 manifest、C0 选择结果与 SFT 三个评估节点；
4. S1-B0/S1-B1 的完整冻结配置与启动命令；
5. smoke test 中“局部信号只落在 query token”的可核验证据；
6. step0/25/50/75/100 指标表；
7. 四象限统计和至少 20 条代表性轨迹；
8. 最终结论：成功 / 部分成功 / 失败；
9. 实验对应 git commit。正式训练前必须提交代码，禁止用未记录 diff 跑正式 100 步。
