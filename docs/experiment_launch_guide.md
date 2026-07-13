# TRUST-R1 实验启动指南

本文档只说明 **实验如何启动**。本地 Mac 只用于编辑代码和文档；所有涉及模型、数据、retriever、Ray/vLLM、RL 训练、评测和 checkpoint 的命令都必须在 **AutoDL** 上执行。

相关文档：

- AutoDL 路径和纪律：[`docs/autodl_workflow.md`](autodl_workflow.md)
- 实验矩阵：[`docs/experiment_matrix.md`](experiment_matrix.md)
- B0 策略退化修复执行文档：[`../TRUST-R1_策略退化问题修复执行文档.md`](../TRUST-R1_策略退化问题修复执行文档.md)

---

## 1. 启动前原则

1. **不要在本地 Mac 跑实验**：本地只做代码编辑、文档整理、小型静态检查。
2. **正式实验必须基于 git commit**：不要用未提交核心代码跑正式结果；如果必须用 uncommitted diff，只能作为 debug，并在日志中记录 diff。
3. **当前优先级是 B0 pilot-safe**：先跑通 B0 clean baseline 的 step-0 / 2-step / 10-step，确认不再出现 invalid 文本复制和 `!` 重复后，再讨论 30/100 step 或 B1/M1/M2。
4. **不要从 step 100 退化 checkpoint 续训**。
5. **B0 pilot-safe 不启用 TRUST reward，不启用 retrieval fault**。

---

## 2. AutoDL 推荐目录

默认约定如下，可按实例实际挂载调整，但需要记录到实验日志中。

```text
/root/autodl-tmp/TRUST-R1/       # 代码仓库
/root/autodl-tmp/data/nq_search/ # RL train/test parquet
/root/autodl-tmp/models/         # 模型权重
/root/autodl-tmp/runs/           # 训练输出、checkpoint、日志
/root/autodl-fs/data/            # wiki-18 corpus
/root/autodl-fs/indexes/         # FAISS / BM25 index
```

本指南后续命令默认代码路径为：

```bash
cd /root/autodl-tmp/TRUST-R1
```

---

## 3. 代码同步

### 3.1 本地提交并推送

在本地 Mac：

```bash
git status
git add <changed-files>
git commit -m "..."
git push
```

如果只是 debug，可以不提交，但不能把未提交代码得到的结果写成正式实验结论。

### 3.2 AutoDL 拉取代码

在 AutoDL：

```bash
cd /root/autodl-tmp/TRUST-R1
git pull
git status
git rev-parse HEAD
```

把 `git rev-parse HEAD` 输出的 commit hash 记录到 run log / summary 中。

---

## 4. 准备数据和检索资源

### 4.1 准备 RL parquet 数据

如果 `/root/autodl-tmp/data/nq_search/train.parquet` 和 `test.parquet` 不存在：

```bash
cd /root/autodl-tmp/TRUST-R1
bash scripts/run_trust_r1_experiments.sh \
  --stage prepare-data \
  --data-dir /root/autodl-tmp/data/nq_search
```

### 4.2 准备 wiki-18 corpus 和 e5 index

如果 `/root/autodl-fs/data/wiki-18.jsonl` 和 `/root/autodl-fs/indexes/wiki-18/e5_Flat.index` 不存在：

```bash
cd /root/autodl-tmp/TRUST-R1
SEARCH_DATA_ROOT=/root/autodl-fs
python scripts/download.py --data-root "$SEARCH_DATA_ROOT"
```

注意：不要在本地 Mac 下载 corpus / index。

---

## 5. 启动 retriever

B0 pilot-safe 和后续 RL 都需要 retriever 服务。默认 URL：

```text
http://127.0.0.1:8000/retrieve
```

推荐使用 CPU retriever，避免和 4 张训练 GPU 抢显存。

### 5.1 手动启动 retriever

在 AutoDL：

```bash
cd /root/autodl-tmp/TRUST-R1
mkdir -p /root/autodl-tmp/runs

nohup python3 search_r1/search/retrieval_server.py \
  --index_path /root/autodl-fs/indexes/wiki-18/e5_Flat.index \
  --corpus_path /root/autodl-fs/data/wiki-18.jsonl \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  > /root/autodl-tmp/runs/retriever_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 5.2 检查 retriever health

```bash
curl -fsS \
  -X POST http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["health check"],"topk":1,"return_scores":false}'
```

能返回 JSON 即可继续。

---

## 6. B0 pilot-safe：当前必须先跑的启动流程

B0 pilot-safe 用于修复和验证 clean baseline 的策略退化问题。入口脚本：

```bash
scripts/run_b0_pilot_safe.sh
```

默认关键配置：

```text
model=/root/autodl-tmp/models/Qwen2.5-3B
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

脚本会拒绝：

- `*Instruct*` 模型路径；
- `*step_100*` 或 `*global_step_100*` checkpoint；
- 本地 macOS 环境；
- 缺失的数据、模型或 retriever。

### 6.1 阶段 A：静态检查

```bash
cd /root/autodl-tmp/TRUST-R1
bash scripts/run_b0_pilot_safe.sh check
```

必须检查输出中的：

- `MODEL_PATH` 是否是 `Qwen2.5-3B` Base；
- `model_type`；
- tokenizer / chat template；
- `world_size`；
- `expanded_rollout_batch = train_batch_size * n_agent`；
- batch 整除关系；
- `trust_reward.enabled=false`；
- `retrieval_fault.enabled=false`；
- retriever health 是否通过。

### 6.2 阶段 B：step-0 基线

```bash
bash scripts/run_b0_pilot_safe.sh step0
```

这一步只做 zero-update validation，不进行 optimizer update。

检查重点：

- 是否已经出现 `My previous action is invalid` 复制；
- 是否出现大面积 `!` 重复；
- 原始 Base 模型是否能生成至少部分完整 `<search>` 或 `<answer>`；
- `env/ratio_of_valid_action`、`env/number_of_valid_search`、`env/finish_ratio`；
- `response_length/mean`、`response_length/clip_ratio`；
- `val/test_score/nq`。

如果 step-0 已经大面积退化，停止 RL，优先检查模型路径、tokenizer、chat template、retriever 和生成参数。

### 6.3 阶段 C：2-step dry run

```bash
bash scripts/run_b0_pilot_safe.sh dry2
```

检查重点：

- 4 卡是否无 OOM / NaN / worker exit；
- step 2 checkpoint 是否保存并可加载；
- step 2 相比 step 0 的 valid action rate 是否明显下降；
- 是否出现 invalid 纠错文本复制闭环。

### 6.4 阶段 D：10-step smoke run

只有 step-0 和 dry2 人工确认通过后，再运行：

```bash
bash scripts/run_b0_pilot_safe.sh smoke10
```

验收标准：

- 没有大面积复制 invalid 纠错话术；
- 没有大面积 `!` 单 token 重复；
- `env/ratio_of_valid_action` 没有持续下降到接近 0；
- `response_length/clip_ratio < 0.5`；
- 至少部分样例生成完整 `<search>` 或 `<answer>`；
- step 10 checkpoint 可加载和推理；
- 无 NaN / Inf；
- step 0、2、4、6、8、10 的样例与指标均已保存。

不建议直接运行：

```bash
bash scripts/run_b0_pilot_safe.sh sequence
```

除非你明确想连续执行 step0、dry2、smoke10。推荐每阶段人工检查后再进入下一阶段。

---

## 7. pilot-safe 通过后的下一步

如果 B0 pilot-safe 10-step 通过，下一步优先顺序是：

1. B0 30-step pilot；
2. B0 100-step clean baseline；
3. B1 / M1 / M2 对照实验。

在没有通过 B0 10-step 前，不启动 M1/M2，不启动 100/300/600-step 正式训练。

---

## 8. B0/B1/M1/M2 常规实验入口

常规实验入口是：

```bash
scripts/run_trust_r1_experiments.sh
```

该脚本支持：

```text
--stage check|prepare-data|launch-retriever|dry-run|train|eval|suite
--experiment B0|B1|M1|M2
--suite smoke|core|eval
```

示例：只做环境检查：

```bash
bash scripts/run_trust_r1_experiments.sh --stage check
```

示例：B0 dry run：

```bash
bash scripts/run_trust_r1_experiments.sh \
  --stage dry-run \
  --experiment B0 \
  --model /root/autodl-tmp/models/Qwen2.5-3B \
  --data-dir /root/autodl-tmp/data/nq_search \
  --retriever-url http://127.0.0.1:8000/retrieve \
  --gpus-per-node 4 \
  --cuda-visible-devices 0,1,2,3
```

示例：核心实验组，必须在 B0 pilot-safe 通过后再考虑：

```bash
bash scripts/run_trust_r1_experiments.sh \
  --suite core \
  --experiments B0,B1,M1,M2 \
  --model /root/autodl-tmp/models/Qwen2.5-3B \
  --data-dir /root/autodl-tmp/data/nq_search \
  --run-root /root/autodl-tmp/runs \
  --retriever-url http://127.0.0.1:8000/retrieve \
  --fault-mode mixed \
  --fault-rate 0.2 \
  --seed 42 \
  --gpus-per-node 4 \
  --cuda-visible-devices 0,1,2,3
```

注意：常规脚本不是本轮策略退化修复的首选入口。当前先使用 `scripts/run_b0_pilot_safe.sh`。

---

## 9. 一键脚本入口

一键脚本是：

```bash
scripts/autodl_one_click_experiment.sh
```

它会串联：

```text
smoke: check -> prepare data if missing -> retriever health/start -> B0 dry-run -> M2 dry-run
core:  smoke -> train B0,B1,M1,M2
eval:  check -> retriever health/start -> eval B0,B1,M1,M2
```

示例：

```bash
bash scripts/autodl_one_click_experiment.sh --mode smoke
```

当前不建议直接用它启动正式 core，因为 B0 仍处于 pilot-safe 验证阶段。等 B0 pilot-safe 通过后，再考虑是否使用一键脚本。

---

## 10. 每次启动后必须保存的信息

每次实验至少保存：

```text
run_id
commit hash
git status / uncommitted diff 状态
启动命令
resolved config / overrides.txt
preflight.txt
开始和结束时间
GPU 类型和数量
模型路径和 tokenizer 信息
数据路径和数据条数
retriever URL / corpus / index
fault mode / fault rate / fault seed
train seed
checkpoint 路径
训练日志路径
W&B run URL（如启用）
抽样 trajectory / 生成样例路径
```

`run_b0_pilot_safe.sh` 会在 run 目录下写入：

```text
preflight.txt
command.sh
run.log
checkpoints/
```

常规 `run_trust_r1_experiments.sh` 会写入：

```text
git_commit.txt
env.txt
overrides.txt
command.sh
<mode>.log
checkpoints/
```

---

## 11. 退化熔断条件

出现任一情况立即停止当前 run：

1. 连续两次验证 `env/ratio_of_valid_action < 0.05`；
2. `response_length/clip_ratio > 0.8`；
3. 固定样例中超过 80% 复制 invalid 纠错文本；
4. 固定样例中超过 50% 出现明显单 token 长重复；
5. reward max、mean、min 连续多次全部相同且有效动作率下降；
6. actor loss、KL、grad norm 出现 NaN / Inf；
7. step 2 相比 step 0 的有效动作率明显下降且 step 4 继续下降。

熔断后保存：

```text
当前 checkpoint
resolved config / overrides.txt / preflight.txt
W&B run URL
最近 20 条轨迹或生成样例
reward 与 advantage 摘要
optimizer LR、grad norm、KL、entropy
ACTIVE_TRAJ_NUM
```

---

## 12. 最小启动速查

在 AutoDL 上，最小启动流程如下：

```bash
cd /root/autodl-tmp/TRUST-R1

git pull
git status
git rev-parse HEAD

# 确保 retriever 已启动并健康
curl -fsS \
  -X POST http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["health check"],"topk":1,"return_scores":false}'

# B0 pilot-safe 分阶段执行
bash scripts/run_b0_pilot_safe.sh check
bash scripts/run_b0_pilot_safe.sh step0
bash scripts/run_b0_pilot_safe.sh dry2
bash scripts/run_b0_pilot_safe.sh smoke10
```

每一步结束后先人工检查日志，再进入下一步。
