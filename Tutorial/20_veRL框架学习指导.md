# 20. veRL 框架学习指导

这一章的目标不是泛泛介绍 veRL，而是结合 Search-R1 里的具体代码，把你真正会碰到的 veRL 结构讲清楚。

要回答的问题是：

```text
veRL 在这个项目里扮演什么角色？
trainer、worker、worker group、DataProto 分别是什么？
Search-R1 是怎么挂接到 veRL 上的？
学习 veRL 时应该先看哪些文件，按什么顺序看？
```

---

## 20.1 先给结论：veRL 在这个项目里负责什么

Search-R1 不是从零写了一个 RLHF 训练框架，它是**基于 veRL 做了 search-enhanced rollout 扩展**。

在这个仓库里：

```text
Search-R1 自己负责：
- search/answer agent 逻辑
- 检索服务调用
- reward 规则适配

veRL 负责：
- dataloader
- Ray 分布式调度
- actor / critic / ref / reward model worker
- rollout、logprob、advantage、update_actor 等训练数据流
- FSDP / Megatron / vLLM 等底层基础设施整合
```

所以你学习 veRL，重点不是把 `verl/` 目录每一行都读完，而是看清楚：

```text
Search-R1 的业务逻辑是怎么“插进” veRL 训练主循环里的。
```

---

## 20.2 学 veRL 先看哪几个文件

推荐顺序：

1. [`verl/trainer/main_ppo.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/main_ppo.py:104)
2. [`verl/trainer/ppo/ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:313)
3. [`verl/workers/fsdp_workers.py`](/Users/icarus/Documents/Search-R1-main/verl/workers/fsdp_workers.py:355)
4. [`verl/single_controller/ray/base.py`](/Users/icarus/Documents/Search-R1-main/verl/single_controller/ray/base.py:176)
5. [`verl/__init__.py`](/Users/icarus/Documents/Search-R1-main/verl/__init__.py:22)

原因很直接：

- `main_ppo.py`：入口和装配
- `ray_trainer.py`：训练主循环
- `fsdp_workers.py`：实际干活的 worker
- `single_controller/ray`：worker group 和 Ray 绑定
- `DataProto`：所有阶段通用的数据容器

如果你先从 `models/` 或 `third_party/vllm/` 开始读，会很容易偏离主线。

---

## 20.3 veRL 的入口是什么

入口在：

- [`verl/trainer/main_ppo.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/main_ppo.py:104)

```python
@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(...)

    ray.get(main_task.remote(config))
```

这里你要抓住两件事：

1. `Hydra` 负责把 shell 脚本里的配置合成一个大 `config`
2. `Ray` 负责把训练任务放到远程 worker 环境里执行

然后在：

- [`main_ppo.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/main_ppo.py:113)

```python
@ray.remote
def main_task(config):
```

真正开始装配训练器。

所以对 veRL 的第一层理解可以写成：

```text
veRL 的 main 入口负责接配置、起 Ray、选 worker 实现、创建 trainer。
```

---

## 20.4 veRL 里 trainer 是什么

核心 trainer 是：

- [`RayPPOTrainer`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:313)

```python
class RayPPOTrainer(object):
```

它不是一个模型类，而是**训练调度器**。

它负责：

- 创建 dataloader
- 初始化日志
- 初始化 actor/critic/ref/reward model worker
- 组织 rollout
- 调 reward 函数
- 计算 advantage
- 更新 actor 和 critic
- 定期验证和保存

看构造函数：

```python
def __init__(
    self,
    config,
    tokenizer,
    role_worker_mapping,
    resource_pool_manager,
    ray_worker_group_cls,
    reward_fn=None,
    val_reward_fn=None
):
```

这说明 trainer 自己不训练模型参数，它更像“调度中心”。

---

## 20.5 veRL 里的 DataProto 是什么

在：

- [`verl/__init__.py`](/Users/icarus/Documents/Search-R1-main/verl/__init__.py:22)

```python
from .protocol import DataProto
```

虽然这里没展开实现，但你在整个 Search-R1 和 veRL 代码里会反复看到 `DataProto`。

你可以先把它理解成：

```text
veRL 统一使用的“批数据容器”。
```

它里面通常分两部分：

- `batch`：张量数据，比如 `input_ids`、`attention_mask`、`responses`
- `non_tensor_batch`：非张量元信息，比如 `data_source`、`reward_model`、`index`
- `meta_info`：流程控制信息，比如 `pad_token_id`、`eos_token_id`、`global_token_num`

在 Search-R1 里你已经见过：

- [`generation.py`](/Users/icarus/Documents/Search-R1-main/search_r1/llm_agent/generation.py:111)
  `DataProto.from_dict(...)`
- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:701)
  `DataProto.from_single_dict(batch_dict)`

所以 `DataProto` 的作用是：

```text
让 rollout、logprob、reward、advantage、actor update 都用同一种数据接口传递批数据。
```

这点很重要，因为它是 veRL 数据流的骨架。

---

## 20.6 veRL 里的 worker 是什么

在 veRL 里，真正调用模型、计算 logprob、更新参数的，不是 trainer 本体，而是 worker。

在 Search-R1 的 `main_ppo.py` 里：

- [`main_ppo.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/main_ppo.py:151)

```python
role_worker_mapping = {
    Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
    Role.Critic: ray.remote(CriticWorker),
    Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
}
```

这段说明 veRL 把训练角色抽象成几类：

- `ActorRollout`
- `Critic`
- `RefPolicy`
- 可能还有 `RewardModel`

每个角色由一个具体 worker 类实现。

所以：

```text
trainer 决定“什么时候做什么”
worker 决定“具体怎么做”
```

---

## 20.7 veRL 里的 WorkerGroup 是什么

单个 worker 不够，因为训练往往要多卡、多进程、多个角色协作。

所以 veRL 在 worker 之上又有 `WorkerGroup`。

在 `main_ppo.py` 里，FSDP 分支使用：

- [`main_ppo.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/main_ppo.py:136)

```python
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
from verl.single_controller.ray import RayWorkerGroup
ray_worker_group_cls = RayWorkerGroup
```

也就是说，具体 worker 会被包装进 `RayWorkerGroup`。

你可以这样理解：

```text
Worker：单个干活的远程执行单元
WorkerGroup：一组同类 worker 的组织器，负责统一调度和 RPC 调用
```

在 `ray_trainer.py` 的 `init_workers()` 里能看到它的创建过程：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:593)

```python
worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
```

这一步的意思是：

```text
先定义每种角色要起哪些 worker
再把它们放到一个 RayWorkerGroup 里
再 spawn 出真正可调用的远程对象
```

---

## 20.8 veRL 里的资源池是什么

Search-R1 用的是：

- [`main_ppo.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/main_ppo.py:157)

```python
global_pool_id = 'global_pool'
resource_pool_spec = {
    global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
}
```

这表示：

```text
定义一个全局资源池，包含每个节点可用的 GPU 数量。
```

然后不同角色会被映射到这个池：

```python
mapping = {
    Role.ActorRollout: global_pool_id,
    Role.Critic: global_pool_id,
    Role.RefPolicy: global_pool_id,
}
```

所以 veRL 的资源调度思路是：

```text
先定义 GPU 资源池
-> 再把不同角色映射到资源池
-> 再由 WorkerGroup 在资源池上 spawn worker
```

这是它支持多角色、多 GPU 的基础。

---

## 20.9 Search-R1 是怎么接到 veRL 的 trainer 上的

核心在：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:688)

```python
generation_manager = LLMGenerationManager(
    tokenizer=self.tokenizer,
    actor_rollout_wg=self.actor_rollout_wg,
    config=gen_config,
)
```

然后在训练循环中：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:729)

```python
final_gen_batch_output = generation_manager.run_llm_loop(
    gen_batch=gen_batch,
    initial_input_ids=first_input_ids,
)
```

这说明：

```text
Search-R1 没有改掉 veRL 的训练主循环。
它是在 rollout 阶段，把默认的单轮生成替换成了自己的多轮 search rollout。
```

这是你学习 veRL 时最值得记住的挂接点。

---

## 20.10 veRL 的训练数据流是什么样

看 `RayPPOTrainer.fit()`：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:654)

可以概括成：

```text
DataLoader 取出 batch_dict
-> DataProto.from_single_dict(batch_dict)
-> repeat 成多条 rollout 样本
-> pop 出 input_ids / attention_mask / position_ids 给 rollout
-> generation_manager.run_llm_loop() 生成完整轨迹
-> actor_rollout_wg.compute_log_prob() 计算 logprob
-> reward_fn(batch) 计算分数
-> compute_advantage() 算优势
-> actor_rollout_wg.update_actor() 更新策略
```

如果是 GAE，还会有：

- `critic_wg.compute_values()`
- `critic_wg.update_critic()`

但 Search-R1 当前 `train_grpo.sh` 用的是 GRPO，所以通常 `use_critic=False`。

---

## 20.11 veRL 的 dataloader 怎么接 parquet

看：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:372)

```python
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
self.train_dataset = RLHFDataset(
    parquet_files=self.config.data.train_files,
    tokenizer=self.tokenizer,
    prompt_key=self.config.data.prompt_key,
    max_prompt_length=self.config.data.max_prompt_length,
    filter_prompts=True,
    return_raw_chat=self.config.data.get('return_raw_chat', False),
    truncation='error'
)
```

这里说明 veRL 默认期望训练数据已经整理成它认识的 parquet 格式。

也就是说：

```text
Search-R1 的数据处理脚本负责“把 QA 数据整理成 veRL 能读的 parquet”
veRL 的 RLHFDataset 负责“把 parquet 转成训练 batch”
```

这也是 Search-R1 和 veRL 的一个重要接口边界。

---

## 20.12 veRL 的 validation 怎么做

在：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:436)

`_validate()` 会重新构建一个 `GenerationConfig` 和 `LLMGenerationManager`，然后对验证集运行 rollout，再用 `val_reward_fn` 打分。

关键区别：

```python
'do_sample': False,
'validate': True,
```

这说明验证阶段通常会关闭采样随机性，按更稳定的方式生成。

对学习 veRL 来说，这里值得记住的点是：

```text
训练和验证共享同一套 rollout 基础设施，只是 meta_info 和采样策略不同。
```

---

## 20.13 veRL 的 FSDPWorker 实际做什么

看：

- [`verl/workers/fsdp_workers.py`](/Users/icarus/Documents/Search-R1-main/verl/workers/fsdp_workers.py:355)

有三个非常关键的方法：

1. `update_actor()`
2. `compute_log_prob()`
3. `generate_sequences()`

### `update_actor()`

```python
def update_actor(self, data: DataProto):
    ...
    metrics = self.actor.update_policy(data=data)
```

作用：

```text
接收 trainer 准备好的 batch，真正执行策略网络参数更新。
```

### `compute_log_prob()`

```python
def compute_log_prob(self, data: DataProto) -> DataProto:
    ...
    old_log_probs = self.actor.compute_log_prob(data=data)
```

作用：

```text
对 rollout 轨迹重新计算 old_log_probs，供 PPO/GRPO 后续使用。
```

### `generate_sequences()`

```python
def generate_sequences(self, prompts: DataProto):
    ...
    output = self.rollout.generate_sequences(prompts=prompts)
```

作用：

```text
对给定 prompt 做 rollout 生成。
```

所以你可以把 FSDPWorker 看成 veRL 对底层模型调用能力的封装层。

---

## 20.14 veRL 的 RayPPOTrainer 和 FSDPWorker 的关系

关系可以这样写：

```text
RayPPOTrainer
负责控制训练流程

ActorRolloutRefWorker / CriticWorker / RewardModelWorker
负责具体执行模型相关计算

RayWorkerGroup
负责把这些 worker 组织成可远程调用的分布式组
```

trainer 并不会自己直接做：

- generate
- compute_log_prob
- update_actor
- update_critic

它只是通过 worker group 去调用这些能力。

---

## 20.15 学 veRL 时最该关注的几个概念

### `DataProto`

veRL 全链路统一的数据容器。

### `RayPPOTrainer`

训练调度器，组织整个数据流。

### `Worker / WorkerGroup`

真正执行模型计算的远程角色，以及这些角色的分组调度器。

### `resource_pool`

GPU 资源映射和角色部署的基础。

### `reward_fn`

trainer 外挂的奖励计算逻辑。Search-R1 在这里接了自己的 `RewardManager`。

### `rollout backend`

底层生成后端，比如 FSDP、Megatron、vLLM。Search-R1 当前主要走 `fsdp_workers + vllm rollout` 的组合。

---

## 20.16 针对 Search-R1，veRL 应该怎么学

不要用“完整掌握 veRL”做第一目标。合理顺序是：

1. 看懂 `main_ppo.py`：入口和装配
2. 看懂 `ray_trainer.py`：训练主循环
3. 看懂 `generation.py`：Search-R1 是怎么插入 rollout 的
4. 看懂 `RewardManager + qa_em.py`：Search-R1 是怎么接 reward 的
5. 最后再看 `fsdp_workers.py`：实际执行 generate / logprob / update_actor

也就是说，对你当前目标来说：

```text
先学 veRL 的“控制流”
再学 veRL 的“执行层”
最后再补 veRL 的“底层并行细节”
```

这是更有效的顺序。

---

## 20.17 如果你要排 veRL 相关 bug，先看哪里

### 训练没启动起来

先看：

- `main_ppo.py`
- `Hydra config`
- `Ray init`

### batch 读不到数据

先看：

- `ray_trainer.py / _create_dataloader`
- `RLHFDataset`
- parquet 路径和字段名

### rollout 阶段报错

先看：

- `ray_trainer.py / run_llm_loop 调用点`
- `generation.py`
- `fsdp_workers.py / generate_sequences`

### reward 异常

先看：

- `reward_fn(batch)` 调用点
- `RewardManager`
- `qa_em.py`

### actor 更新异常

先看：

- `ray_trainer.py / update_actor`
- `fsdp_workers.py / update_actor`

这种分层定位比直接在 `verl/` 目录里漫无目的搜要有效得多。

---

## 20.18 一句话总结 veRL

结合 Search-R1，你可以把 veRL 概括成：

**一个把 RLHF 训练流程拆成 trainer、DataProto、worker、worker group 和资源池的分布式框架；Search-R1 则在 rollout 和 reward 两个位置插入了自己的 search agent 逻辑。**

如果后面继续写 Tutorial，下一篇适合补：

```text
21_veRL中的DataProto和WorkerGroup到底怎么工作
```

那一篇可以把 `DataProto`、`RayWorkerGroup`、`ActorRolloutRefWorker` 的关系继续拆细。
