# 24. GRPO 更新代码与数学

这一章只讲 `GRPO` 路径。重点回答：

```text
一批 rollout 采样出来之后
-> reward 怎么变成 token_level_rewards
-> 为什么不需要 critic
-> group-relative advantage 是怎么计算的
-> actor 最后怎么更新
```

这篇是当前 Search-R1 默认训练脚本最相关的一条更新路径，因为：

- [`train_grpo.sh`](/Users/icarus/Documents/Search-R1-main/train_grpo.sh:41)

```bash
algorithm.adv_estimator=grpo
```

---

## 24.1 GRPO 路径什么时候生效

trainer 里：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:140)

```python
elif adv_estimator == 'grpo':
    ...
```

以及：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:573)

```python
elif self.config.algorithm.adv_estimator == 'grpo':
    self.use_critic = False
```

这说明 GRPO 路径的一个根本区别是：

```text
不训练 critic
不依赖 value function
直接从一组 rollout 的结果构造相对优势
```

所以这篇的主线是：

```text
采样多条 rollout
-> 比较同一问题下这些 rollout 的结果
-> 构造 relative advantage
-> 只更新 actor
```

---

## 24.2 GRPO 为什么要先重复采样

在 trainer 里：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:702)

```python
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
```

以及 rollout 结束后：

```python
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
```

这两处都说明一个事实：

```text
同一个原始问题会生成多条 rollout。
```

为什么这对 GRPO 必要？

因为 GRPO 的核心思想不是“拿单条样本的绝对 reward”，而是：

```text
把同一 prompt 下的多条回答放在一起比较
用组内相对表现构造优势
```

没有组，就没有 group-relative advantage。

---

## 24.3 GRPO 的第一步：reward 先变成 `token_level_scores`

和 PPO 一样，先走：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:788)

```python
reward_tensor = self.reward_fn(batch)
batch.batch['token_level_scores'] = reward_tensor
```

当前 Search-R1 中，reward 常常只落在 response 最后一个有效 token 上。

因此：

```text
一条 rollout 的 sequence-level outcome
在实现里被编码成 token-level tensor
但主要只有最后一个非零位置有值
```

这点对后面的 `compute_grpo_outcome_advantage()` 很关键。

---

## 24.4 GRPO 的第二步：先得到 `token_level_rewards`

继续看 trainer：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:791)

```python
if not self.config.actor_rollout_ref.actor.use_kl_loss:
    batch, kl_metrics = apply_kl_penalty(...)
else:
    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
```

所以 GRPO 也一样，先把 raw score 变成最终 reward：

```text
token_level_rewards = token_level_scores - KL penalty
```

或者在某些配置下：

```text
token_level_rewards = token_level_scores
```

也就是说，GRPO 和 PPO 的区别不在 reward 构造这一步，而在 **advantage 的定义**。

---

## 24.5 GRPO 数学直觉

GRPO 的直觉可以先写成：

```text
同一个 prompt 下采样 K 条回答
如果某条回答比组内平均更好，它应该拿正优势
如果更差，它应该拿负优势
```

也就是说，GRPO 不需要学一个 value baseline：

```text
V(s)
```

而是直接用组内相对表现做 baseline。

一个简化写法是：

```text
A_i = (r_i - mean_group(r)) / std_group(r)
```

其中：

- `r_i`：同一组里第 `i` 条 rollout 的结果
- `mean_group(r)`：该组平均分
- `std_group(r)`：该组标准差

这就是“Group Relative”。

---

## 24.6 GRPO 代码里如何取出每条 rollout 的 sequence reward

核心函数在：

- [`core_algos.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/core_algos.py:111)

```python
def compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index, epsilon=1e-6):
```

先看这几行：

```python
response_length = token_level_rewards.shape[-1]
non_zero_mask = (token_level_rewards != 0)
scores = (token_level_rewards * non_zero_mask).sum(dim=-1)
```

这里的语义是：

```text
把每条 response 的 token-level rewards 压成一个 sequence-level score
```

由于当前 Search-R1 里通常只有最后一个 token 有分，所以这基本等价于：

```text
取这条 rollout 的最终 outcome reward
```

---

## 24.7 GRPO 代码里如何按“组”聚合

接着看：

```python
id2score = defaultdict(list)
```

然后：

```python
for i in range(bsz):
    id2score[index[i]].append(scores[i])
```

这里的 `index` 来自：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:142)

```python
index = data.non_tensor_batch['uid']
```

而 `uid` 在 Search-R1 路径里通常是：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:744)

```python
batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
```

也就是说：

```text
同一个原始样本 index 对应的一组 rollout
会被归到同一个 group
```

所以 GRPO 的“组”在这里其实就是：

```text
同一个问题的多次采样结果
```

---

## 24.8 GRPO 代码里如何做组内标准化

继续看：

```python
for idx in id2score:
    if len(id2score[idx]) == 1:
        id2mean[idx] = torch.tensor(0.0)
        id2std[idx] = torch.tensor(1.0)
    elif len(id2score[idx]) > 1:
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
```

然后：

```python
for i in range(bsz):
    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
```

数学上就是：

```text
A_i = (r_i - mu_group) / (sigma_group + epsilon)
```

这就是 GRPO 的核心。

需要注意两个实现细节：

1. 如果某组只有 1 条样本，代码直接设：
   - `mean = 0`
   - `std = 1`

2. 标准差分母里加了 `epsilon`
   - 防止除零

---

## 24.9 GRPO 为什么返回的是 token 级 advantage

最后这两行：

```python
scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
return scores, scores
```

意思是：

```text
虽然 GRPO 本质上先算的是 sequence-level 相对分数，
但为了兼容 veRL 后续 actor update 的 token-level 接口，
代码把这个标量复制到 response 的每个有效 token 上。
```

所以返回的：

- `advantages`
- `returns`

在 GRPO 路径下其实是同一个东西。

这和 PPO 不同。

PPO 里：

```text
advantages = GAE 结果
returns = advantages + values
```

GRPO 里：

```text
advantages = returns = group-relative normalized score
```

---

## 24.10 trainer 里什么时候调用 GRPO advantage

在：

- [`compute_advantage()`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:140)

```python
elif adv_estimator == 'grpo':
    token_level_rewards = data.batch['token_level_rewards']
    index = data.non_tensor_batch['uid']
    responses = data.batch['responses']
    response_length = responses.size(-1)
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    advantages, returns = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        eos_mask=response_mask,
        index=index
    )
    data.batch['advantages'] = advantages
    data.batch['returns'] = returns
```

这里直接说明：

```text
GRPO 不需要 values
GRPO 不需要 critic
它只需要 rewards + 组标识 uid + response mask
```

---

## 24.11 GRPO 的 actor 更新和 PPO 有什么关系

优势构造不同，但 actor 的更新形式仍然沿用 PPO clipped policy loss。

具体看：

- [`dp_actor.py`](/Users/icarus/Documents/Search-R1-main/verl/workers/actor/dp_actor.py:252)

```python
pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
    old_log_prob=old_log_prob,
    log_prob=log_prob,
    advantages=advantages,
    eos_mask=response_mask,
    cliprange=clip_ratio
)
```

这说明：

```text
GRPO 和 PPO 的核心区别主要在 advantage 的构造
不是在 actor policy loss 的形式
```

也就是：

- PPO：`advantages` 来自 GAE
- GRPO：`advantages` 来自组内相对标准化分数

但最后都用 PPO 风格的 clipped objective 来更新 actor。

---

## 24.12 GRPO 路径为什么不更新 critic

trainer 里已经明确：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:573)

```python
elif self.config.algorithm.adv_estimator == 'grpo':
    self.use_critic = False
```

因此：

- 不会走 `critic_wg.compute_values(batch)`
- 不会走 `critic_wg.update_critic(batch)`

原因是 GRPO 已经用组内相对分数做了 baseline 替代，不再需要单独学一个 value function 来估计状态价值。

这就是它工程上比 PPO 更简洁的地方。

---

## 24.13 GRPO 路径完整数据流

压缩成一条链：

```text
rollout 完成
-> reward_fn(batch) 得到 token_level_scores
-> apply_kl_penalty 得到 token_level_rewards
-> 根据 uid 把同一个问题的多条 rollout 归组
-> 对组内 sequence reward 做 mean/std 标准化
-> 把组内相对分数 broadcast 到每个有效 token
-> 写成 advantages / returns
-> actor_rollout_wg.update_actor(batch)
```

这就是当前 Search-R1 默认 GRPO 训练的核心更新路径。

---

## 24.14 GRPO 路径里最容易出问题的点

### `uid` 没对齐

如果同一问题的 rollout 没被归到同一组，group-relative advantage 会失真。

先看：

- `batch.non_tensor_batch['uid']`
- `batch.non_tensor_batch['index']`

### 每组只有 1 个样本

那 GRPO 的组内相对比较就退化了。

先看：

- `n_agent`
- `rollout.n`
- repeat 后的 batch 结构

### reward 全相同

组内标准化后优势接近 0，actor 更新会很弱。

先看：

- reward 设计
- rollout 多样性

### reward 只落在末 token 但解析失败

sequence score 会直接错。

先看：

- `RewardManager`
- `qa_em.py`

---

## 24.15 GRPO 和 PPO 的最短对比

### PPO

```text
需要 critic
advantages 由 GAE 计算
returns 用于 value loss
更新 actor 和 critic
```

### GRPO

```text
不需要 critic
advantages 来自组内相对 reward 标准化
returns 与 advantages 相同
只更新 actor
```

所以你可以把 GRPO 理解成：

**把 value baseline 换成同组 rollout 的相对 baseline。**

---

## 24.16 本章结论

GRPO 路径的本质是：

**先对同一个问题采样多条 rollout，再用组内 reward 的相对高低构造 advantage，最后仍然通过 PPO 风格的 clipped policy loss 更新 actor，但不再训练 critic。**

关键代码落点是：

- GRPO advantage：[`core_algos.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/core_algos.py:111)
- trainer 分支：[`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:140)
- 关闭 critic：[`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:573)
- actor loss：[`dp_actor.py`](/Users/icarus/Documents/Search-R1-main/verl/workers/actor/dp_actor.py:252)
