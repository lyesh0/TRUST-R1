# 23. PPO 更新代码与数学

这一章只讲 `PPO` 路径。重点回答：

```text
一批数据采样出来之后
-> reward 怎么变成 token_level_rewards
-> GAE 怎么变成 advantages / returns
-> critic 怎么更新
-> actor 怎么更新
```

这里不讲 Search-R1 的检索细节，只讲 **采样后的 RL 更新链路**。

---

## 23.1 PPO 路径在当前项目里什么时候会走

配置入口在：

- [`train_grpo.sh`](/Users/icarus/Documents/Search-R1-main/train_grpo.sh:41)

```bash
algorithm.adv_estimator=grpo
```

当前默认脚本用的是 `grpo`，所以**默认并不走 PPO 的 GAE + critic 路径**。

PPO 路径会在：

```text
config.algorithm.adv_estimator = gae
```

时生效。

在 trainer 里对应代码：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:126)

```python
if adv_estimator == 'gae':
    ...
elif adv_estimator == 'grpo':
    ...
```

以及：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:567)

```python
if self.config.algorithm.adv_estimator == 'gae':
    self.use_critic = True
elif self.config.algorithm.adv_estimator == 'grpo':
    self.use_critic = False
```

所以这篇你要先记住：

```text
PPO 路径 = adv_estimator=gae + use_critic=True
GRPO 路径 = adv_estimator=grpo + use_critic=False
```

---

## 23.2 先看采样后 batch 长什么样

trainer 在 rollout 完成后，先拿到一个 `batch`。

关键代码在：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:788)

```python
reward_tensor = self.reward_fn(batch)
batch.batch['token_level_scores'] = reward_tensor
```

到这个时点，`batch` 里通常已经有：

- `input_ids`
- `responses`
- `attention_mask`
- `position_ids`
- `old_log_probs`
- 可能还有 `ref_log_prob`
- 可能还有 `values`

以及 Search-R1 自己塞进去的：

- `prompts`
- `info_mask`

这些字段的统一容器就是 `DataProto`。

---

## 23.3 PPO 的第一步：reward 先落成 `token_level_scores`

当前项目里 reward 入口是：

- [`main_ppo.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/main_ppo.py:41)

```python
def __call__(self, data: DataProto):
```

它最终把分数写到：

```python
reward_tensor[i, valid_response_length - 1] = score
```

这说明 Search-R1 这里的 reward 结构是：

```text
response 的最后一个有效 token 上有奖励
前面 token 大多是 0
```

也就是说，先有：

```text
token_level_scores
```

还没有经过 KL 惩罚，也还没变成 advantage。

---

## 23.4 PPO 的第二步：KL 惩罚生成 `token_level_rewards`

在 trainer 里：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:791)

```python
if not self.config.actor_rollout_ref.actor.use_kl_loss:
    batch, kl_metrics = apply_kl_penalty(...)
else:
    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
```

看 `apply_kl_penalty()`：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:91)

```python
token_level_rewards = token_level_scores - beta * kld
```

数学上这一步是：

```text
r_t = s_t - beta * KL_t
```

其中：

- `s_t`：原始 token-level score
- `KL_t`：当前策略和 reference policy 的偏离
- `beta`：KL 系数

这一步的意义是：

```text
不仅要答对
还要避免策略一下子偏离参考模型太远
```

---

## 23.5 PPO 的第三步：先算 values

PPO 路径下会用 critic。

先看 trainer：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:773)

```python
if self.use_critic:
    values = self.critic_wg.compute_values(batch)
    batch = batch.union(values)
```

然后看 FSDP critic worker：

- [`fsdp_workers.py`](/Users/icarus/Documents/Search-R1-main/verl/workers/fsdp_workers.py:705)

```python
def compute_values(self, data: DataProto):
    ...
    values = self.critic.compute_values(data=data)
    output = DataProto.from_dict(tensors={'values': values})
```

所以 PPO 路径里，`values` 是 critic 对每个 response token 的 value 预测。

---

## 23.6 PPO 的第四步：GAE 数学原理

GAE 在：

- [`core_algos.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/core_algos.py:70)

```python
def compute_gae_advantage_return(token_level_rewards, values, eos_mask, gamma, lam):
```

核心代码：

```python
delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
lastgaelam = delta + gamma * lam * lastgaelam
advantages_reversed.append(lastgaelam)
```

数学上对应：

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

A_t = delta_t + gamma * lambda * A_{t+1}
```

最后：

```python
returns = advantages + values
advantages = verl_F.masked_whiten(advantages, eos_mask)
```

对应：

```text
R_t = A_t + V(s_t)
```

其中：

- `advantages` 用于 actor update
- `returns` 用于 critic update

你可以把 GAE 理解成：

```text
用 value baseline 降低方差，
再通过 gamma 和 lambda 在偏差/方差之间折中。
```

---

## 23.7 PPO 的第五步：trainer 里什么时候算 GAE

在：

- [`compute_advantage()`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:123)

```python
if adv_estimator == 'gae':
    values = data.batch['values']
    token_level_rewards = data.batch['token_level_rewards']
    advantages, returns = core_algos.compute_gae_advantage_return(...)
    data.batch['advantages'] = advantages
    data.batch['returns'] = returns
```

所以数据流是：

```text
token_level_scores
-> token_level_rewards
-> values
-> advantages + returns
```

这一步是 PPO 的核心。

---

## 23.8 PPO 的第六步：critic 更新

critic 更新发生在：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:807)

```python
if self.use_critic:
    critic_output = self.critic_wg.update_critic(batch)
```

再看具体 loss：

- [`dp_critic.py`](/Users/icarus/Documents/Search-R1-main/verl/workers/critic/dp_critic.py:184)

```python
vf_loss, vf_clipfrac = core_algos.compute_value_loss(
    vpreds=vpreds,
    values=values,
    returns=returns,
    eos_mask=eos_mask,
    cliprange_value=self.config.cliprange_value
)
```

`compute_value_loss()` 在：

- [`core_algos.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/core_algos.py:216)

```python
vpredclipped = clip(values)
vf_losses1 = (vpreds - returns)**2
vf_losses2 = (vpredclipped - returns)**2
vf_loss = 0.5 * masked_mean(max(vf_losses1, vf_losses2))
```

数学上就是 PPO 风格的 clipped value loss：

```text
L_value = 1/2 * max((V_theta - R)^2, (clip(V_theta) - R)^2)
```

目的：

```text
让 value head 拟合 returns，
但又避免一步更新过猛。
```

---

## 23.9 PPO 的第七步：actor 更新

actor 更新在：

- [`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:817)

```python
actor_output = self.actor_rollout_wg.update_actor(batch)
```

具体 policy loss 在：

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

再看 `compute_policy_loss()`：

- [`core_algos.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/core_algos.py:163)

```python
ratio = torch.exp(log_prob - old_log_prob)
pg_losses = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
```

数学上对应 PPO clipped objective：

```text
r_t(theta) = exp(log pi_theta(a_t|s_t) - log pi_old(a_t|s_t))

L_policy = E[min(r_t(theta) A_t, clip(r_t(theta), 1-eps, 1+eps) A_t)]
```

代码里因为是 loss 形式，所以前面带负号。

这一步的意义是：

```text
沿 advantage 的方向更新策略，
但通过 clip 限制单步策略变化。
```

---

## 23.10 PPO 的第八步：entropy 和可选 KL loss

actor update 里还有两项：

- [`dp_actor.py`](/Users/icarus/Documents/Search-R1-main/verl/workers/actor/dp_actor.py:258)

```python
entropy_loss = masked_mean(entropy, response_mask)
policy_loss = pg_loss - entropy_loss * entropy_coeff
```

数学上：

```text
L = L_policy - c_ent * H(pi)
```

目的：

```text
鼓励策略保持一定探索性，避免过早塌缩。
```

如果启用 `use_kl_loss`，还会额外加：

```python
kl_loss = masked_mean(kld, response_mask)
policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
```

这时不是把 KL 融到 reward，而是直接加到 actor loss 上。

---

## 23.11 PPO 路径完整数据流

压缩成一条链：

```text
rollout 完成
-> reward_fn(batch) 得到 token_level_scores
-> apply_kl_penalty 得到 token_level_rewards
-> critic_wg.compute_values 得到 values
-> compute_gae_advantage_return 得到 advantages + returns
-> critic_wg.update_critic 用 returns 更新 value function
-> actor_rollout_wg.update_actor 用 advantages 更新 policy
```

如果你排查 PPO 更新问题，按这个顺序看最有效。

---

## 23.12 PPO 路径里最容易出问题的点

### reward 全 0

后面 `advantages` 也会几乎全 0，actor 更新会失效。

先看：

- `RewardManager`
- `qa_em.py`

### `values` 异常

GAE 会失真，critic/actor 都会受影响。

先看：

- `critic_wg.compute_values`
- `returns` 和 `values` 的量级

### `old_log_probs` 不对

PPO ratio 会错。

先看：

- `compute_log_prob`
- `generate_sequences`

### `cliprange` 太大或太小

会影响更新稳定性。

---

## 23.13 本章结论

PPO 路径的本质是：

**用 reward 构造 token-level rewards，用 critic 估计 value，用 GAE 得到 advantage，再用 clipped policy loss 更新 actor，同时用 clipped value loss 更新 critic。**

在这套实现里，最关键的代码落点是：

- reward：[`main_ppo.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/main_ppo.py:41)
- GAE：[`core_algos.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/core_algos.py:70)
- compute_advantage：[`ray_trainer.py`](/Users/icarus/Documents/Search-R1-main/verl/trainer/ppo/ray_trainer.py:123)
- actor loss：[`dp_actor.py`](/Users/icarus/Documents/Search-R1-main/verl/workers/actor/dp_actor.py:252)
- critic loss：[`dp_critic.py`](/Users/icarus/Documents/Search-R1-main/verl/workers/critic/dp_critic.py:184)
