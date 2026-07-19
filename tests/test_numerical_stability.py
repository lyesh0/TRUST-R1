import torch
import pytest

from verl.trainer.ppo import core_algos
from verl.utils.torch_functional import masked_mean, masked_sum


def test_policy_ratio_clamp_keeps_loss_and_gradient_finite():
    old_log_prob = torch.zeros(2)
    log_prob = torch.tensor([100.0, -100.0], requires_grad=True)
    advantages = torch.tensor([-1.0, 1.0])
    mask = torch.ones(2)

    pg_loss, _, ppo_kl = core_algos.compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=mask,
        cliprange=0.2,
    )
    (pg_loss + ppo_kl).backward()

    assert torch.isfinite(pg_loss)
    assert torch.isfinite(ppo_kl)
    assert torch.isfinite(log_prob.grad).all()


def test_low_var_kl_clamp_keeps_loss_and_gradient_finite():
    log_prob = torch.tensor([-100.0, 100.0], requires_grad=True)
    ref_log_prob = torch.zeros(2)

    loss = core_algos.kl_penalty(log_prob, ref_log_prob, "low_var_kl").mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(log_prob.grad).all()


def test_non_finite_log_prob_is_not_silently_repaired():
    log_prob = torch.tensor([float("nan")])
    ref_log_prob = torch.zeros(1)

    penalty = core_algos.kl_penalty(log_prob, ref_log_prob, "low_var_kl")

    assert torch.isnan(penalty).all()


def test_masked_reductions_ignore_non_finite_padding():
    values = torch.tensor([1.0, float("nan"), float("inf")], requires_grad=True)
    mask = torch.tensor([1.0, 0.0, 0.0])

    total = masked_sum(values, mask)
    mean = masked_mean(values, mask)
    mean.backward()

    assert total.item() == 1.0
    assert mean.item() == 1.0
    assert torch.equal(values.grad, torch.tensor([1.0, 0.0, 0.0]))


def test_grpo_group_size_mismatch_fails_fast():
    rewards = torch.ones(4, 2)
    mask = torch.ones(4, 2)
    uid = ["prompt-a", "prompt-a", "prompt-b", "prompt-b"]

    with pytest.raises(ValueError, match="GRPO group size mismatch"):
        core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=rewards,
            eos_mask=mask,
            index=uid,
            expected_group_size=5,
        )
