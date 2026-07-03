from trust_r1.config import RewardConfig
from trust_r1.rewards import compute_reward_breakdown, count_duplicate_queries


def test_recovery_reward_requires_correct_answer_and_recovery_conditions():
    reward = compute_reward_breakdown(
        answer_correct=True,
        had_fault=True,
        changed_query_after_fault=True,
        evidence_recovered=True,
        config=RewardConfig(recovery_weight=0.2),
    )
    assert reward.answer == 1.0
    assert reward.recovery == 0.2
    assert reward.total == 1.2


def test_recovery_reward_not_given_when_answer_wrong():
    reward = compute_reward_breakdown(
        answer_correct=False,
        had_fault=True,
        changed_query_after_fault=True,
        evidence_recovered=True,
    )
    assert reward.recovery == 0.0
    assert reward.total == 0.0


def test_duplicate_query_penalty():
    assert count_duplicate_queries(["Who is Ada?", "who   is ada?", "different"]) == 1
    reward = compute_reward_breakdown(answer_correct=True, duplicate_query_count=2)
    assert reward.duplicate_penalty == 0.2
    assert reward.total == 0.8
