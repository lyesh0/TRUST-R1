import pytest

from trust_r1.config import RewardConfig
from trust_r1.reward_adapter import answer_matches, compute_trust_reward, parse_solution


def test_parse_solution_extracts_search_information_and_answer():
    solution = """
<think>Need evidence</think><search>Ada Lovelace</search>
<information>Ada Lovelace wrote notes.</information>
<think>Now answer</think><answer>Ada Lovelace</answer>
"""
    parsed = parse_solution(solution)
    assert parsed.search_queries == ["Ada Lovelace"]
    assert parsed.information_blocks == ["Ada Lovelace wrote notes."]
    assert parsed.answer == "Ada Lovelace"
    assert parsed.valid_format is True


def test_answer_matches_uses_normalized_exact_match():
    assert answer_matches("The Ada Lovelace!", "Ada Lovelace")
    assert not answer_matches("Ada", "Ada Lovelace")


def test_compute_trust_reward_counts_recovery_and_duplicate_penalty():
    solution = """
<search>who is ada</search><information>empty</information>
<search>who is ada</search><information>Ada Lovelace wrote notes.</information>
<answer>Ada Lovelace</answer>
"""
    result = compute_trust_reward(
        solution_str=solution,
        ground_truth={"target": "Ada Lovelace"},
        had_fault=True,
        changed_query_after_fault=True,
        config=RewardConfig(recovery_weight=0.2, duplicate_penalty_weight=0.1),
    )
    assert result.answer_correct is True
    assert result.evidence_recovered is True
    assert result.duplicate_query_count == 1
    assert result.reward.answer == 1.0
    assert result.reward.recovery == 0.2
    assert result.reward.duplicate_penalty == 0.1
    assert result.reward.total == pytest.approx(1.1)


def test_compute_trust_reward_does_not_give_recovery_when_answer_wrong():
    solution = """
<search>who is ada</search><information>Ada Lovelace wrote notes.</information>
<answer>Alan Turing</answer>
"""
    result = compute_trust_reward(
        solution_str=solution,
        ground_truth={"target": "Ada Lovelace"},
        had_fault=True,
        changed_query_after_fault=True,
    )
    assert result.answer_correct is False
    assert result.reward.recovery == 0.0
