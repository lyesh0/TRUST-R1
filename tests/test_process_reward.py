import pytest
import torch

from trust_r1.process_reward import (
    add_query_local_advantage,
    build_process_features,
    contains_alias,
    first_hit_rewards,
)


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(int(token_id)) for token_id in token_ids)


def _features(text, trace_queries, aliases=("moon",), max_steps=2):
    tokenizer = CharTokenizer()
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    trace = {"searches": [{"query": query} for query in trace_queries]}
    return build_process_features(ids, len(ids), aliases, trace, tokenizer, max_steps)


def test_alias_matching_uses_complete_normalized_token_windows():
    assert contains_alias("The U.S. landed there.", ["US"])
    assert not contains_alias("This business expanded.", ["US"])
    assert contains_alias("Apollo, the program", ["the Apollo"])
    assert not contains_alias("anything", ["the"])


@pytest.mark.parametrize(
    "hits, expected",
    [
        ([False, True], [0.0, 1.0]),
        ([True, True], [1.0, 0.0]),
        ([False, False], [0.0, 0.0]),
    ],
)
def test_first_hit_rewards(hits, expected):
    assert first_hit_rewards(hits, 2) == expected


def test_process_features_mark_only_query_content_tokens():
    text = (
        "prompt example <search>ignored</search>"
        "<think>x</think><search>apollo moon</search>"
        "<information>The Moon landing succeeded.</information>"
    )
    response = text[text.index("<think>"):]
    features = _features(response, ["apollo moon"])

    assert features.alignment_valid
    assert features.step_rewards.tolist() == [1.0, 0.0]
    marked = torch.nonzero(features.query_step_ids == 1).flatten().tolist()
    marked_text = "".join(response[index] for index in marked)
    assert marked_text == "apollo moon"


def test_alignment_mismatch_clears_query_spans():
    features = _features(
        "<search>apollo moon</search><information>moon</information>",
        ["different query"],
    )
    assert not features.alignment_valid
    assert features.query_step_ids.sum().item() == 0


def test_unclosed_information_is_a_parse_error():
    features = _features("<search>q</search><information>moon", ["q"])
    assert not features.alignment_valid
    assert features.parse_error_count == 1


def test_local_advantage_is_grouped_by_uid_and_length_normalized():
    answer = torch.zeros(4, 8)
    rewards = torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    step_ids = torch.zeros(4, 8, dtype=torch.long)
    step_ids[0, 1:3] = 1
    step_ids[1, 1:5] = 1
    step_ids[2, 2:4] = 1
    step_ids[3, 2:4] = 1
    loss_mask = torch.ones_like(step_ids)

    combined, local, metrics = add_query_local_advantage(
        answer,
        rewards,
        step_ids,
        ["q1", "q1", "q2", "q2"],
        loss_mask,
        weight=0.2,
        z_clip=2.0,
    )

    assert torch.equal(combined, local)
    assert metrics["process/eligible_group_count"] == 2.0
    assert metrics["process/informative_group_count"] == 1.0
    assert torch.isclose(local[0].sum(), torch.tensor(0.2 / 2**0.5), atol=1e-6)
    assert torch.isclose(local[1].sum(), torch.tensor(-0.2 / 2**0.5), atol=1e-6)
    assert local[2:].sum().item() == 0.0


def test_missing_search_step_does_not_participate():
    answer = torch.zeros(4, 4)
    rewards = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    step_ids = torch.zeros(4, 4, dtype=torch.long)
    step_ids[0, 1:3] = 2
    loss_mask = torch.ones_like(step_ids)

    _, local, metrics = add_query_local_advantage(
        answer, rewards, step_ids, ["same"] * 4, loss_mask
    )

    assert local.sum().item() == 0.0
    assert metrics["process/eligible_group_count"] == 0.0


def test_local_advantage_respects_loss_mask():
    answer = torch.zeros(2, 4)
    rewards = torch.tensor([[1.0], [0.0]])
    step_ids = torch.tensor([[0, 1, 1, 0], [0, 1, 1, 0]])
    loss_mask = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1]])

    _, local, _ = add_query_local_advantage(
        answer, rewards, step_ids, ["same", "same"], loss_mask
    )

    assert local[:, 2].sum().item() == 0.0
    assert torch.isclose(local[0].sum(), torch.tensor(0.2 / 2**0.5), atol=1e-6)


def test_b0_weight_zero_keeps_answer_advantage_and_zero_local_tensor():
    answer = torch.randn(2, 4)
    rewards = torch.tensor([[1.0], [0.0]])
    step_ids = torch.tensor([[0, 1, 1, 0], [0, 1, 1, 0]])
    combined, local, metrics = add_query_local_advantage(
        answer, rewards, step_ids, ["same", "same"], torch.ones_like(step_ids), weight=0.0
    )
    assert torch.equal(combined, answer)
    assert torch.count_nonzero(local).item() == 0
    assert metrics["process/informative_group_count"] == 1.0
