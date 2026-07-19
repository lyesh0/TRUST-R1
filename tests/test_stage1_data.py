import pytest

from scripts.data_process.build_stage1_data import (
    TYPE_COUNTS,
    build_sft_records,
    stable_select_indices,
)


def _row(index):
    return {
        "stage1_question_id": "train:%d" % index,
        "prompt": [{"role": "user", "content": "Question: question %d?\n" % index}],
        "reward_model": {"ground_truth": {"target": ["answer%d" % index]}},
    }


def test_stable_selection_is_reproducible_and_unique():
    first = stable_select_indices(20, 10, 42)
    assert first == stable_select_indices(20, 10, 42)
    assert len(first) == len(set(first)) == 10


def test_stable_selection_rejects_small_source():
    with pytest.raises(ValueError):
        stable_select_indices(3, 4, 42)


def test_sft_types_have_frozen_counts_and_disjoint_question_ids():
    rows = [_row(index) for index in range(800)]
    passages = ["passage with answer%d" % index if index < 160 else "no hit" for index in range(800)]
    records, hit_count = build_sft_records(rows, passages)
    counts = {kind: sum(record["type"] == kind for record in records) for kind in TYPE_COUNTS}
    assert counts == TYPE_COUNTS
    assert hit_count == 160
    assert len({record["question_id"] for record in records}) == 800


def test_sft_build_fails_when_type_b_hits_are_insufficient():
    rows = [_row(index) for index in range(800)]
    with pytest.raises(RuntimeError, match="Type B requires"):
        build_sft_records(rows, ["no hit"] * 800)
