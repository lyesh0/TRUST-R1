from scripts.analyze_stage1 import analyze_records
from scripts.eval_action_format import score_outputs


def test_action_format_metrics_use_type_specific_denominators():
    records = [{"type": "A"}, {"type": "B"}, {"type": "C"}]
    outputs = [
        "<think>x</think><search>query</search>",
        "<think>x</think><answer>answer</answer>",
        "<think>x</think><search>recovery</search>",
    ]
    metrics = score_outputs(records, outputs)
    assert metrics["valid_action_ratio"] == 1.0
    assert metrics["invalid_recovery_rate"] == 1.0
    assert metrics["evidence_answer_rate"] == 1.0


def test_analysis_uses_second_search_and_evidence_subset_denominators():
    records = [
        {
            "valid_action": True,
            "finish_reason": "answer",
            "answer_correct": True,
            "final_answer": "Moon",
            "gold_aliases": ["the moon"],
            "search_count": 2,
            "evidence_hit_by_step": [False, True],
            "queries": ["first", "second"],
        },
        {
            "valid_action": False,
            "finish_reason": "max_turns",
            "answer_correct": False,
            "final_answer": "wrong",
            "gold_aliases": ["answer"],
            "search_count": 1,
            "evidence_hit_by_step": [False, False],
            "queries": ["same"],
        },
    ]
    metrics = analyze_records(records)
    assert metrics["any_search_success"] == 0.5
    assert metrics["incremental_evidence_rate"] == 1.0
    assert metrics["evidence_utilization"] == 1.0
    assert metrics["token_f1"] == 0.5
