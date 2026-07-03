import pytest

from trust_r1.faults import FaultEvent
from trust_r1.rollout_logging import RolloutTraceRecorder


def event(query="q", fault_type="empty"):
    return FaultEvent(
        enabled=True,
        applied=True,
        fault_type=fault_type,
        query=query,
        original_count=3,
        returned_count=0,
        seed=42,
    )


def test_rollout_trace_recorder_groups_searches_by_sample_index():
    recorder = RolloutTraceRecorder(batch_size=3)
    recorder.record_searches(
        sample_indices=[2, 0],
        queries=["query two", "query zero"],
        events=[event("query two"), event("query zero", "drop_top")],
        step=5,
    )

    meta = recorder.to_meta()
    assert meta[0]["sample_index"] == 0
    assert meta[0]["searches"][0]["query"] == "query zero"
    assert meta[0]["searches"][0]["fault_type"] == "drop_top"
    assert meta[1]["searches"] == []
    assert meta[2]["searches"][0]["query"] == "query two"
    assert meta[2]["searches"][0]["step"] == 5


def test_rollout_trace_recorder_builds_summary():
    recorder = RolloutTraceRecorder(batch_size=1)
    recorder.record_searches(
        sample_indices=[0],
        queries=["same query"],
        events=[event("same query", "empty")],
        step=0,
    )
    recorder.record_searches(
        sample_indices=[0],
        queries=["new query"],
        events=[event("new query", "clean")],
        step=1,
    )
    summary = recorder.to_summary()[0]
    assert summary["had_fault"] is True
    assert summary["searched_again_after_fault"] is True
    assert summary["changed_query_after_fault"] is True
    assert summary["search_count"] == 2
    assert summary["duplicate_query_count"] == 0
    assert summary["fault_types"] == ["empty"]


def test_rollout_trace_summary_counts_duplicate_queries():
    recorder = RolloutTraceRecorder(batch_size=1)
    recorder.record_searches(
        sample_indices=[0],
        queries=["same query"],
        events=[event("same query", "clean")],
        step=0,
    )
    recorder.record_searches(
        sample_indices=[0],
        queries=["same   query"],
        events=[event("same query", "clean")],
        step=1,
    )
    assert recorder.to_summary()[0]["duplicate_query_count"] == 1


def test_rollout_trace_recorder_validates_lengths():
    recorder = RolloutTraceRecorder(batch_size=1)
    with pytest.raises(ValueError):
        recorder.record_searches(sample_indices=[0], queries=[], events=[], step=0)


def test_rollout_trace_recorder_validates_sample_index():
    recorder = RolloutTraceRecorder(batch_size=1)
    with pytest.raises(IndexError):
        recorder.record_searches(sample_indices=[1], queries=["q"], events=[event()], step=0)
