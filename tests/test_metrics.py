from trust_r1.metrics import aggregate_trace_summary_metrics, aggregate_trajectory_metrics
from trust_r1.trajectory import SearchTurn, Trajectory


def make_trajectory(sample_id, correct, turns):
    trajectory = Trajectory(run_id="run", sample_id=sample_id, question="q", is_correct=correct)
    for turn in turns:
        trajectory.add_turn(turn)
    return trajectory


def test_aggregate_trajectory_metrics():
    trajectories = [
        make_trajectory(
            "s1",
            True,
            [
                SearchTurn(step=1, action="search", query="first", fault_enabled=True, fault_type="empty"),
                SearchTurn(step=2, action="search", query="second"),
            ],
        ),
        make_trajectory(
            "s2",
            False,
            [
                SearchTurn(step=1, action="search", query="same"),
                SearchTurn(step=2, action="search", query="same"),
            ],
        ),
    ]

    metrics = aggregate_trajectory_metrics(trajectories)
    assert metrics.count == 2
    assert metrics.accuracy == 0.5
    assert metrics.fault_rate == 0.5
    assert metrics.average_search_calls == 2.0
    assert metrics.query_rewrite_rate == 1.0
    assert metrics.duplicate_query_rate == 0.5
    assert metrics.first_failure_recovery_rate == 1.0


def test_aggregate_trace_summary_metrics():
    metrics = aggregate_trace_summary_metrics([
        {
            "search_count": 2,
            "had_fault": True,
            "searched_again_after_fault": True,
            "changed_query_after_fault": True,
            "duplicate_query_count": 0,
            "fault_types": ["empty"],
        },
        {
            "search_count": 1,
            "had_fault": False,
            "searched_again_after_fault": False,
            "changed_query_after_fault": False,
            "duplicate_query_count": 1,
            "fault_types": [],
        },
    ])
    assert metrics["trust_r1/trace_count"] == 2.0
    assert metrics["trust_r1/search_count_mean"] == 1.5
    assert metrics["trust_r1/fault_rate"] == 0.5
    assert metrics["trust_r1/searched_again_after_fault_rate"] == 1.0
    assert metrics["trust_r1/changed_query_after_fault_rate"] == 1.0
    assert metrics["trust_r1/duplicate_query_rate"] == 0.5
    assert metrics["trust_r1/duplicate_query_count_mean"] == 0.5
    assert metrics["trust_r1/fault_type/empty_count"] == 1.0


def test_aggregate_trace_summary_metrics_empty_input():
    assert aggregate_trace_summary_metrics([]) == {}
