from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from .trajectory import Trajectory
from .rewards import count_duplicate_queries


@dataclass(frozen=True)
class TrajectoryMetrics:
    count: int = 0
    accuracy: float = 0.0
    fault_rate: float = 0.0
    average_search_calls: float = 0.0
    query_rewrite_rate: float = 0.0
    duplicate_query_rate: float = 0.0
    first_failure_recovery_rate: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def aggregate_trajectory_metrics(trajectories: list[Trajectory]) -> TrajectoryMetrics:
    if not trajectories:
        return TrajectoryMetrics()

    count = len(trajectories)
    correct = sum(1 for traj in trajectories if traj.is_correct)
    had_fault = sum(1 for traj in trajectories if traj.had_fault)
    search_counts = [sum(1 for turn in traj.turns if turn.action == "search") for traj in trajectories]
    rewrites = sum(1 for traj in trajectories if traj.changed_query_after_fault)
    recoveries = sum(1 for traj in trajectories if traj.had_fault and traj.is_correct)

    duplicate_trajectories = 0
    for traj in trajectories:
        queries = [turn.query for turn in traj.turns if turn.action == "search"]
        if count_duplicate_queries(queries) > 0:
            duplicate_trajectories += 1

    return TrajectoryMetrics(
        count=count,
        accuracy=correct / count,
        fault_rate=had_fault / count,
        average_search_calls=sum(search_counts) / count,
        query_rewrite_rate=rewrites / had_fault if had_fault else 0.0,
        duplicate_query_rate=duplicate_trajectories / count,
        first_failure_recovery_rate=recoveries / had_fault if had_fault else 0.0,
    )


def aggregate_trace_summary_metrics(trace_summaries) -> dict[str, float]:
    summaries = [summary for summary in trace_summaries if isinstance(summary, dict)]
    if not summaries:
        return {}

    count = len(summaries)
    fault_count = sum(1 for summary in summaries if summary.get("had_fault", False))
    searched_again = sum(1 for summary in summaries if summary.get("searched_again_after_fault", False))
    changed_query = sum(1 for summary in summaries if summary.get("changed_query_after_fault", False))
    duplicate_count = sum(int(summary.get("duplicate_query_count", 0) or 0) for summary in summaries)
    duplicate_trajectory_count = sum(1 for summary in summaries if int(summary.get("duplicate_query_count", 0) or 0) > 0)
    search_counts = [int(summary.get("search_count", 0) or 0) for summary in summaries]

    metrics = {
        "trust_r1/trace_count": float(count),
        "trust_r1/search_count_mean": sum(search_counts) / count,
        "trust_r1/fault_rate": fault_count / count,
        "trust_r1/searched_again_after_fault_rate": searched_again / fault_count if fault_count else 0.0,
        "trust_r1/changed_query_after_fault_rate": changed_query / fault_count if fault_count else 0.0,
        "trust_r1/duplicate_query_rate": duplicate_trajectory_count / count,
        "trust_r1/duplicate_query_count_mean": duplicate_count / count,
    }

    fault_types = Counter()
    for summary in summaries:
        for fault_type in summary.get("fault_types", []) or []:
            fault_types[str(fault_type)] += 1
    for fault_type, value in fault_types.items():
        metrics[f"trust_r1/fault_type/{fault_type}_count"] = float(value)

    return metrics
