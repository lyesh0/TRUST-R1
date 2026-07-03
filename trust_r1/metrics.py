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
