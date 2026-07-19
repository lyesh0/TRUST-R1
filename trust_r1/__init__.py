"""TRUST-R1 utilities for robust search-agent experiments."""

from .config import FaultConfig, RewardConfig
from .faults import FaultEvent, RetrievalFaultInjector
from .metrics import TrajectoryMetrics, aggregate_trace_summary_metrics, aggregate_trajectory_metrics
from .reward_adapter import ParsedSolution, TrustRewardResult, compute_trust_reward, parse_solution
from .rewards import RewardBreakdown, compute_reward_breakdown
from .rollout_logging import RolloutSearchTrace, RolloutTrace, RolloutTraceRecorder, summarize_search_trace
from .trajectory import SearchTurn, Trajectory, TrajectoryJsonlWriter

__all__ = [
    "FaultConfig",
    "FaultEvent",
    "ParsedSolution",
    "ProcessFeatures",
    "RetrievalFaultInjector",
    "RewardBreakdown",
    "RewardConfig",
    "RolloutSearchTrace",
    "RolloutTrace",
    "RolloutTraceRecorder",
    "SearchTurn",
    "Trajectory",
    "TrajectoryJsonlWriter",
    "TrajectoryMetrics",
    "TrustRewardResult",
    "aggregate_trace_summary_metrics",
    "add_query_local_advantage",
    "build_process_features",
    "aggregate_trajectory_metrics",
    "compute_reward_breakdown",
    "compute_trust_reward",
    "parse_solution",
    "summarize_search_trace",
]


def __getattr__(name):
    if name in {"ProcessFeatures", "add_query_local_advantage", "build_process_features"}:
        from . import process_reward
        return getattr(process_reward, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
