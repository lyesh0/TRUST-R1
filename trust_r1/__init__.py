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
    "aggregate_trajectory_metrics",
    "compute_reward_breakdown",
    "compute_trust_reward",
    "parse_solution",
    "summarize_search_trace",
]
