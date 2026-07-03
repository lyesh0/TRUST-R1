"""TRUST-R1 utilities for robust search-agent experiments."""

from .config import FaultConfig, RewardConfig
from .faults import FaultEvent, RetrievalFaultInjector
from .metrics import TrajectoryMetrics, aggregate_trajectory_metrics
from .rewards import RewardBreakdown, compute_reward_breakdown
from .trajectory import SearchTurn, Trajectory, TrajectoryJsonlWriter

__all__ = [
    "FaultConfig",
    "FaultEvent",
    "RetrievalFaultInjector",
    "RewardBreakdown",
    "RewardConfig",
    "SearchTurn",
    "Trajectory",
    "TrajectoryJsonlWriter",
    "TrajectoryMetrics",
    "aggregate_trajectory_metrics",
    "compute_reward_breakdown",
]
