from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .config import RewardConfig


@dataclass(frozen=True)
class RewardBreakdown:
    answer: float = 0.0
    format: float = 0.0
    recovery: float = 0.0
    duplicate_penalty: float = 0.0
    invalid_penalty: float = 0.0

    @property
    def total(self) -> float:
        return self.answer + self.format + self.recovery - self.duplicate_penalty - self.invalid_penalty

    def to_dict(self) -> Dict[str, float]:
        values = asdict(self)
        values["total"] = self.total
        return values


def compute_reward_breakdown(
    *,
    answer_correct: bool,
    valid_format: bool = True,
    had_fault: bool = False,
    changed_query_after_fault: bool = False,
    evidence_recovered: bool = False,
    duplicate_query_count: int = 0,
    invalid_action_count: int = 0,
    config: Optional[RewardConfig] = None,
) -> RewardBreakdown:
    cfg = config or RewardConfig()
    answer = cfg.answer_weight if answer_correct else 0.0
    fmt = cfg.format_weight if valid_format else 0.0
    recovery = 0.0
    if had_fault and changed_query_after_fault and evidence_recovered and answer_correct:
        recovery = cfg.recovery_weight
    duplicate_penalty = max(0, duplicate_query_count) * cfg.duplicate_penalty_weight
    invalid_penalty = max(0, invalid_action_count) * cfg.invalid_penalty_weight
    return RewardBreakdown(
        answer=answer,
        format=fmt,
        recovery=recovery,
        duplicate_penalty=duplicate_penalty,
        invalid_penalty=invalid_penalty,
    )


def count_duplicate_queries(queries: List[str]) -> int:
    seen = set()
    duplicates = 0
    for query in queries:
        key = " ".join(query.lower().split())
        if not key:
            continue
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates
