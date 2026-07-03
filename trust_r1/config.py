from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class FaultConfig:
    enabled: bool = False
    mode: str = "clean"
    fault_rate: float = 0.0
    seed: int = 0
    topk: int | None = None

    def __post_init__(self):
        if not 0.0 <= self.fault_rate <= 1.0:
            raise ValueError("fault_rate must be between 0 and 1")


@dataclass(frozen=True)
class RewardConfig:
    answer_weight: float = 1.0
    format_weight: float = 0.0
    recovery_weight: float = 0.2
    duplicate_penalty_weight: float = 0.1
    invalid_penalty_weight: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]):
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: values[key] for key in allowed if key in values})
