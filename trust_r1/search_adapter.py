from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .config import FaultConfig
from .faults import FaultEvent, RetrievalFaultInjector


def fault_config_from_mapping(values: Mapping[str, Any] | None, *, topk: int | None = None) -> FaultConfig:
    if values is None:
        return FaultConfig(topk=topk)
    return FaultConfig(
        enabled=bool(values.get("enabled", False)),
        mode=str(values.get("mode", "clean")),
        fault_rate=float(values.get("fault_rate", 0.0)),
        seed=int(values.get("seed", 0)),
        topk=topk,
    )


def apply_retrieval_faults(
    *,
    queries: Sequence[str],
    batch_results: Sequence[Sequence[Any]],
    config: FaultConfig,
    step: int = 0,
) -> tuple[list[list[Any]], list[FaultEvent]]:
    injector = RetrievalFaultInjector(config)
    return injector.apply_batch(queries=queries, batch_documents=batch_results, step=step)
