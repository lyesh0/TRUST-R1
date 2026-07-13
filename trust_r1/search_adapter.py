from collections.abc import Mapping, Sequence
from typing import Any, List, Optional, Tuple

from .config import FaultConfig
from .faults import FaultEvent, RetrievalFaultInjector


def fault_config_from_mapping(values: Optional[Mapping[str, Any]], *, topk: Optional[int] = None) -> FaultConfig:
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
) -> Tuple[List[List[Any]], List[FaultEvent]]:
    injector = RetrievalFaultInjector(config)
    return injector.apply_batch(queries=queries, batch_documents=batch_results, step=step)
