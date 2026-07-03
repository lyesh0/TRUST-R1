from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .faults import FaultEvent
from .rewards import count_duplicate_queries


@dataclass
class RolloutSearchTrace:
    sample_index: int
    step: int
    query: str
    fault_enabled: bool
    fault_applied: bool
    fault_type: str
    original_count: int
    returned_count: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RolloutTrace:
    sample_index: int
    searches: list[RolloutSearchTrace] = field(default_factory=list)

    def add_search(self, search: RolloutSearchTrace) -> None:
        self.searches.append(search)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "searches": [search.to_dict() for search in self.searches],
        }


def summarize_search_trace(trace: RolloutTrace) -> dict[str, Any]:
    queries = [search.query for search in trace.searches]
    first_fault_index = None
    first_fault_query = None
    for idx, search in enumerate(trace.searches):
        if search.fault_enabled and search.fault_type != "clean":
            first_fault_index = idx
            first_fault_query = search.query.strip().lower()
            break

    changed_query_after_fault = False
    searched_again_after_fault = False
    if first_fault_index is not None:
        for search in trace.searches[first_fault_index + 1 :]:
            searched_again_after_fault = True
            if first_fault_query and search.query.strip().lower() != first_fault_query:
                changed_query_after_fault = True
                break

    return {
        "sample_index": trace.sample_index,
        "search_count": len(trace.searches),
        "had_fault": first_fault_index is not None,
        "searched_again_after_fault": searched_again_after_fault,
        "changed_query_after_fault": changed_query_after_fault,
        "duplicate_query_count": count_duplicate_queries(queries),
        "fault_types": [search.fault_type for search in trace.searches if search.fault_type != "clean"],
    }


class RolloutTraceRecorder:
    """Record lightweight per-sample search traces during rollout.

    This class deliberately records only search-side information. Final answers,
    correctness, and rewards are added later in the reward/eval path where that
    information is available.
    """

    def __init__(self, batch_size: int):
        self.traces = [RolloutTrace(sample_index=i) for i in range(batch_size)]

    def record_searches(
        self,
        *,
        sample_indices: Sequence[int],
        queries: Sequence[str],
        events: Sequence[FaultEvent],
        step: int,
    ) -> None:
        if not (len(sample_indices) == len(queries) == len(events)):
            raise ValueError("sample_indices, queries, and events must have the same length")

        for sample_index, query, event in zip(sample_indices, queries, events):
            if sample_index < 0 or sample_index >= len(self.traces):
                raise IndexError(f"sample_index out of range: {sample_index}")
            self.traces[sample_index].add_search(
                RolloutSearchTrace(
                    sample_index=sample_index,
                    step=step,
                    query=query,
                    fault_enabled=event.enabled,
                    fault_applied=event.applied,
                    fault_type=event.fault_type,
                    original_count=event.original_count,
                    returned_count=event.returned_count,
                    seed=event.seed,
                )
            )

    def to_meta(self) -> list[dict[str, Any]]:
        return [trace.to_dict() for trace in self.traces]

    def to_summary(self) -> list[dict[str, Any]]:
        return [summarize_search_trace(trace) for trace in self.traces]
