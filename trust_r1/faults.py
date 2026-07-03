import hashlib
import random
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence

from .config import FaultConfig


@dataclass(frozen=True)
class FaultEvent:
    enabled: bool
    applied: bool
    fault_type: str
    query: str
    original_count: int
    returned_count: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RetrievalFaultInjector:
    """Apply deterministic retrieval faults to Search-R1 style result lists.

    The injector is intentionally independent from FAISS/BM25 so it can be tested
    locally and inserted either server-side or rollout-side later.
    """

    VALID_MODES = {"clean", "empty", "drop_top", "duplicate", "mixed"}
    MIXED_MODES = ("empty", "drop_top", "duplicate")

    def __init__(self, config: FaultConfig):
        if config.mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported fault mode: {config.mode}")
        self.config = config

    def apply(self, query: str, documents: Sequence[Any], step: int = 0) -> tuple[list[Any], FaultEvent]:
        docs = list(documents)
        fault_type = self._choose_fault_type(query=query, step=step)
        should_apply = self.config.enabled and fault_type != "clean" and self._should_apply(query, step)

        if not should_apply:
            returned = self._limit_topk(docs)
            return returned, FaultEvent(
                enabled=self.config.enabled,
                applied=False,
                fault_type="clean",
                query=query,
                original_count=len(docs),
                returned_count=len(returned),
                seed=self.config.seed,
            )

        returned = self._apply_fault(fault_type, docs)
        returned = self._limit_topk(returned)
        return returned, FaultEvent(
            enabled=True,
            applied=True,
            fault_type=fault_type,
            query=query,
            original_count=len(docs),
            returned_count=len(returned),
            seed=self.config.seed,
        )

    def apply_batch(self, queries: Sequence[str], batch_documents: Sequence[Sequence[Any]], step: int = 0):
        if len(queries) != len(batch_documents):
            raise ValueError("queries and batch_documents must have the same length")
        outputs, events = [], []
        for idx, (query, docs) in enumerate(zip(queries, batch_documents)):
            result, event = self.apply(query=query, documents=docs, step=step + idx)
            outputs.append(result)
            events.append(event)
        return outputs, events

    def _apply_fault(self, fault_type: str, docs: list[Any]) -> list[Any]:
        if fault_type == "empty":
            return []
        if fault_type == "drop_top":
            return docs[1:]
        if fault_type == "duplicate":
            if not docs:
                return []
            return [docs[0] for _ in range(len(docs))]
        if fault_type == "clean":
            return docs
        raise ValueError(f"Unsupported fault type: {fault_type}")

    def _choose_fault_type(self, query: str, step: int) -> str:
        if self.config.mode != "mixed":
            return self.config.mode
        rng = self._rng(query=query, step=step, salt="mode")
        return rng.choice(self.MIXED_MODES)

    def _should_apply(self, query: str, step: int) -> bool:
        rng = self._rng(query=query, step=step, salt="apply")
        return rng.random() < self.config.fault_rate

    def _limit_topk(self, docs: list[Any]) -> list[Any]:
        if self.config.topk is None:
            return docs
        return docs[: self.config.topk]

    def _rng(self, query: str, step: int, salt: str) -> random.Random:
        key = f"{self.config.seed}:{step}:{salt}:{query}".encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()[:16]
        return random.Random(int(digest, 16))


def extract_doc_ids(documents: Iterable[Any]) -> list[str]:
    ids = []
    for idx, doc in enumerate(documents):
        if isinstance(doc, dict):
            value = doc.get("id") or doc.get("docid") or doc.get("document_id")
            if value is not None:
                ids.append(str(value))
                continue
        ids.append(str(idx))
    return ids
