from trust_r1.config import FaultConfig
from trust_r1.faults import RetrievalFaultInjector, extract_doc_ids


def docs():
    return [
        {"id": "d1", "contents": "one"},
        {"id": "d2", "contents": "two"},
        {"id": "d3", "contents": "three"},
    ]


def test_clean_fault_returns_original_documents():
    injector = RetrievalFaultInjector(FaultConfig(enabled=False, mode="clean", fault_rate=0.0, seed=1))
    returned, event = injector.apply("query", docs())
    assert returned == docs()
    assert event.applied is False
    assert event.fault_type == "clean"


def test_empty_fault_returns_no_documents():
    injector = RetrievalFaultInjector(FaultConfig(enabled=True, mode="empty", fault_rate=1.0, seed=1))
    returned, event = injector.apply("query", docs())
    assert returned == []
    assert event.applied is True
    assert event.fault_type == "empty"
    assert event.original_count == 3
    assert event.returned_count == 0


def test_drop_top_removes_first_document():
    injector = RetrievalFaultInjector(FaultConfig(enabled=True, mode="drop_top", fault_rate=1.0, seed=1))
    returned, event = injector.apply("query", docs())
    assert extract_doc_ids(returned) == ["d2", "d3"]
    assert event.fault_type == "drop_top"


def test_duplicate_fault_repeats_first_document():
    injector = RetrievalFaultInjector(FaultConfig(enabled=True, mode="duplicate", fault_rate=1.0, seed=1))
    returned, event = injector.apply("query", docs())
    assert extract_doc_ids(returned) == ["d1", "d1", "d1"]
    assert event.fault_type == "duplicate"


def test_mixed_fault_is_deterministic_for_same_query_and_step():
    config = FaultConfig(enabled=True, mode="mixed", fault_rate=1.0, seed=123)
    injector = RetrievalFaultInjector(config)
    first, first_event = injector.apply("same query", docs(), step=2)
    second, second_event = injector.apply("same query", docs(), step=2)
    assert first == second
    assert first_event == second_event
