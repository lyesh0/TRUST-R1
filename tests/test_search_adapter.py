from trust_r1.config import FaultConfig
from trust_r1.search_adapter import apply_retrieval_faults, fault_config_from_mapping


def search_results():
    return [
        [{"document": {"contents": "Doc 1"}}, {"document": {"contents": "Doc 2"}}],
        [{"document": {"contents": "Doc A"}}, {"document": {"contents": "Doc B"}}],
    ]


def test_fault_config_from_mapping_defaults_to_clean():
    config = fault_config_from_mapping(None, topk=3)
    assert config.enabled is False
    assert config.mode == "clean"
    assert config.fault_rate == 0.0
    assert config.topk == 3


def test_fault_config_from_mapping_reads_values():
    config = fault_config_from_mapping(
        {"enabled": True, "mode": "empty", "fault_rate": 0.5, "seed": 7},
        topk=2,
    )
    assert config.enabled is True
    assert config.mode == "empty"
    assert config.fault_rate == 0.5
    assert config.seed == 7
    assert config.topk == 2


def test_apply_retrieval_faults_keeps_clean_results_by_default():
    results, events = apply_retrieval_faults(
        queries=["q1", "q2"],
        batch_results=search_results(),
        config=FaultConfig(enabled=False, mode="clean", fault_rate=0.0),
    )
    assert results == search_results()
    assert [event.applied for event in events] == [False, False]


def test_apply_retrieval_faults_can_empty_batch_results():
    results, events = apply_retrieval_faults(
        queries=["q1", "q2"],
        batch_results=search_results(),
        config=FaultConfig(enabled=True, mode="empty", fault_rate=1.0),
    )
    assert results == [[], []]
    assert [event.fault_type for event in events] == ["empty", "empty"]
