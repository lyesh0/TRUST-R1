import json

from trust_r1.trajectory import SearchTurn, Trajectory, TrajectoryJsonlWriter


def test_trajectory_recovery_flags():
    trajectory = Trajectory(run_id="run", sample_id="s1", question="q", is_correct=True)
    trajectory.add_turn(SearchTurn(step=1, action="search", query="first", fault_enabled=True, fault_type="empty"))
    trajectory.add_turn(SearchTurn(step=2, action="search", query="second", fault_enabled=False, fault_type="clean"))

    recovery = trajectory.recovery_dict()
    assert recovery["had_fault"] is True
    assert recovery["searched_again_after_fault"] is True
    assert recovery["changed_query_after_fault"] is True
    assert recovery["answered_correctly_after_fault"] is True


def test_trajectory_jsonl_writer(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    writer = TrajectoryJsonlWriter(path)
    trajectory = Trajectory(run_id="run", sample_id="s1", question="q")
    trajectory.add_turn(SearchTurn(step=1, action="search", query="q1"))

    writer.write(trajectory)
    loaded = json.loads(path.read_text().strip())
    assert loaded["run_id"] == "run"
    assert loaded["turns"][0]["query"] == "q1"
