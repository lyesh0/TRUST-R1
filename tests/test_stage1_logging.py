import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("ray")

from verl.trainer.main_ppo import RewardManager


def test_stage1_trajectory_write_does_not_require_trust_reward(tmp_path):
    manager = RewardManager(
        tokenizer=None,
        num_examine=0,
        trust_reward_config={"enabled": False},
        trust_logging_config={
            "enabled": True,
            "write_trajectories": True,
            "output_dir": str(tmp_path),
            "sample_limit_per_call": 32,
        },
        process_reward_config={"compute_diagnostics": True},
    )
    manager.last_stage1_records = [{
        "question_id": "q1",
        "first_hit_reward_by_step": [1.0, 0.0],
        "answer_correct": True,
    }]

    manager.write_stage1_records(split="train", step=1)

    record = json.loads((tmp_path / "train_trajectories.jsonl").read_text().strip())
    assert record["question_id"] == "q1"
    assert record["trainer_step"] == 1
    assert record["local_z_by_step"] == []
