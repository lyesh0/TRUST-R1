from pathlib import Path

import pytest

from scripts.sft_action_lora import build_assistant_only_features, validate_c0_directory


class CharTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def test_assistant_only_labels_mask_prompt_and_keep_target():
    features = build_assistant_only_features(CharTokenizer(), "prompt", "target", max_length=32)
    assert features["labels"][:6] == [-100] * 6
    assert features["labels"][6:] == [ord(char) for char in "target"] + [0]


def test_left_truncation_preserves_entire_target_and_recent_prompt():
    features = build_assistant_only_features(CharTokenizer(), "abcdefghij", "XYZ", max_length=7)
    assert features["input_ids"] == [ord(char) for char in "hijXYZ"] + [0]
    assert features["labels"][:3] == [-100] * 3


def test_target_larger_than_context_is_rejected():
    with pytest.raises(ValueError):
        build_assistant_only_features(CharTokenizer(), "p", "target", max_length=3)


def test_target_exactly_filling_context_drops_entire_prompt():
    features = build_assistant_only_features(CharTokenizer(), "prompt", "XY", max_length=3)
    assert features["input_ids"] == [ord("X"), ord("Y"), 0]
    assert features["labels"] == features["input_ids"]


def test_c0_validation_rejects_adapter_only_directory(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError):
        validate_c0_directory(tmp_path)


def test_c0_validation_accepts_merged_model(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    assert validate_c0_directory(Path(tmp_path))
