#!/usr/bin/env python3
"""Train and merge the frozen 50-step Stage1 action-format LoRA."""

import argparse
import json
from pathlib import Path


def build_assistant_only_features(tokenizer, prompt, target, max_length=1024):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        target_ids = target_ids + [int(eos_token_id)]
    if len(target_ids) > max_length:
        raise ValueError("target alone exceeds max_length")
    available_prompt_length = max_length - len(target_ids)
    prompt_ids = prompt_ids[-available_prompt_length:] if available_prompt_length else []
    input_ids = prompt_ids + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + list(target_ids),
    }


def validate_c0_directory(path):
    path = Path(path)
    required = [path / "config.json"]
    weight_candidates = [
        path / "model.safetensors",
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin",
        path / "pytorch_model.bin.index.json",
    ]
    if not all(item.is_file() for item in required) or not any(item.is_file() for item in weight_candidates):
        raise ValueError("C0 must contain a full Hugging Face config and merged model weights")
    if (path / "adapter_config.json").exists():
        raise ValueError("C0 still looks like an adapter-only directory")
    return True


class Stage1SFTDataset:
    def __init__(self, path, tokenizer, max_length):
        self.examples = []
        with Path(path).open("r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                self.examples.append(build_assistant_only_features(
                    tokenizer, record["prompt"], record["target"], max_length=max_length
                ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def _collator(tokenizer):
    import torch

    pad_id = tokenizer.pad_token_id

    def collate(examples):
        width = max(len(item["input_ids"]) for item in examples)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in examples:
            padding = width - len(item["input_ids"])
            result["input_ids"].append(item["input_ids"] + [pad_id] * padding)
            result["attention_mask"].append(item["attention_mask"] + [0] * padding)
            result["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}

    return collate


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    ))
    dataset = Stage1SFTDataset(args.data, tokenizer, args.max_length)
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=50,
        learning_rate=1e-4,
        warmup_steps=5,
        bf16=True,
        max_grad_norm=1.0,
        save_steps=25,
        save_strategy="steps",
        logging_steps=1,
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        report_to=["wandb"],
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=_collator(tokenizer),
    )
    trainer.train()


def merge(args):
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    peft_config = PeftConfig.from_pretrained(args.checkpoint)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint).merge_and_unload()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path).save_pretrained(output)
    validate_c0_directory(output)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--base-model", required=True)
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--max-length", type=int, default=1024)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--checkpoint", required=True)
    merge_parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments) if arguments.command == "train" else merge(arguments)
