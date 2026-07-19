#!/usr/bin/env python3
"""Evaluate Base/step25/step50 Stage1 action-format checkpoints."""

import argparse
import hashlib
import json
import re
from pathlib import Path


THRESHOLDS = {
    "valid_action_ratio": (">=", 0.95),
    "nonempty_query_ratio": (">=", 0.98),
    "invalid_recovery_rate": (">=", 0.80),
    "long_error_repetition_rate": ("<=", 0.02),
    "evidence_answer_rate": (">=", 0.85),
    "repetition_rate": ("<=", 0.05),
}


def parse_action(text):
    match = re.search(r"<(search|answer)>(.*?)</\1>", text, re.DOTALL)
    if not match:
        return None, ""
    return match.group(1), match.group(2).strip()


def has_repetition(text):
    if re.search(r"(.)\1{19,}", text, re.DOTALL):
        return True
    actions = re.findall(r"<(?:search|answer)>.*?</(?:search|answer)>", text, re.DOTALL)
    return len(actions) >= 3 and len(set(actions[-3:])) == 1


def passes_thresholds(metrics):
    for name, (operator, threshold) in THRESHOLDS.items():
        value = metrics[name]
        if operator == ">=" and value < threshold:
            return False
        if operator == "<=" and value > threshold:
            return False
    return True


def score_outputs(records, outputs):
    if len(records) != len(outputs):
        raise ValueError("records and outputs must align")
    valid = 0
    search_count = 0
    nonempty_search = 0
    recovery_total = 0
    recovery_valid = 0
    evidence_total = 0
    evidence_answer = 0
    error_repetitions = 0
    repetitions = 0
    for record, output in zip(records, outputs):
        action, content = parse_action(output)
        valid += int(action is not None and bool(content))
        if action == "search":
            search_count += 1
            nonempty_search += int(bool(content))
        if record["type"] == "C":
            recovery_total += 1
            recovery_valid += int(action == "search" and bool(content))
            error_repetitions += int("I am unsure. I will keep explaining" in output)
        if record["type"] == "B":
            evidence_total += 1
            evidence_answer += int(action == "answer" and bool(content))
        repetitions += int(has_repetition(output))
    count = len(records)
    return {
        "valid_action_ratio": valid / count if count else 0.0,
        "nonempty_query_ratio": nonempty_search / search_count if search_count else 0.0,
        "invalid_recovery_rate": recovery_valid / recovery_total if recovery_total else 0.0,
        "long_error_repetition_rate": error_repetitions / recovery_total if recovery_total else 0.0,
        "evidence_answer_rate": evidence_answer / evidence_total if evidence_total else 0.0,
        "repetition_rate": repetitions / count if count else 0.0,
    }


def _load_records(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _generate(model_path, records, batch_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    outputs = []
    for start in range(0, len(records), batch_size):
        prompts = [item["prompt"] for item in records[start:start + batch_size]]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_width = inputs["input_ids"].shape[1]
        outputs.extend(tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True))
    return outputs


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_hash(path):
    path = Path(path)
    digest = hashlib.sha256()
    names = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    found = False
    for name in names:
        candidate = path / name
        if candidate.is_file():
            found = True
            digest.update(name.encode("utf-8"))
            digest.update(candidate.read_bytes())
    return digest.hexdigest() if found else None


def evaluate(args):
    records = _load_records(args.data)
    candidates = [
        ("base", args.base_model),
        ("step25", str(Path(args.sft_output) / "checkpoint-25")),
        ("step50", str(Path(args.sft_output) / "checkpoint-50")),
    ]
    results = {}
    for name, path in candidates:
        metrics = score_outputs(records, _generate(path, records, args.batch_size))
        results[name] = {"path": path, "metrics": metrics, "passes": passes_thresholds(metrics)}
        print(name, json.dumps(metrics, sort_keys=True))
    selected = next((name for name in ("step25", "step50") if results[name]["passes"]), None)
    if selected is None:
        raise SystemExit("Neither step25 nor step50 passes all frozen C0 gates")
    artifact = {
        "schema_version": "stage1-c0-v2.1",
        "thresholds": THRESHOLDS,
        "candidates": results,
        "selected": selected,
        "selected_adapter_checkpoint": results[selected]["path"],
        "selection_reason": "earliest checkpoint passing every frozen action-format threshold",
        "c0": None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_c0(args):
    artifact_path = Path(args.output)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    c0 = Path(args.c0)
    config = c0 / "config.json"
    if not config.is_file():
        raise SystemExit("C0 config.json is missing")
    artifact["c0"] = {
        "path": str(c0),
        "config_sha256": _sha256(config),
        "tokenizer_sha256": _tokenizer_hash(c0),
    }
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-candidates", action="store_true")
    parser.add_argument("--verify-c0", action="store_true")
    parser.add_argument("--base-model", default="/root/autodl-tmp/models/Qwen2.5-3B")
    parser.add_argument("--sft-output", default="/root/autodl-tmp/TRUST-R1-stage1/checkpoints/sft_action")
    parser.add_argument("--data", default="/root/autodl-tmp/TRUST-R1-stage1/data/sft_train.jsonl")
    parser.add_argument("--c0", default="/root/autodl-tmp/TRUST-R1-stage1/checkpoints/C0")
    parser.add_argument("--output", default="artifacts/stage1/c0_selection.json")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.all_candidates == args.verify_c0:
        parser.error("select exactly one of --all-candidates or --verify-c0")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    evaluate(arguments) if arguments.all_candidates else verify_c0(arguments)
