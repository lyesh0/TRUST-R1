#!/usr/bin/env python3
"""Build and verify the frozen TRUST-R1 Stage1 datasets."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from trust_r1.process_reward import contains_alias


TYPE_COUNTS = {"A": 400, "B": 160, "C": 240}
RL_TRAIN_COUNT = 3200
RL_VALIDATION_COUNT = 100
DIAGNOSTIC_COUNT = 50


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_select_indices(size, count, seed):
    if size < count:
        raise ValueError("dataset has %d rows, but %d are required" % (size, count))
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return indices[:count]


def _prompt_text(row):
    prompt = row["prompt"]
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, list) and prompt:
        return str(prompt[0].get("content", ""))
    return str(prompt)


def _question_text(row):
    if "question" in row and row["question"]:
        return str(row["question"]).strip()
    prompt = _prompt_text(row)
    marker = "Question:"
    return prompt.rsplit(marker, 1)[-1].strip() if marker in prompt else prompt.strip()


def _aliases(row):
    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth") or {}
    aliases = ground_truth.get("target", [])
    return [aliases] if isinstance(aliases, str) else list(aliases)


def _retrieve_passages(questions, retriever_url, batch_size=64, topk=3):
    import requests

    outputs = []
    for start in range(0, len(questions), batch_size):
        query_batch = questions[start:start + batch_size]
        response = requests.post(
            retriever_url,
            json={"queries": query_batch, "topk": topk, "return_scores": True},
            timeout=120,
        )
        response.raise_for_status()
        results = response.json()["result"]
        if len(results) != len(query_batch):
            raise RuntimeError("retriever result count does not match query count")
        for result in results:
            passages = []
            for item in result:
                document = item.get("document", item)
                passages.append(str(document.get("contents", "")))
            outputs.append("\n".join(passages))
    return outputs


def build_sft_records(rows, retrieved_passages):
    if len(rows) != len(retrieved_passages):
        raise ValueError("rows and retrieved_passages must align")
    hit_indices = [
        index for index, (row, passage) in enumerate(zip(rows, retrieved_passages))
        if contains_alias(passage, _aliases(row))
    ]
    if len(hit_indices) < TYPE_COUNTS["B"]:
        raise RuntimeError(
            "Type B requires %d top-3 hits, but only %d were found"
            % (TYPE_COUNTS["B"], len(hit_indices))
        )

    selected_b = hit_indices[:TYPE_COUNTS["B"]]
    selected = set(selected_b)
    remaining = [index for index in range(len(rows)) if index not in selected]
    selected_a = remaining[:TYPE_COUNTS["A"]]
    selected.update(selected_a)
    selected_c = [index for index in remaining if index not in selected][:TYPE_COUNTS["C"]]
    if len(selected_a) != TYPE_COUNTS["A"] or len(selected_c) != TYPE_COUNTS["C"]:
        raise RuntimeError("not enough disjoint questions to construct Stage1 SFT data")

    records = []
    for kind, indices in (("A", selected_a), ("B", selected_b), ("C", selected_c)):
        for index in indices:
            row = rows[index]
            prompt = _prompt_text(row)
            question = _question_text(row).rstrip(" ?")
            aliases = _aliases(row)
            answer = str(aliases[0]) if aliases else ""
            if kind == "A":
                target = "<think>I need external evidence.</think><search>%s</search>" % question
            elif kind == "B":
                prompt += (
                    "<think>I need external evidence.</think><search>%s</search>"
                    "\n<information>%s</information>\n" % (question, retrieved_passages[index])
                )
                target = "<think>The evidence is sufficient.</think><answer>%s</answer>" % answer
            else:
                prompt += "<think>I am unsure. I will keep explaining without an action.</think>\n"
                prompt += '<tool_error code="INVALID_ACTION"/>\n'
                target = "<think>I should issue a valid action.</think><search>%s</search>" % question
            records.append({
                "type": kind,
                "question_id": str(row.get("stage1_question_id", index)),
                "prompt": prompt,
                "target": target,
            })
    return records, len(hit_indices)


def _write_jsonl(path, records):
    def json_default(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        raise TypeError("not JSON serializable: %r" % (type(value),))

    with Path(path).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(
                record, ensure_ascii=False, sort_keys=True, default=json_default
            ) + "\n")


def build(args):
    import pandas as pd

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_source = source_dir / "train.parquet"
    validation_source = source_dir / "test.parquet"
    train = pd.read_parquet(train_source)
    validation = pd.read_parquet(validation_source)

    train_indices = stable_select_indices(len(train), RL_TRAIN_COUNT, args.seed)
    validation_indices = stable_select_indices(len(validation), RL_VALIDATION_COUNT, args.seed + 1)
    train = train.iloc[train_indices].copy().reset_index(drop=True)
    validation = validation.iloc[validation_indices].copy().reset_index(drop=True)
    train["stage1_question_id"] = ["train:%d" % index for index in train_indices]
    validation["stage1_question_id"] = ["test:%d" % index for index in validation_indices]
    train["extra_info"] = [
        dict((value or {}), index=question_id)
        for value, question_id in zip(train["extra_info"], train["stage1_question_id"])
    ]
    validation["extra_info"] = [
        dict((value or {}), index=question_id)
        for value, question_id in zip(validation["extra_info"], validation["stage1_question_id"])
    ]

    rows = train.to_dict("records")
    questions = [_question_text(row).rstrip(" ?") for row in rows]
    retrieved = _retrieve_passages(questions, args.retriever_url, topk=args.topk)
    sft_records, type_b_hit_count = build_sft_records(rows, retrieved)

    paths = {
        "rl_train": output_dir / "rl_train.parquet",
        "rl_validation": output_dir / "rl_validation.parquet",
        "diagnostic": output_dir / "diagnostic_50.jsonl",
        "sft": output_dir / "sft_train.jsonl",
    }
    train.to_parquet(paths["rl_train"], index=False)
    validation.to_parquet(paths["rl_validation"], index=False)
    _write_jsonl(paths["diagnostic"], validation.head(DIAGNOSTIC_COUNT).to_dict("records"))
    _write_jsonl(paths["sft"], sft_records)

    manifest = {
        "schema_version": "stage1-v2.1",
        "seed": args.seed,
        "retriever_url": args.retriever_url,
        "topk": args.topk,
        "source_files": {
            "train.parquet": sha256_file(train_source),
            "test.parquet": sha256_file(validation_source),
        },
        "question_ids": {
            "train": train["stage1_question_id"].tolist(),
            "validation": validation["stage1_question_id"].tolist(),
            "diagnostic": validation["stage1_question_id"].head(DIAGNOSTIC_COUNT).tolist(),
        },
        "counts": {
            "rl_train": len(train),
            "rl_validation": len(validation),
            "diagnostic": DIAGNOSTIC_COUNT,
            "sft_by_type": TYPE_COUNTS,
            "type_b_top3_hit_candidates": type_b_hit_count,
        },
        "output_files": {path.name: sha256_file(path) for path in paths.values()},
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify(args):
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir)
    for name, expected in manifest["output_files"].items():
        path = data_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit("hash mismatch or missing file: %s" % path)
    train_ids = set(manifest["question_ids"]["train"])
    validation_ids = set(manifest["question_ids"]["validation"])
    diagnostic_ids = set(manifest["question_ids"]["diagnostic"])
    if train_ids & validation_ids or not diagnostic_ids <= validation_ids:
        raise SystemExit("manifest split isolation check failed")
    print("Stage1 data verified: %s" % data_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source-dir", required=True)
    build_parser.add_argument("--output-dir", required=True)
    build_parser.add_argument("--manifest", required=True)
    build_parser.add_argument("--retriever-url", required=True)
    build_parser.add_argument("--seed", type=int, default=42)
    build_parser.add_argument("--topk", type=int, default=3)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--data-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build(arguments) if arguments.command == "build" else verify(arguments)
