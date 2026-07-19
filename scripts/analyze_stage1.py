#!/usr/bin/env python3
"""Analyze S1-B0/S1-B1 validation trajectories with frozen denominators."""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path


def normalize_answer(text):
    text = str(text or "").lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction, aliases):
    prediction_tokens = normalize_answer(prediction).split()
    if isinstance(aliases, str):
        aliases = [aliases]
    best = 0.0
    for alias in aliases or []:
        answer_tokens = normalize_answer(alias).split()
        overlap = sum((Counter(prediction_tokens) & Counter(answer_tokens)).values())
        if not prediction_tokens or not answer_tokens or not overlap:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def analyze_records(records):
    if not records:
        raise ValueError("trajectory file is empty")
    count = len(records)
    totals = Counter()
    quadrants = Counter()
    f1_sum = 0.0
    for record in records:
        search_count = int(record.get("search_count", 0))
        hits = list(record.get("evidence_hit_by_step", []))[:search_count]
        any_hit = any(hits)
        first_hit = bool(hits[0]) if hits else False
        correct = bool(record.get("answer_correct", False))
        totals["valid"] += int(bool(record.get("valid_action", False)))
        totals["finished"] += int(record.get("finish_reason") == "answer")
        totals["correct"] += int(correct)
        totals["first_hit"] += int(first_hit)
        totals["any_hit"] += int(any_hit)
        totals["evidence_correct"] += int(any_hit and correct)
        totals["searches"] += search_count
        if search_count >= 2:
            totals["second_search"] += 1
            totals["incremental"] += int(not first_hit and len(hits) > 1 and hits[1])
        queries = [" ".join(str(query).lower().split()) for query in record.get("queries", [])]
        totals["repeated"] += int(len(queries) != len(set(queries)))
        f1_sum += token_f1(record.get("final_answer", ""), record.get("gold_aliases", []))
        quadrants[(any_hit, correct)] += 1
    metrics = {
        "trajectory_count": count,
        "valid_action_ratio": totals["valid"] / count,
        "finish_ratio": totals["finished"] / count,
        "exact_match": totals["correct"] / count,
        "token_f1": f1_sum / count,
        "first_search_success": totals["first_hit"] / count,
        "any_search_success": totals["any_hit"] / count,
        "incremental_evidence_rate": totals["incremental"] / totals["second_search"] if totals["second_search"] else 0.0,
        "average_search_count": totals["searches"] / count,
        "repeated_query_rate": totals["repeated"] / count,
        "evidence_utilization": totals["evidence_correct"] / totals["any_hit"] if totals["any_hit"] else 0.0,
        "retrieved_but_wrong": (totals["any_hit"] - totals["evidence_correct"]) / totals["any_hit"] if totals["any_hit"] else 0.0,
        "quadrants": {},
    }
    for any_hit, evidence_name in ((False, "no_evidence"), (True, "evidence")):
        for correct, outcome_name in ((False, "wrong"), (True, "correct")):
            value = quadrants[(any_hit, correct)]
            metrics["quadrants"]["%s_%s" % (evidence_name, outcome_name)] = {
                "count": value,
                "rate": value / count,
            }
    return metrics


def load_run(run_dir):
    results = {}
    for path in sorted(Path(run_dir).glob("validation_step_*.jsonl")):
        match = re.search(r"validation_step_(\d+)\.jsonl$", path.name)
        if not match:
            continue
        with path.open("r", encoding="utf-8") as file:
            records = [json.loads(line) for line in file if line.strip()]
        if len(records) != 100:
            raise ValueError("%s must contain exactly 100 validation trajectories" % path)
        results[int(match.group(1))] = analyze_records(records)
    return results


def main(args):
    report = {"S1-B0": load_run(args.b0), "S1-B1": load_run(args.b1)}
    required_steps = {0, 25, 50, 75, 100}
    for experiment, results in report.items():
        missing = required_steps - set(results)
        if missing:
            raise SystemExit("%s is missing validation steps: %s" % (experiment, sorted(missing)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for step in sorted(required_steps):
        b0 = report["S1-B0"][step]
        b1 = report["S1-B1"][step]
        print(
            "step=%d recall B0=%.3f B1=%.3f EM B0=%.3f B1=%.3f searches B0=%.3f B1=%.3f"
            % (step, b0["any_search_success"], b1["any_search_success"],
               b0["exact_match"], b1["exact_match"],
               b0["average_search_count"], b1["average_search_count"])
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", required=True)
    parser.add_argument("--b1", required=True)
    parser.add_argument("--output", default="artifacts/stage1/stage1_analysis.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
