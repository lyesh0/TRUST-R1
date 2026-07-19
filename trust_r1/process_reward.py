from __future__ import annotations

import re
import string
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

try:
    import torch
except ImportError:  # Pure text helpers are also used by the data builder.
    torch = None


@dataclass
class ProcessFeatures:
    step_rewards: torch.Tensor
    query_step_ids: torch.Tensor
    evidence_hits: List[bool]
    queries: List[str]
    information_blocks: List[str]
    alignment_valid: bool
    parse_error_count: int = 0


def normalize_to_tokens(text: str) -> List[str]:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()


def contains_alias(information: str, aliases: Sequence[str]) -> bool:
    information_tokens = normalize_to_tokens(information)
    for alias in aliases:
        alias_tokens = normalize_to_tokens(str(alias))
        if not alias_tokens or len(alias_tokens) > len(information_tokens):
            continue
        width = len(alias_tokens)
        for start in range(len(information_tokens) - width + 1):
            if information_tokens[start:start + width] == alias_tokens:
                return True
    return False


def first_hit_rewards(evidence_hits: Sequence[bool], max_search_steps: int) -> List[float]:
    rewards = [0.0] * max_search_steps
    previously_hit = False
    for step, hit in enumerate(evidence_hits[:max_search_steps]):
        hit = bool(hit)
        if hit and not previously_hit:
            rewards[step] = 1.0
        previously_hit = previously_hit or hit
    return rewards


def _as_token_list(token_ids: Any, valid_length: int) -> Tuple[List[int], torch.device]:
    if torch is None:
        raise ImportError("torch is required for process feature tensors")
    if isinstance(token_ids, torch.Tensor):
        device = token_ids.device
        values = token_ids.detach().cpu().tolist()
    else:
        device = torch.device("cpu")
        values = list(token_ids)
    valid_length = max(0, min(int(valid_length), len(values)))
    return [int(value) for value in values[:valid_length]], device


def _find_paired_spans(token_ids: Sequence[int], open_ids: Sequence[int], close_ids: Sequence[int]):
    if not open_ids or not close_ids:
        raise ValueError("tag token sequences must be non-empty")
    spans = []
    cursor = 0
    while cursor <= len(token_ids) - len(open_ids):
        if list(token_ids[cursor:cursor + len(open_ids)]) != list(open_ids):
            cursor += 1
            continue
        content_start = cursor + len(open_ids)
        close_start = content_start
        while close_start <= len(token_ids) - len(close_ids):
            if list(token_ids[close_start:close_start + len(close_ids)]) == list(close_ids):
                spans.append((content_start, close_start))
                cursor = close_start + len(close_ids)
                break
            close_start += 1
        else:
            return spans, 1
    return spans, 0


def _trace_queries(rollout_trace: Any) -> List[str]:
    if not isinstance(rollout_trace, dict):
        return []
    searches = rollout_trace.get("searches", []) or []
    return [str(search.get("query", "")) for search in searches if isinstance(search, dict)]


def build_process_features(
    response_token_ids,
    valid_response_length,
    gold_aliases,
    rollout_trace,
    tokenizer,
    max_search_steps=2,
) -> ProcessFeatures:
    if torch is None:
        raise ImportError("torch is required to build process feature tensors")
    if max_search_steps <= 0:
        raise ValueError("max_search_steps must be positive")
    token_ids, device = _as_token_list(response_token_ids, int(valid_response_length))
    response_length = int(response_token_ids.shape[0]) if isinstance(response_token_ids, torch.Tensor) else len(response_token_ids)
    query_step_ids = torch.zeros(response_length, dtype=torch.long, device=device)

    search_open = tokenizer.encode("<search>", add_special_tokens=False)
    search_close = tokenizer.encode("</search>", add_special_tokens=False)
    info_open = tokenizer.encode("<information>", add_special_tokens=False)
    info_close = tokenizer.encode("</information>", add_special_tokens=False)

    query_spans, query_errors = _find_paired_spans(token_ids, search_open, search_close)
    info_spans, info_errors = _find_paired_spans(token_ids, info_open, info_close)
    parse_error_count = query_errors + info_errors

    queries = [tokenizer.decode(token_ids[start:end], skip_special_tokens=False).strip()
               for start, end in query_spans]
    information_blocks = [tokenizer.decode(token_ids[start:end], skip_special_tokens=False).strip()
                          for start, end in info_spans]
    trace_queries = _trace_queries(rollout_trace)

    alignment_valid = (
        parse_error_count == 0
        and len(query_spans) == len(trace_queries)
        and len(info_spans) == len(trace_queries)
        and len(trace_queries) <= max_search_steps
    )
    if alignment_valid:
        for parsed, traced in zip(queries, trace_queries):
            if not normalize_to_tokens(parsed) or normalize_to_tokens(parsed) != normalize_to_tokens(traced):
                alignment_valid = False
                break

    if alignment_valid:
        for step, (start, end) in enumerate(query_spans, start=1):
            if start >= end:
                alignment_valid = False
                break
            query_step_ids[start:end] = step
    if not alignment_valid:
        query_step_ids.zero_()

    if isinstance(gold_aliases, str):
        gold_aliases = [gold_aliases]
    aliases = [str(alias) for alias in (gold_aliases or [])]
    evidence_hits = [contains_alias(block, aliases) for block in information_blocks[:max_search_steps]]
    evidence_hits.extend([False] * (max_search_steps - len(evidence_hits)))
    rewards = first_hit_rewards(evidence_hits, max_search_steps)
    step_rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

    return ProcessFeatures(
        step_rewards=step_rewards,
        query_step_ids=query_step_ids,
        evidence_hits=evidence_hits,
        queries=queries,
        information_blocks=information_blocks,
        alignment_valid=alignment_valid,
        parse_error_count=parse_error_count,
    )


def add_query_local_advantage(
    answer_advantages: torch.Tensor,
    step_rewards: torch.Tensor,
    query_step_ids: torch.Tensor,
    uids: Sequence[Any],
    loss_mask: torch.Tensor,
    weight=0.2,
    z_clip=2.0,
):
    if torch is None:
        raise ImportError("torch is required to add query-local advantage")
    if answer_advantages.shape != query_step_ids.shape or answer_advantages.shape != loss_mask.shape:
        raise ValueError("answer_advantages, query_step_ids, and loss_mask must have the same shape")
    if step_rewards.ndim != 2 or step_rewards.shape[0] != answer_advantages.shape[0]:
        raise ValueError("step_rewards must have shape [batch, max_search_steps]")
    if len(uids) != answer_advantages.shape[0]:
        raise ValueError("uids must align with the batch dimension")
    if weight < 0 or z_clip <= 0:
        raise ValueError("weight must be non-negative and z_clip must be positive")

    local_advantages = torch.zeros_like(answer_advantages)
    groups: Dict[Any, List[int]] = defaultdict(list)
    for index, uid in enumerate(uids):
        groups[uid].append(index)

    eligible_groups = 0
    informative_groups = 0
    max_steps = step_rewards.shape[1]
    for indices in groups.values():
        for step_index in range(max_steps):
            step_id = step_index + 1
            participants = []
            masks = {}
            for batch_index in indices:
                mask = (query_step_ids[batch_index] == step_id) & loss_mask[batch_index].bool()
                if mask.any().item():
                    participants.append(batch_index)
                    masks[batch_index] = mask
            if len(participants) < 2:
                continue
            eligible_groups += 1
            rewards = step_rewards[participants, step_index].to(dtype=torch.float32)
            if torch.all(rewards == rewards[0]).item():
                continue
            informative_groups += 1
            mean = rewards.mean()
            std = rewards.std(unbiased=True)
            z_values = torch.clamp((rewards - mean) / (std + 1e-6), min=-z_clip, max=z_clip)
            for participant, z_value in zip(participants, z_values):
                mask = masks[participant]
                token_count = mask.sum().to(dtype=answer_advantages.dtype)
                local_advantages[participant, mask] = weight * z_value.to(answer_advantages.dtype) / token_count

    combined = answer_advantages + local_advantages
    query_mask = (query_step_ids > 0) & loss_mask.bool()
    nonzero_query_tokens = (local_advantages != 0) & query_mask
    query_token_count = int(query_mask.sum().item())
    metrics = {
        "process/eligible_group_count": float(eligible_groups),
        "process/informative_group_count": float(informative_groups),
        "process/informative_group_rate": informative_groups / eligible_groups if eligible_groups else 0.0,
        "process/nonzero_query_token_rate": (
            nonzero_query_tokens.sum().item() / query_token_count if query_token_count else 0.0
        ),
        "process/local_adv_abs_mean": (
            local_advantages[query_mask].abs().mean().item() if query_token_count else 0.0
        ),
    }
    return combined, local_advantages, metrics
