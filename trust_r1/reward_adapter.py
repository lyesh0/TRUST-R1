import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

from .config import RewardConfig
from .rewards import RewardBreakdown, compute_reward_breakdown, count_duplicate_queries
from .text import contains_answer, normalize_answer


@dataclass(frozen=True)
class ParsedSolution:
    answer: Optional[str]
    search_queries: List[str]
    information_blocks: List[str]
    valid_format: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrustRewardResult:
    parsed: ParsedSolution
    reward: RewardBreakdown
    answer_correct: bool
    evidence_recovered: bool
    duplicate_query_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parsed": self.parsed.to_dict(),
            "reward": self.reward.to_dict(),
            "answer_correct": self.answer_correct,
            "evidence_recovered": self.evidence_recovered,
            "duplicate_query_count": self.duplicate_query_count,
        }


def extract_tag_blocks(text: str, tag: str) -> List[str]:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    return [match.strip() for match in re.findall(pattern, text, re.DOTALL)]


def extract_final_answer(text: str) -> Optional[str]:
    answers = extract_tag_blocks(text, "answer")
    if not answers:
        return None
    return answers[-1]


def parse_solution(solution_str: str) -> ParsedSolution:
    search_queries = extract_tag_blocks(solution_str, "search")
    information_blocks = extract_tag_blocks(solution_str, "information")
    answer = extract_final_answer(solution_str)
    valid_format = answer is not None
    return ParsedSolution(
        answer=answer,
        search_queries=search_queries,
        information_blocks=information_blocks,
        valid_format=valid_format,
    )


def answer_matches(answer: Optional[str], targets: Union[str, List[str]]) -> bool:
    if answer is None:
        return False
    if isinstance(targets, str):
        targets = [targets]
    normalized_answer = normalize_answer(answer)
    return any(normalized_answer == normalize_answer(target) for target in targets)


def compute_trust_reward(
    *,
    solution_str: str,
    ground_truth: Dict[str, Any],
    had_fault: bool = False,
    changed_query_after_fault: bool = False,
    config: Optional[RewardConfig] = None,
) -> TrustRewardResult:
    parsed = parse_solution(solution_str)
    targets = ground_truth["target"]
    answer_correct = answer_matches(parsed.answer, targets)
    evidence_recovered = any(contains_answer(block, targets) for block in parsed.information_blocks)
    duplicate_query_count = count_duplicate_queries(parsed.search_queries)
    reward = compute_reward_breakdown(
        answer_correct=answer_correct,
        valid_format=parsed.valid_format,
        had_fault=had_fault,
        changed_query_after_fault=changed_query_after_fault,
        evidence_recovered=evidence_recovered,
        duplicate_query_count=duplicate_query_count,
        config=config,
    )
    return TrustRewardResult(
        parsed=parsed,
        reward=reward,
        answer_correct=answer_correct,
        evidence_recovered=evidence_recovered,
        duplicate_query_count=duplicate_query_count,
    )
