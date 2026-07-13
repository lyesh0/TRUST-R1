import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class SearchTurn:
    step: int
    action: str
    query: str = ""
    fault_enabled: bool = False
    fault_type: str = "clean"
    retrieved_doc_ids: List[str] = field(default_factory=list)
    observation_chars: int = 0
    valid_action: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    run_id: str
    sample_id: str
    question: str
    gold_answer: Optional[Union[str, List[str]]] = None
    final_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    turns: List[SearchTurn] = field(default_factory=list)
    reward: Dict[str, float] = field(default_factory=dict)

    def add_turn(self, turn: SearchTurn) -> None:
        self.turns.append(turn)

    @property
    def had_fault(self) -> bool:
        return any(turn.fault_enabled and turn.fault_type != "clean" for turn in self.turns)

    @property
    def searched_again_after_fault(self) -> bool:
        first_fault = None
        for idx, turn in enumerate(self.turns):
            if turn.fault_enabled and turn.fault_type != "clean":
                first_fault = idx
                break
        if first_fault is None:
            return False
        return any(turn.action == "search" for turn in self.turns[first_fault + 1 :])

    @property
    def changed_query_after_fault(self) -> bool:
        first_fault_query = None
        first_fault_idx = None
        for idx, turn in enumerate(self.turns):
            if turn.fault_enabled and turn.fault_type != "clean":
                first_fault_query = turn.query.strip().lower()
                first_fault_idx = idx
                break
        if first_fault_idx is None or not first_fault_query:
            return False
        for turn in self.turns[first_fault_idx + 1 :]:
            if turn.action == "search" and turn.query.strip().lower() != first_fault_query:
                return True
        return False

    def recovery_dict(self) -> dict[str, bool]:
        return {
            "had_fault": self.had_fault,
            "searched_again_after_fault": self.searched_again_after_fault,
            "changed_query_after_fault": self.changed_query_after_fault,
            "answered_correctly_after_fault": bool(self.had_fault and self.is_correct),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "final_answer": self.final_answer,
            "is_correct": self.is_correct,
            "turns": [turn.to_dict() for turn in self.turns],
            "recovery": self.recovery_dict(),
            "reward": self.reward,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class TrajectoryJsonlWriter:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, trajectory: Trajectory) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(trajectory.to_json() + "\n")
