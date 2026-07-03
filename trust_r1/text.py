import re
import string


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def contains_answer(text: str, answers: str | list[str]) -> bool:
    if isinstance(answers, str):
        answers = [answers]
    normalized_text = normalize_answer(text)
    return any(normalize_answer(answer) in normalized_text for answer in answers)
