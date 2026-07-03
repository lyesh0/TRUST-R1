from trust_r1.text import contains_answer, normalize_answer


def test_normalize_answer_matches_search_r1_style():
    assert normalize_answer("The Ada-Lovelace!") == "adalovelace"


def test_contains_answer():
    assert contains_answer("Ada Lovelace wrote notes.", ["Lovelace"])
    assert not contains_answer("Ada Lovelace wrote notes.", ["Turing"])
