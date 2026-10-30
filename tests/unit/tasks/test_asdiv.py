from lighteval.tasks.tasks.asdiv import asdiv_prompt


def test_asdiv_prompt_keeps_full_gold_answer():
    # asdiv stored choices as a bare string, so Doc.get_golds() indexed the
    # string and truncated a multi-character answer to its first character
    # (e.g. "35" -> "3"). choices must be a list so the gold answer survives.
    line = {
        "body": "There are some apples.",
        "question": "How many apples are there?",
        "answer": "35 (apples)",
    }
    doc = asdiv_prompt(line)
    assert doc.choices == ["35"]
    assert doc.get_golds() == ["35"]
