from lighteval.tasks.templates.multichoice_config import MCQInput
from lighteval.utils.language import Language


def test_multichoice_prompt():
    # Define test input
    test_input = MCQInput(
        question="What is the capital of France?",
        choices=["London", "Paris", "Berlin", "Madrid"],
        gold_idxs=1,
        context="France is a country in Western Europe.",
        instruction="Please answer the following question about geography.",
    )

    # Test mcq_prompt_functions directly
    from lighteval.tasks.templates.formulation import MCFFormulation
    from lighteval.tasks.templates.multichoice_config import mcq_prompt_functions

    # Generate prompt using mcq_prompt_functions
    doc = mcq_prompt_functions(test_input, "test_task", Language.english, MCFFormulation())

    assert (
        doc.query
        == """\
Please answer the following question about geography.
France is a country in Western Europe.
Question: What is the capital of France?
 A. London
 B. Paris
 C. Berlin
 D. Madrid
Answer:\
"""
    )

    assert doc.unconditioned_query == "Answer:"
    assert doc.choices == [" A", " B", " C", " D"]


def test_multichoice_prompt_no_context():
    # Define test input
    test_input = MCQInput(
        question="What is the capital of France?",
        choices=["London", "Paris", "Berlin", "Madrid"],
        gold_idxs=1,
        instruction="Please answer the following question about geography.",
    )

    # Test mcq_prompt_functions directly
    from lighteval.tasks.templates.formulation import MCFFormulation
    from lighteval.tasks.templates.multichoice_config import mcq_prompt_functions

    # Generate prompt using mcq_prompt_functions
    doc = mcq_prompt_functions(test_input, "test_task", Language.english, MCFFormulation())

    assert (
        doc.query
        == """\
Please answer the following question about geography.
Question: What is the capital of France?
 A. London
 B. Paris
 C. Berlin
 D. Madrid
Answer:\
"""
    )

    assert doc.unconditioned_query == "Answer:"
    assert doc.choices == [" A", " B", " C", " D"]
