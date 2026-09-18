from lighteval.tasks.tasks.boolq import boolq_contrastset_prompt


def test_boolq_contrastset_gold_index_matches_answer():
    # boolq:contrastset derived its gold index from a reversed ["No", "Yes"]
    # lookup while its choices were ["Yes", "No"], so every sample was graded
    # against the opposite answer. The gold choice must equal the answer.
    for answer in ("Yes", "No"):
        line = {
            "answer": answer,
            "contrast_inputs": {
                "passage": ["A passage."],
                "question": ["A question?"],
            },
        }
        doc = boolq_contrastset_prompt(line)
        assert doc.choices[doc.gold_index] == answer
