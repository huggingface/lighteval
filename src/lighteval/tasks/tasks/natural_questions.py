"""
abstract:
This dataset is a collection of question-answer pairs from the Natural Questions
dataset. See Natural Questions for additional information. This dataset can be
used directly with Sentence Transformers to train embedding models.

languages:
en

tags:
general-knowledge, qa

paper:
https://ai.google.com/research/NaturalQuestions
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


natural_questions = LightevalTaskConfig(
    name="natural_questions",
    prompt_function=get_qa_prompt_function(
        Language.ENGLISH,
        lambda line: {"question": line["question"], "choices": [line["answer"]]},
    ),
    suite=("lighteval",),
    hf_repo="lighteval/small_natural_questions",
    hf_subset="default",
    evaluation_splits=("test",),
    few_shots_split="few_shot",
    generation_size=250,
    stop_sequence=["\n", "Question:", "question:"],
    metrics=[Metrics.exact_match],
    version=1,
)
