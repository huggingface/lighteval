# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


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
