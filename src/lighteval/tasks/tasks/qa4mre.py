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

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


"""
abstract:
QA4MRE is a machine reading comprehension benchmark from the CLEF 2011-2013
challenges. It evaluates systems' ability to answer questions requiring deep
understanding of short texts, supported by external background knowledge.
Covering tasks like modality, negation, biomedical reading, and entrance exams,
QA4MRE tests reasoning beyond surface-level text matching.

languages:
en

tags:
reading-comprehension, qa, health, biomedical

paper:
https://link.springer.com/chapter/10.1007/978-3-642-40802-1_29
"""


qa4mre_2011 = LightevalTaskConfig(
    name="qa4mre:2011",
    suite=["lighteval"],
    prompt_function=prompt.qa4mre,
    hf_repo="qa4mre",
    hf_subset="2011.main.EN",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)


qa4mre_2012 = LightevalTaskConfig(
    name="qa4mre:2012",
    suite=["lighteval"],
    prompt_function=prompt.qa4mre,
    hf_repo="qa4mre",
    hf_subset="2012.main.EN",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)


qa4mre_2013 = LightevalTaskConfig(
    name="qa4mre:2013",
    suite=["lighteval"],
    prompt_function=prompt.qa4mre,
    hf_repo="qa4mre",
    hf_subset="2013.main.EN",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)
