"""
abstract:
We introduce Social IQa: Social Interaction QA, a new question-answering
benchmark for testing social commonsense intelligence. Contrary to many prior
benchmarks that focus on physical or taxonomic knowledge, Social IQa focuses on
reasoning about people's actions and their social implications. For example,
given an action like "Jesse saw a concert" and a question like "Why did Jesse do
this?", humans can easily infer that Jesse wanted "to see their favorite
performer" or "to enjoy the music", and not "to see what's happening inside" or
"to see if it works". The actions in Social IQa span a wide variety of social
situations, and answer candidates contain both human-curated answers and
adversarially-filtered machine-generated candidates. Social IQa contains over
37,000 QA pairs for evaluating models' abilities to reason about the social
implications of everyday events and situations.

languages:
en

tags:
qa, social-intelligence, commonsense

paper:

"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


siqa = LightevalTaskConfig(
    name="siqa",
    suite=["lighteval"],
    prompt_function=prompt.siqa,
    hf_repo="allenai/social_i_qa",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)
