from typing import Literal

from ..utils.metrics import get_qa_metric
from ..utils.prompts import get_kenswquad_prompt, get_mlqa_prompt
from lighteval.tasks.lighteval_task import LightevalTaskConfig
EVAL_TYPE = Literal["exact", "f1"]

class ThaiQATask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"thaiqa",
            prompt_function=get_mlqa_prompt("th", answer_key="answer"),
            suite=("custom",),
            hf_repo="HuggingFaceFW-Dev/thaiqa_squad_fixed",
            hf_subset="default",
            evaluation_splits=("train",),
            few_shots_split="validation",
            generation_size=40,
            stop_sequence=("\n",),
            metric=(get_qa_metric("th", "exact"), get_qa_metric("th", "f1")),
        )
        

class SberSquadTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"sber_squad",
            prompt_function=get_mlqa_prompt("ru"),
            suite=("custom",),
            hf_repo="kuznetsoffandrey/sberquad",
            hf_subset="sberquad",
            evaluation_splits=("validation",),
            few_shots_split="train",
            hf_revision="92d74b272206a76fb3fec1f0355acab370a4de3a",
            trust_dataset=True,
            metric=(get_qa_metric("ru", "exact"), get_qa_metric("ru", "f1")),
            generation_size=50,
            stop_sequence=("\n",),
        )
        
class ARCDSquadTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"arcd",
            prompt_function=get_mlqa_prompt("ar"),
            suite=("custom",),
            hf_repo="hsseinmz/arcd",
            hf_subset="plain_text",
            # test is less than 1k
            evaluation_splits=("train", "validation"),
            trust_dataset=True,
            metric=(get_qa_metric("ar", "exact"), get_qa_metric("ar", "f1")),
            generation_size=75,
            stop_sequence=("\n",),
        )
        
class KenswQuADTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"kenswquad",
            prompt_function=get_kenswquad_prompt("sw"),
            suite=("custom",),
            hf_repo="HuggingFaceFW-Dev/KenSwQuAD",
            hf_subset="default",
            evaluation_splits=("test",),
            few_shots_split="validation",
            metric=(get_qa_metric("sw", "exact"), get_qa_metric("sw", "f1")),
            generation_size=40,
            stop_sequence=("\n",),
        )