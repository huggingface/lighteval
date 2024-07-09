from lighteval.utils import as_list


import json
from dataclasses import asdict, dataclass
from typing import Optional, Union


@dataclass(slots=True)
class Doc:
    """
    Dataclass used to represent the content of a task example
    almost every field is optional, but some tasks require some fields to be present.
    When adding a new task, please add the required fields to the doc class.
    Each task will have a different set of fields needed.
    """

    query: str
    choices: list[str]
    gold_index: Union[int, list[int]]
    original_query: Optional[str] = ""  # the query before preprocessing, if stored
    specific: dict = None  # Information which is specific to the current eval
    uncoditioned_prefix: str | None = (
        None  # Prefix to use during pmi normalization for each chioce, if None PMI is not supported
    )
    task_name: str = ""

    # For few-shot
    instruction: Optional[list[str]] = None
    target_for_fewshot_sorting: Optional[str] = None  # will probably have to be removed in the future

    # Filled when parsing and adding the few-shot context
    ctx: Optional[str] = ""
    num_asked_few_shots: int = -1
    num_effective_few_shots: int = -1

    def get_golds(self, few_shot: bool = False):
        """Return gold targets extracted from the target dict"""
        gold_indices = as_list(self.gold_index)
        if few_shot and self.target_for_fewshot_sorting is not None:
            choices = self.target_for_fewshot_sorting
            if isinstance(choices, str):  # correct choice is already selected
                return choices
        else:
            choices = self.choices
        golds = []
        for gold_ix in gold_indices:
            local_golds = as_list(choices[gold_ix])
            for local_gold in local_golds:
                golds.append(local_gold)
        return golds

    def __repr__(self):
        doc_dict = asdict(self)
        return json.dumps(doc_dict)