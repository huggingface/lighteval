from dataclasses import dataclass
from typing import Callable

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils import Metric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.formatting_utils import capitalize, fix_ending_punct
from lighteval.tasks.templates.formulation import Formulation, MCFFormulation, build_answers, build_options
from lighteval.tasks.templates.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language


@dataclass
class MCQInput:
    question: str
    choices: list[str]
    gold_idxs: list[int] | int
    context: str | None = None
    instruction: str | None = None


@dataclass
class DatasetConfig:
    hf_repo: str
    hf_subset: str
    trust_dataset: bool = False


@dataclass
class SplitConfig:
    evaluation_split: tuple[str]
    few_shots_split: str | None = None
    few_shots_select: str | None = None


MULTI_CHOICE_QA_TEMPLATE = (
    "{instruction}{context}{question_word}{colon}{sentence_space}{question}\n{options}{answer_word}{colon}"
)


def mcq_prompt_functions(mcf_input: MCQInput, task_name: str, language: Language, formulation: Formulation) -> Doc:
    translation_literals = TRANSLATION_LITERALS[language]

    instruction = f"{mcf_input.instruction}\n" if mcf_input.instruction else ""
    context = f"{capitalize(fix_ending_punct(mcf_input.context, translation_literals))}\n" if mcf_input.context else ""
    question = capitalize(fix_ending_punct(mcf_input.question, translation_literals))
    answers = [capitalize(fix_ending_punct(answer, translation_literals)) for answer in mcf_input.choices]

    options = build_options(answers, formulation, translation_literals)
    options = f"{options}\n" if options else ""
    answers = build_answers(answers, formulation, translation_literals)

    query = MULTI_CHOICE_QA_TEMPLATE.format(
        instruction=instruction,
        question=question,
        context=context,
        question_word=capitalize(translation_literals.question),
        answer_word=capitalize(translation_literals.answer),
        colon=translation_literals.colon,
        sentence_space=translation_literals.sentence_space,
        options=options,
    )

    return Doc(
        task_name=task_name,
        query=query,
        gold_index=mcf_input.gold_idxs,
        choices=answers,
        instruction=mcf_input.instruction,
        unconditioned_query=f"{capitalize(translation_literals.answer)}{translation_literals.colon}",
    )


class MultiChoiceTaskConfig(LightevalTaskConfig):
    def __init__(
        self,
        name: str,
        adapter: Callable[[dict], MCQInput],
        language: Language,
        dataset_config: DatasetConfig,
        split_config: SplitConfig,
        metrics: list[Metric | Metrics],
        formulation: Formulation = MCFFormulation(),
        suite: tuple[str] = ("multilingual",),
        version: int = 0,
    ):
        """
        # TODO: More description here
        Recommended metrics:
        - probability_metric
        - noormalized_multi_choice_prob_metric
        - loglikehood_acc_metric
        - exact_match
        """

        def prompt_function(line: dict, task_name: str):
            return mcq_prompt_functions(adapter(line), task_name, language, formulation)

        super().__init__(
            name=name,
            hf_repo=dataset_config.hf_repo,
            hf_subset=dataset_config.hf_subset,
            trust_dataset=dataset_config.trust_dataset,
            evaluation_splits=split_config.evaluation_split,
            few_shots_select=split_config.few_shots_select,
            few_shots_split=split_config.few_shots_split,
            metric=metrics,
            prompt_function=prompt_function,
            suite=suite,
            version=version,
        )
