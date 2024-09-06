from dataclasses import dataclass
from typing import Callable, Literal

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils import Metric
from lighteval.tasks.default_prompts import INTEGER_INDICES, LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.translation_literals import SUPPORTED_LANGUAGES, TRANSLATION_LITERALS


@dataclass
class MCQInput:
    question: str
    choices: list[str]
    gold_idxs: list[int] | int
    context: str | None = None


@dataclass
class DatasetConfig:
    hf_repo: str
    hf_subset: str
    trust_dataset: bool = False


ChoicePrefix = Literal["Letters", "NativeLetters", "Numbers"]


@dataclass
class CFFormulation:
    pass


@dataclass
class HybridFormulation:
    choice_prefix: ChoicePrefix = "Letters"


@dataclass
class MCFFormulation:
    choice_prefix: ChoicePrefix = "Letters"


Formulation = CFFormulation | HybridFormulation | MCFFormulation


def decapitalize(word: str):
    if len(word) == 0:
        return word
    return word[0].lower() + word[1:]


def capitalize(word: str):
    if len(word) == 0:
        return word
    return word[0].upper() + word[1:]


def fix_ending_punct(ctx: str, lang: SUPPORTED_LANGUAGES):
    ctx = ctx.strip()
    if len(ctx) == 0:
        return ctx
    if ctx.endswith("?"):
        ctx = ctx[:-1] + QUESTION_MARK[lang]
    elif ctx.endswith("."):
        ctx = ctx[:-1] + FULL_STOP[lang]
    elif ctx.endswith(","):
        ctx = ctx[:-1] + COMMA[lang]
    elif ctx.endswith(":"):
        ctx = ctx[:-1] + COLON[lang]
    return ctx


def is_ended_sentence(text: str, lang: LANGS):
    return text.strip().endswith(f"{QUESTION_MARK[lang]}{FULL_STOP[lang]}{COLON[lang]}")


def should_follow_sentence_space(prefix: str, lang: LANGS):
    return prefix.strip().endswith(f"{QUESTION_MARK[lang]}{FULL_STOP[lang]}{COLON[lang]}{COMMA[lang]}")


def fix_capitalization(prefix: str, text: str, lang: LANGS):
    # TODO: Prob cache this
    return capitalize(text) if is_ended_sentence(prefix, lang) else decapitalize(text)


MULTI_CHOICE_QA_TEMPLATE = "{context}{question_word}{colon}{sentence_space}{question}\n{options}{answer_word}{colon}"


def mcq_prompt_functions(
    mcf_input: MCQInput, task_name: str, language: SUPPORTED_LANGUAGES, formulation: Formulation
) -> Doc:
    """ """
    translation_literals = TRANSLATION_LITERALS[language]

    def get_prefix(choice_prefix: ChoicePrefix):
        if choice_prefix == "Letters":
            return LETTER_INDICES
        elif choice_prefix == "NativeLetters":
            return translation_literals.indices
        elif choice_prefix == "Numbers":
            return INTEGER_INDICES

    def build_options(answers: list[str]):
        if isinstance(formulation, CFFormulation):
            return None

        prefixes = get_prefix(formulation.choice_prefix)
        # Note the sentence space before each option, this ensures consistent tokenization with answers
        options = "\n".join(
            [f"{translation_literals.options}{translation_literals.colon}"]
            + [f"{translation_literals.sentence_space}{prefixes[i]}. {c}" for i, c in enumerate(answers)]
        )
        return f"{options}"

    context = f"{capitalize(fix_ending_punct(mcf_input.context, language))}\n" if mcf_input.context else ""
    question = fix_capitalization(context, fix_ending_punct(mcf_input.question, language), language)
    answers = [capitalize(fix_ending_punct(answer, language)) for answer in mcf_input.choices]

    options = build_options(answers)
    options = f"{options}\n" if options else ""

    query = MULTI_CHOICE_QA_TEMPLATE.format(
        question=question,
        context=context,
        question_word=translation_literals.question,
        answer_word=translation_literals.answer,
        colon=translation_literals.colon,
        sentence_space=translation_literals.sentence_space,
        options=options,
    )

    return Doc(
        task_name=task_name,
        query=query,
        gold_index=mcf_input.gold_idxs,
        choices=[f"{translation_literals.sentence_space}{c}" for c in answers if c],
        unconditioned_query=f"{translation_literals.answer}{translation_literals.colon}",
    )


class MCQTaskConfig(LightevalTaskConfig):
    def __init__(
        self,
        name: str,
        adapter: Callable[[dict], MCQInput],
        language: SUPPORTED_LANGUAGES,
        dataset_config: DatasetConfig,
        metrics: list[Metric | Metrics] | None = None,
        formulation: Formulation = MCFFormulation(),
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
            metric=metrics,
            prompt_function=prompt_function,
        )
