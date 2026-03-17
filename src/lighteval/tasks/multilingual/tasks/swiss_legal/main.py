from dataclasses import dataclass
from typing import Callable, Literal, Optional

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.tasks.swiss_legal.metrics import (
    METRICS_TO_USE,
    device,
    get_bert_score,
    get_extractiveness,
    get_metrics,
    get_swiss_landmark_decision_summarization_judge,
)
from lighteval.tasks.requests import Doc, SamplingMethod


def create_translation_pairs(langs_list: list) -> list[tuple]:
    """
    Create all possible translation pairs from a given list of languages.

    Args:
    langs_list (list): A list of languages.

    Returns:
    lang_pairs_list (list): A list of tuples representing a translation pair.
    """
    lang_pairs_list = []
    for i, lang1 in enumerate(langs_list):
        for lang2 in langs_list[i + 1 :]:
            lang_pairs_list.append((lang1, lang2))
            lang_pairs_list.append((lang2, lang1))
    return lang_pairs_list


@dataclass
class LevelConfig:
    name: str
    text_col_name: str
    generation_size: int
    stop_sequence: list[str]
    metadata_cols: Optional[list[str]] = None
    custom_attributes: Optional[dict] = None
    dataset_filter: Optional[Callable[[dict], bool]] = None


@dataclass
class DatasetConfig:
    name: str
    hf_repo: str
    languages: list[str]
    task_type: Literal["translation", "summarization"]
    subsets: dict[str, LevelConfig]

    def __post_init__(self):
        self.translation_pairs = create_translation_pairs(self.languages)


# Translation of Swiss Federal Supreme Court Decision Summaries on three levels: the entire decision, the regeste level and the text level.
SwissDecisionSummaryTranslations = DatasetConfig(
    name="sdst",
    hf_repo="joelniklaus/SwissDecisionSummaryTranslations",
    languages=["de", "fr", "it"],
    task_type="translation",
    subsets={
        "bge_level": LevelConfig(
            name="bge_level",
            text_col_name="bgeText",
            metadata_cols=["bge"],
            generation_size=2048,
            stop_sequence=["</s>", ".\n\n", "\n\n"],
        ),
        "regeste_level": LevelConfig(
            name="regeste_level",
            text_col_name="regesteText",
            metadata_cols=["bge"],
            generation_size=512,
            stop_sequence=["</s>", ".\n\n", "\n\n"],
        ),
        "text_level": LevelConfig(
            name="text_level",
            text_col_name="text",
            metadata_cols=["bge"],
            generation_size=256,
            stop_sequence=["</s>", ".\n", "\n"],
        ),
    },
)

# Translation of Swiss Federal Laws on three levels: the entire law, the article level and the paragraph level.
SwissLawTranslations = DatasetConfig(
    name="slt",
    hf_repo="joelniklaus/SwissLawTranslations",
    languages=["de", "fr", "it", "rm", "en"],
    task_type="translation",
    subsets={
        "law_level": LevelConfig(
            name="law_level",
            text_col_name="lawText",
            metadata_cols=["rsNr"],
            generation_size=16384,
            stop_sequence=["</s>", ".\n\n", "\n\n"],
        ),
        "article_level": LevelConfig(
            name="article_level",
            text_col_name="artText",
            metadata_cols=["rsNr"],
            generation_size=1024,
            stop_sequence=["</s>", ".\n\n", "\n\n"],
        ),
        "paragraph_level": LevelConfig(
            name="paragraph_level",
            text_col_name="parText",
            metadata_cols=["rsNr"],
            generation_size=256,
            stop_sequence=["</s>", ".\n", "\n"],
        ),
    },
)

# Translation of Swiss Federal Supreme Court Press Releases on one level: the entire press release.
SwissSupremeCourtPressReleaseTranslations = DatasetConfig(
    name="sscprt",
    hf_repo="joelniklaus/SwissSupremeCourtPressReleaseTranslations",
    languages=["de", "fr", "it"],
    task_type="translation",
    subsets={
        "press_release": LevelConfig(
            name="press_release",
            text_col_name="text",
            metadata_cols=["filename"],
            generation_size=1024,
            stop_sequence=["</s>"],
        )
    },
)

# Headnote generation (summarization) for Swiss Landmark Decisions on one level: the entire landmark decision.
slds_languages = ["de", "fr", "it"]


def get_slds_filter_fn(decision_language: str, headnote_language: str):
    def filter_dataset(example):
        return example["decision_language"] == decision_language and example["headnote_language"] == headnote_language

    return filter_dataset


SwissLandmarkDecisionHeadnotes = DatasetConfig(
    name="slds",
    hf_repo="ipst/slds",
    languages=slds_languages,
    task_type="summarization",
    subsets={
        **{
            f"{decision_lang}_{headnote_lang}": LevelConfig(
                name=f"{decision_lang}_{headnote_lang}",
                custom_attributes={
                    "decision_language": decision_lang,
                    "headnote_language": headnote_lang,
                },
                text_col_name="decision",
                generation_size=512,
                dataset_filter=get_slds_filter_fn(decision_lang, headnote_lang),
                stop_sequence=["</s>"],
            )
            for decision_lang in slds_languages
            for headnote_lang in slds_languages
        }
    },
)


def create_translation_prompt_fn(level_config: LevelConfig, source_lang: str, target_lang: str):
    """
    Create a prompt function for a given level configuration.
    """
    text_col = level_config.text_col_name
    src_text_col = f"{source_lang}_{text_col}"
    target_text_col = f"{target_lang}_{text_col}"

    def prompt_fn(line: dict, task_name: str = None):
        # Following Template A from https://github.com/huggingface/lighteval/pull/389#issuecomment-2471580177
        custom_query = f"{source_lang.upper()}: {line[src_text_col]}\n{target_lang.upper()}: "

        return Doc(
            task_name=task_name,
            query=custom_query,
            choices=[str(line[target_text_col])],
            gold_index=0,
            specific={
                **{col: line[col] for col in level_config.metadata_cols},
                "question": custom_query,
                "source": line[src_text_col],
                "source_lang": source_lang,
                "target_lang": target_lang,
            },
        )

    return prompt_fn


def iso2lang(iso_code: str) -> str:
    """
    Convert an ISO 639-1 code to a language name.
    """
    assert iso_code in ["de", "fr", "it"], f"Invalid ISO code for SLDS dataset: {iso_code}"
    if iso_code == "de":
        return "German"
    if iso_code == "fr":
        return "French"
    if iso_code == "it":
        return "Italian"
    return None


def slds_prompt_fn(line: dict, task_name: str = None):
    """
    Create a prompt for the Swiss Legal Decision Summaries dataset.
    """
    template = (
        "Leading decision:\n```{decision}```\n\nGenerate a headnote in {language} for the leading decision above."
    )

    return Doc(
        task_name=task_name,
        query=template.format(language=iso2lang(line["headnote_language"]), decision=line["decision"]),
        choices=[str(line["headnote"])],
        gold_index=0,
        specific={
            "sample_id": line["sample_id"],
            "decision_id": line["decision_id"],
            "decision_language": line["decision_language"],
            "headnote_language": line["headnote_language"],
            "law_area": line["law_area"],
            "year": line["year"],
            "text": line["decision"],  # Needs to be called "text" for the extractiveness metric
            "headnote": line["headnote"],
        },
    )


class TranslationTask(LightevalTaskConfig):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        level_name: str,
        source_lang: str,
        target_lang: str,
    ):
        level_config = dataset_config.subsets[level_name]
        super().__init__(
            name=f"{dataset_config.name}-{level_name}:{source_lang}-{target_lang}",
            suite=["community"],
            prompt_function=create_translation_prompt_fn(level_config, source_lang, target_lang),
            hf_repo=dataset_config.hf_repo,
            hf_subset=level_name,
            hf_filter=None,
            hf_avail_splits=["train", "validation", "test"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            generation_size=level_config.generation_size,
            metrics=get_metrics(METRICS_TO_USE, target_lang, level_config.generation_size),
            stop_sequence=level_config.stop_sequence,
            # Remove the target language in the beginning if it exists: e.g., FR: {translation}
            # Is only applied to the generative metrics, but also there seems not to be invoked, maybe not passed through?
            # output_regex=f"(?:{target_lang.upper()}:\s*?)?(.*)",
        )


class HeadnoteGenerationTask(LightevalTaskConfig):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        level_name: str,
    ):
        level_config = dataset_config.subsets[level_name]
        headnote_language = dataset_config.subsets[level_name].custom_attributes["headnote_language"]

        super().__init__(
            name=f"{dataset_config.name}:{level_name}",
            suite=["community"],
            prompt_function=slds_prompt_fn,
            hf_repo="ipst/slds",
            hf_subset=level_name,
            hf_filter=level_config.dataset_filter,
            hf_avail_splits=["train", "validation", "test", "one_shot_examples"],
            evaluation_splits=["test"],
            few_shots_split="one_shot_examples",
            few_shots_select="random",
            generation_size=level_config.generation_size,
            metrics=self._get_metrics(headnote_language),
            stop_sequence=level_config.stop_sequence,
        )

    def _get_metrics(self, headnote_language: Literal["de", "fr", "it"]) -> list[Metrics]:
        return [
            get_bert_score(
                language=headnote_language,
                model_type="xlm-roberta-large",
                device=device,
                metric_category=SamplingMethod.GENERATIVE,
            ),
            Metrics.bleu,
            Metrics.rouge1,
            Metrics.rouge2,
            Metrics.rougeL,
            get_swiss_landmark_decision_summarization_judge(
                language=headnote_language,
            ),
            get_extractiveness(language=headnote_language),
        ]


DATASETS = [
    SwissDecisionSummaryTranslations,
    SwissLawTranslations,
    SwissSupremeCourtPressReleaseTranslations,
    SwissLandmarkDecisionHeadnotes,
]

TASKS_TABLE = [
    *[
        TranslationTask(
            dataset_config=dataset,
            level_name=subset,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        for dataset in DATASETS
        for subset in dataset.subsets
        for source_lang, target_lang in dataset.translation_pairs
        if dataset.task_type == "translation"
    ],
    *[
        HeadnoteGenerationTask(
            dataset_config=SwissLandmarkDecisionHeadnotes,
            level_name=subset,
        )
        for subset in SwissLandmarkDecisionHeadnotes.subsets
    ],
]
