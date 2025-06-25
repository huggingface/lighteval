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

from functools import partial

import numpy as np
from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    IndicesExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.metrics.metrics import MetricCategory, MetricUseCase, SampleLevelMetric
from lighteval.metrics.metrics_sample import (
    PassAtK,
)
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


TASKS_TABLE = []

lang_to_literal = {
    "deu": Language.GERMAN,
    "fra": Language.FRENCH,
    "ita": Language.ITALIAN,
    "por": Language.PORTUGUESE,
    "spa": Language.SPANISH,
}


def belebele_prompt(line, task_name: str = None):
    lang_to_template = {
        "eng_Latn": "Given the following passage, query, and answer choices, output the letter corresponding to the correct answer. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of A, B, C, or D. Think step by step before answering.\n\n###\nPassage:\n{Passage}\n###\nQuery:\n{Question}\n###\nChoices:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
        "deu_Latn": "Gib basierend auf dem folgenden Textabschnitt, der Frage und den Antwortmöglichkeiten den Buchstaben aus, der der richtigen Antwort entspricht. Die letzte Zeile deiner Antwort sollte folgendes Format haben: 'Antwort: $BUCHSTABE' (ohne Anführungszeichen), wobei BUCHSTABE einer der folgenden ist: A, B, C oder D. Denke Schritt für Schritt, bevor du antwortest.\n\n###\nTextabschnitt:\n{Passage}\n###\nFrage:\n{Question}\n###\nAntwortmöglichkeiten:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
        "fra_Latn": "A partir du passage suivant, de la question et des choix de réponses, indiquez la lettre correspondant à la bonne réponse. La dernière ligne de votre réponse doit avoir le format suivant : 'Réponse: '$LETTRE' (sans les guillemets) où LETTRE est l'une des lettres: A, B, C ou D. Réfléchissez étape par étape avant de répondre.\n\n###\nPassage:\n{Passage}\n###\nRequête:\n{Question}\n###\nChoix:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
        "ita_Latn": "Dato il seguente passaggio, un quesito e le diverse opzioni per una risposta, indicare la lettera corrispondente alla risposta corretta. L'ultima riga della risposta deve avere il seguente formato: 'Risposta: $LETTERA' (senza virgolette), e LETTERA è necessariamente una tra A, B, C, D. Prima di rispondere, è importante che si ragioni passo per passo.\n\n###\nPassaggio:\n{Passage}\n###\nQuesito:\n{Question}\n###\nOpzioni:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
        "por_Latn": "Tendo em conta a seguinte passagem, pergunta e opções de resposta, indique a letra correspondente à resposta correta. A última linha da sua resposta deve ter o seguinte formato: 'Resposta: $LETRA' (sem aspas) em que LETRA é uma de A, B, C ou D. Pense passo a passo antes de responder.\n\n###\nPassagem:\n{Passage}\n###\nPergunta:\n{Question}\n###\nOpções:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
        "spa_Latn": "Dado el siguiente contexto, pregunta y opciones para la respuesta, escriba la letra correspondiente a la respuesta correcta. La última línea de su respuesta debe seguir el siguiente formato: 'Respuesta: $LETTER' (sin comillas) donde LETTER es A, B, C o D. Piense paso a paso antes de responder.\n\n###\nContexto:\n{Passage}\n###\nPregunta:\n{Question}\n###\nOpciones:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
    }

    gold_index = int(line["correct_answer_num"]) - 1
    choices = [line["mc_answer1"], line["mc_answer2"], line["mc_answer3"], line["mc_answer4"]]
    query_template = lang_to_template.get(line["dialect"], "eng_Latn")
    query = query_template.format(
        A=choices[0],
        B=choices[1],
        C=choices[2],
        D=choices[3],
        Passage=line["flores_passage"],
        Question=line["question"],
    )
    instruction = query_template.split("\n\n###")[0]

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(choices)],
        gold_index=gold_index,
        instruction=instruction,
    )


BELEBELE_TASKS = [
    LightevalTaskConfig(
        name=f"belebele_instruct_{lang}_Latn",
        prompt_function=belebele_prompt,
        suite=["extended"],
        hf_repo="facebook/belebele",
        hf_subset=f"{lang}_Latn",
        evaluation_splits=["test"],
        hf_avail_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=32768,  # needed for reasoning models like R1
        metric=[
            SampleLevelMetric(
                metric_name="pass@1:1_samples",
                sample_level_fn=PassAtK(
                    k=1,
                    n=1,
                    sample_scoring_function=lambda pred, ref, doc: multilingual_extractive_match_metric(
                        language=lang_to_literal[lang],
                        gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
                        pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
                        precision=6,
                    ).sample_level_fn([ref], [pred], doc),
                ).compute,
                category=MetricCategory.GENERATIVE_SAMPLING,
                use_case=MetricUseCase.REASONING,
                corpus_level_fn=np.mean,
                higher_is_better=True,
            )
        ],
        stop_sequence=[],  # no stop sequence, will use eos token
        trust_dataset=True,
        version=1,
    )
    for lang in [
        "deu",
        "fra",
        "ita",
        "por",
        "spa",
    ]
]

TASKS_TABLE.extend(BELEBELE_TASKS)


MMLU_SUBSETS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


class GlobalMMLUPrompt:
    def __init__(self, lang):
        self.lang = lang
        self.lang_to_template = {
            "eng": "Given the following query and answer choices, output the letter corresponding to the correct answer. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of A, B, C, or D. Think step by step before answering.\n\n###\nQuery:\n{Question}\n###\nChoices:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
            "deu": "Gib basierend auf der folgenden Frage und den Antwortmöglichkeiten den Buchstaben aus, der der richtigen Antwort entspricht. Die letzte Zeile deiner Antwort sollte folgendes Format haben: 'Antwort: $BUCHSTABE' (ohne Anführungszeichen), wobei BUCHSTABE einer der folgenden ist: A, B, C oder D. Denke Schritt für Schritt, bevor du antwortest.\n\n###\nFrage:\n{Question}\n###\nAntwortmöglichkeiten:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
            "fra": "A partir de la question et des choix de réponses suivants, indiquez la lettre correspondant à la bonne réponse. La dernière ligne de votre réponse doit avoir le format suivant : 'Réponse: '$LETTRE' (sans les guillemets) où LETTRE est l'une des lettres: A, B, C ou D. Réfléchissez étape par étape avant de répondre.\n\n###\nRequête:\n{Question}\n###\nChoix:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
            "ita": "Dato il seguente quesito e le diverse opzioni per una risposta, indicare la lettera corrispondente alla risposta corretta. L'ultima riga della risposta deve avere il seguente formato: 'Risposta: $LETTERA' (senza virgolette), e LETTERA è necessariamente una tra A, B, C, D. Prima di rispondere, è importante che si ragioni passo per passo.\n\n###\nQuesito:\n{Question}\n###\nOpzioni:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
            "por": "Tendo em conta a seguinte pergunta e opções de resposta, indique a letra correspondente à resposta correta. A última linha da sua resposta deve ter o seguinte formato: 'Resposta: $LETRA' (sem aspas) em que LETRA é uma de A, B, C ou D. Pense passo a passo antes de responder.\n\n###\nPergunta:\n{Question}\n###\nOpções:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
            "spa": "Dado el siguiente pregunta y opciones para la respuesta, escriba la letra correspondiente a la respuesta correcta. La última línea de su respuesta debe seguir el siguiente formato: 'Respuesta: $LETTER' (sin comillas) donde LETTER es A, B, C o D. Piense paso a paso antes de responder.\n\\###\nPregunta:\n{Question}\n###\nOpciones:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
        }

    def prompt(self, line, task_name: str = None):
        gold_index = LETTER_INDICES.index(line["answer"])
        choices = [line["option_a"], line["option_b"], line["option_c"], line["option_d"]]
        query_template = self.lang_to_template.get(self.lang, "eng")
        query = query_template.format(
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
            Question=line["question"],
        )
        instruction = query_template.split("\n\n###")[0]

        return Doc(
            task_name=task_name,
            query=query,
            choices=LETTER_INDICES[: len(choices)],
            gold_index=gold_index,
            instruction=instruction,
        )


global_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"global_mmlu_instruct_{sensitivity_label.lower()}_{language.value}:{subset}",
        prompt_function=GlobalMMLUPrompt(language).prompt,
        suite=("extended"),
        hf_repo="CohereForAI/Global-MMLU",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="dev",
        hf_filter=partial(
            lambda subset, sensitivity_label, x: x["subject"].lower() == subset
            and (
                sensitivity_label == "ALL" or sensitivity_label in x["cultural_sensitivity_label"].replace("-", "UNK")
            )
            and all(x[f"option_{opt}"] is not None and x[f"option_{opt}"].strip() for opt in "abcd"),
            subset,
            sensitivity_label,
        ),
        metric=SampleLevelMetric(
            metric_name="pass@1:1_samples",
            sample_level_fn=PassAtK(
                k=1,
                n=1,
                sample_scoring_function=lambda pred, ref, doc: multilingual_extractive_match_metric(
                    language=language,
                    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
                    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
                    precision=6,
                ).sample_level_fn([ref], [pred], doc),
            ).compute,
            category=MetricCategory.GENERATIVE_SAMPLING,
            use_case=MetricUseCase.REASONING,
            corpus_level_fn=np.mean,
            higher_is_better=True,
        ),
        generation_size=32768,  # needed for reasoning models like R1
        stop_sequence=[],  # no stop sequence, will use eos token
    )
    for subset in MMLU_SUBSETS
    for language in [
        Language.GERMAN,
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.HEBREW,
        Language.HINDI,
        Language.INDONESIAN,
        Language.ITALIAN,
        Language.JAPANESE,
        Language.KOREAN,
        Language.MALAY,
        Language.DUTCH,
        Language.NORWEGIAN,
        Language.POLISH,
        Language.PORTUGUESE,
        Language.ROMANIAN,
        Language.RUSSIAN,
        Language.SERBIAN,
        Language.SWEDISH,
        Language.SWAHILI,
        Language.TAMIL,
        Language.TELUGU,
        Language.THAI,
        Language.TURKISH,
        Language.UKRAINIAN,
        Language.URDU,
        Language.VIETNAMESE,
        Language.YORUBA,
        Language.ZULU,
    ]
    for sensitivity_label in ["ALL", "CA", "CS", "UNK"]
]


def mmlu_pro(line, task_name: str = None):
    instruction = f"Given the following question about {line['category']} and answer choices, output the letter corresponding to the correct answer. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {' ,'.join(LETTER_INDICES[: len(line['choices'] - 1)])}, or {LETTER_INDICES[len(line['choices'])]}. Think step by step before answering.\n\n"
    query = f"{instruction}###\nQuery:\n{line['question']}\n###\nChoices:\n"
    query += "".join([f"\n{key}) {choice}" for key, choice in zip(LETTER_INDICES, line["choices"])])

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(line["choices"])],
        gold_index=line["answer_index"],
        instruction=instruction,
    )


mmlu_pro = LightevalTaskConfig(
    name="mmlu_pro",
    suite=["lighteval"],
    prompt_function=mmlu_pro,
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_avail_splits=["validation", "test"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    stop_sequence=[],  # no stop sequence, will use eos token
    metric=SampleLevelMetric(
        metric_name="pass@1:1_samples",
        sample_level_fn=PassAtK(
            k=1,
            n=1,
            sample_scoring_function=lambda pred, ref, doc: multilingual_extractive_match_metric(
                language=Language.ENGLISH,
                gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
                pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
                precision=6,
            ).sample_level_fn([ref], [pred], doc),
        ).compute,
        category=MetricCategory.GENERATIVE_SAMPLING,
        use_case=MetricUseCase.REASONING,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    ),
    trust_dataset=True,
    version=0,
)
