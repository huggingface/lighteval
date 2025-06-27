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


import numpy as np

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


class GlobalMMLUPrompt:
    def __init__(self, lang):
        self.lang = lang
        self.lang_to_template = {
            "eng": "Given the following query and answer choices, output the letter corresponding to the correct answer. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of A, B, C, or D. Think step by step before answering.\n\n###\nQuery:\n{Question}\n###\nChoices:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
            "deu": "Gib basierend auf der folgenden Frage und den Antwortmöglichkeiten den Buchstaben aus, der der richtigen Antwort entspricht. Die letzte Zeile deiner Antwort sollte folgendes Format haben: 'Antwort: $BUCHSTABE' (ohne Anführungszeichen), wobei BUCHSTABE einer der folgenden ist: A, B, C oder D. Denke Schritt für Schritt, bevor du antwortest.\n\n###\nFrage:\n{Question}\n###\nAntwortmöglichkeiten:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
            "fra": "A partir de la question et des choix de réponses suivants, indiquez la lettre correspondant à la bonne réponse. La dernière ligne de votre réponse doit avoir le format suivant : 'Réponse: '$LETTRE' (sans les guillemets) où LETTRE est l'une des lettres: A, B, C ou D. Réfléchissez étape par étape avant de répondre.\n\n###\nRequête:\n{Question}\n###\nChoix:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
            "ita": "Dato il seguente quesito e le diverse opzioni per una risposta, indicare la lettera corrispondente alla risposta corretta. L'ultima riga della risposta deve avere il seguente formato: 'Risposta: $LETTERA' (senza virgolette), e LETTERA è necessariamente una tra A, B, C, D. Prima di rispondere, è importante che si ragioni passo per passo.\n\n###\nQuesito:\n{Question}\n###\nOpzioni:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
            "por": "Tendo em conta a seguinte pergunta e opções de resposta, indique a letra correspondente à resposta correta. A última linha da sua resposta deve ter o seguinte formato: 'Resposta: $LETRA' (sem aspas) em que LETRA é uma de A, B, C ou D. Pense passo a passo antes de responder.\n\n###\nPergunta:\n{Question}\n###\nOpções:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
            "spa": "Dado el siguiente pregunta y opciones para la respuesta, escriba la letra correspondiente a la respuesta correcta. La última línea de su respuesta debe seguir el siguiente formato: 'Respuesta: $LETTER' (sin comillas) donde LETTER es A, B, C o D. Piense paso a paso antes de responder.\n\n###\nPregunta:\n{Question}\n###\nOpciones:\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
        }

    def prompt(self, line, task_name: str = None):
        gold_index = LETTER_INDICES.index(line["answer"])
        choices = [line["option_a"], line["option_b"], line["option_c"], line["option_d"]]
        lang = self.lang if self.lang in self.lang_to_template.keys() else "eng"
        query_template = self.lang_to_template[lang]
        query = query_template.format(
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
            Question=line["question"],
        )
        instruction = query_template.split("###\n")[0]

        return Doc(
            task_name=task_name,
            query=query,
            choices=LETTER_INDICES[: len(choices)],
            gold_index=gold_index,
            instruction=instruction,
        )


GLOBAL_MMLU_TASKS = [
    LightevalTaskConfig(
        name=f"global_mmlu_instruct_{language.value}",
        prompt_function=GlobalMMLUPrompt(language.value).prompt,
        suite=["extended"],
        hf_repo="CohereForAI/Global-MMLU",
        hf_subset=lang,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=[
            SampleLevelMetric(
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
            )
        ],
        generation_size=32768,  # needed for reasoning models like R1
        stop_sequence=[],  # no stop sequence, will use eos token
    )
    for lang, language in [
        ("am", Language.AMHARIC),
        ("ar", Language.ARABIC),
        ("bn", Language.BENGALI),
        ("cs", Language.CZECH),
        ("de", Language.GERMAN),
        ("el", Language.GREEK),
        ("en", Language.ENGLISH),
        ("es", Language.SPANISH),
        ("fa", Language.PERSIAN),
        # ("fil", Language.FILIPINO),
        ("fr", Language.FRENCH),
        ("ha", Language.HAUSA),
        ("he", Language.HEBREW),
        ("hi", Language.HINDI),
        ("id", Language.INDONESIAN),
        ("ig", Language.IGBO),
        ("it", Language.ITALIAN),
        ("ja", Language.JAPANESE),
        ("ko", Language.KOREAN),
        ("ky", Language.KYRGYZ),
        ("lt", Language.LITHUANIAN),
        ("mg", Language.MALAGASY),
        ("ms", Language.MALAY),
        ("ne", Language.NEPALI),
        ("nl", Language.DUTCH),
        ("ny", Language.NORWEGIAN),
        ("pl", Language.POLISH),
        ("pt", Language.PORTUGUESE),
        ("ro", Language.ROMANIAN),
        ("ru", Language.RUSSIAN),
        ("si", Language.SINHALA),
        ("sn", Language.SHONA),
        ("so", Language.SOMALI),
        ("sr", Language.SERBIAN),
        ("sv", Language.SWEDISH),
        ("sw", Language.SWAHILI),
        ("te", Language.TELUGU),
        ("tr", Language.TURKISH),
        ("uk", Language.UKRAINIAN),
        ("vi", Language.VIETNAMESE),
        ("yo", Language.YORUBA),
        ("zh", Language.CHINESE),
    ]
]
TASKS_TABLE.extend(GLOBAL_MMLU_TASKS)

GLOBAL_MMLU_LITE_TASKS = [
    LightevalTaskConfig(
        name=f"global_mmlu_lite_instruct_{language.value}",
        prompt_function=GlobalMMLUPrompt(language.value).prompt,
        suite=["extended"],
        hf_repo="CohereForAI/Global-MMLU-Lite",
        hf_subset=lang,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=[
            SampleLevelMetric(
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
            )
        ],
        generation_size=32768,  # needed for reasoning models like R1
        stop_sequence=[],  # no stop sequence, will use eos token
    )
    for lang, language in [
        ("am", Language.AMHARIC),
        ("ar", Language.ARABIC),
        ("bn", Language.BENGALI),
        ("cs", Language.CZECH),
        ("de", Language.GERMAN),
        ("el", Language.GREEK),
        ("en", Language.ENGLISH),
        ("es", Language.SPANISH),
        ("fa", Language.PERSIAN),
        # ("fil", Language.FILIPINO),
        ("fr", Language.FRENCH),
        ("ha", Language.HAUSA),
        ("he", Language.HEBREW),
        ("hi", Language.HINDI),
        ("id", Language.INDONESIAN),
        ("ig", Language.IGBO),
        ("it", Language.ITALIAN),
        ("ja", Language.JAPANESE),
        ("ko", Language.KOREAN),
        ("ky", Language.KYRGYZ),
        ("lt", Language.LITHUANIAN),
        ("mg", Language.MALAGASY),
        ("ms", Language.MALAY),
        ("ne", Language.NEPALI),
        ("nl", Language.DUTCH),
        ("ny", Language.NORWEGIAN),
        ("pl", Language.POLISH),
        ("pt", Language.PORTUGUESE),
        ("ro", Language.ROMANIAN),
        ("ru", Language.RUSSIAN),
        ("si", Language.SINHALA),
        ("sn", Language.SHONA),
        ("so", Language.SOMALI),
        ("sr", Language.SERBIAN),
        ("sv", Language.SWEDISH),
        ("sw", Language.SWAHILI),
        ("te", Language.TELUGU),
        ("tr", Language.TURKISH),
        ("uk", Language.UKRAINIAN),
        ("vi", Language.VIETNAMESE),
        ("yo", Language.YORUBA),
        ("zh", Language.CHINESE),
    ]
]
TASKS_TABLE.extend(GLOBAL_MMLU_LITE_TASKS)
