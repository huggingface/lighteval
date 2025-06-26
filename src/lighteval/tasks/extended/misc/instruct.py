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
from lighteval.tasks.multilingual.tasks import LangCodeLanguage, iso_639_3_ind_to_iso_639_3_macro
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


def belebele_prompt_en_instruct(line, task_name: str = None):
    line["dialect"] = "eng_Latn"
    return belebele_prompt(line, task_name)


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
    query_template = lang_to_template[line["dialect"]]
    query = query_template.format(
        A=choices[0],
        B=choices[1],
        C=choices[2],
        D=choices[3],
        Passage=line["flores_passage"],
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


BELEBELE_TASKS_NATIVE_INSTRUCT = [
    LightevalTaskConfig(
        name=f"belebele_native_instruct_{lang}_Latn",
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

BELEBELE_TASKS_EN_INSTRUCT = [
    LightevalTaskConfig(
        name=f"belebele_en_instruct_{lang}",
        prompt_function=belebele_prompt_en_instruct,
        suite=["extended"],
        hf_repo="facebook/belebele",
        hf_subset=lang,
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
                        language=iso_639_3_ind_to_iso_639_3_macro[LangCodeLanguage.get(lang).to_alpha3()],
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
        "acm_Arab",
        "arz_Arab",
        "ceb_Latn",
        "fin_Latn",
        "hin_Deva",
        "ita_Latn",
        "khm_Khmr",
        "lvs_Latn",
        "npi_Deva",
        "pol_Latn",
        "slv_Latn",
        "swe_Latn",
        # "tso_Latn",
        # "xho_Latn",
        "afr_Latn",
        "asm_Beng",
        "ces_Latn",
        "fra_Latn",
        "hin_Latn",
        "jav_Latn",
        # "kin_Latn",
        "mal_Mlym",
        "npi_Latn",
        "por_Latn",
        # "sna_Latn",
        "swh_Latn",
        "tur_Latn",
        "yor_Latn",
        "als_Latn",
        "azj_Latn",
        "ckb_Arab",
        # "fuv_Latn",
        "hrv_Latn",
        "jpn_Jpan",
        "kir_Cyrl",
        "mar_Deva",
        # "nso_Latn",
        "snd_Arab",
        "tam_Taml",
        "ukr_Cyrl",
        "zho_Hans",
        "amh_Ethi",
        # "bam_Latn",
        "dan_Latn",
        # "gaz_Latn",
        "hun_Latn",
        # "kac_Latn",
        "kor_Hang",
        "mkd_Cyrl",
        # "nya_Latn",
        "ron_Latn",
        "som_Latn",
        "tel_Telu",
        "urd_Arab",
        "zho_Hant",
        "apc_Arab",
        "ben_Beng",
        "deu_Latn",
        # "grn_Latn",
        "hye_Armn",
        "kan_Knda",
        "lao_Laoo",
        "mlt_Latn",
        "ory_Orya",
        "rus_Cyrl",
        # "sot_Latn",
        "tgk_Cyrl",
        "urd_Latn",
        "zsm_Latn",
        "arb_Arab",
        "ben_Latn",
        "ell_Grek",
        "guj_Gujr",
        # "ibo_Latn",
        "kat_Geor",
        # "lin_Latn",
        # "mri_Latn",
        "pan_Guru",
        # "shn_Mymr",
        "spa_Latn",
        "tgl_Latn",
        "uzn_Latn",
        # "zul_Latn",
        "arb_Latn",
        # "bod_Tibt",
        "eng_Latn",
        # "hat_Latn",
        # "ilo_Latn",
        "kaz_Cyrl",
        "lit_Latn",
        "mya_Mymr",
        "pbt_Arab",
        "sin_Latn",
        "srp_Cyrl",
        "tha_Thai",
        "vie_Latn",
        "ars_Arab",
        "bul_Cyrl",
        "est_Latn",
        # "hau_Latn",
        "ind_Latn",
        # "kea_Latn",
        # "lug_Latn",
        "nld_Latn",
        "pes_Arab",
        "sin_Sinh",
        # "ssw_Latn",
        # "tir_Ethi",
        "war_Latn",
        "ary_Arab",
        "cat_Latn",
        "eus_Latn",
        "heb_Hebr",
        "isl_Latn",
        # "khk_Cyrl",
        # "luo_Latn",
        "nob_Latn",
        "plt_Latn",
        "slk_Latn",
        # "sun_Latn",
        # "tsn_Latn",
        # "wol_Latn",
    ]
]
TASKS_TABLE.extend(BELEBELE_TASKS_NATIVE_INSTRUCT + BELEBELE_TASKS_EN_INSTRUCT)


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
        query_template = self.lang_to_template.get(self.lang, "eng")
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


def mmlu_pro(line, task_name: str = None):
    num_choices = len(line["options"])
    instruction = f"Given the following question about {line['category']} and answer choices, output the letter corresponding to the correct answer. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {' ,'.join(LETTER_INDICES[: num_choices - 1])}, or {LETTER_INDICES[num_choices]}. Think step by step before answering.\n\n"
    query = f"{instruction}###\nQuery:\n{line['question']}\n###\nChoices:"
    query += "".join([f"\n{key}) {choice}" for key, choice in zip(LETTER_INDICES, line["options"])])

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[:num_choices],
        gold_index=line["answer_index"],
        instruction=instruction,
    )


mmlu_pro = LightevalTaskConfig(
    name="mmlu_pro",
    suite=["extended"],
    prompt_function=mmlu_pro,
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_avail_splits=["validation", "test"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    stop_sequence=[],  # no stop sequence, will use eos token
    metric=[
        SampleLevelMetric(
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
        )
    ],
    trust_dataset=True,
    version=0,
)

TASKS_TABLE.append(mmlu_pro)
# print("\n".join([f"extended|{task.name}|0|0" for task in TASKS_TABLE]))
