from dataclasses import dataclass, field
from typing import Literal, get_args

from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.utils.language import Language


@dataclass
class TranslationLiterals:
    question: str
    options: str
    answer: str
    nli_assessment: str
    nli_entailment: str
    nli_contradiction: str
    nli_neutral: str
    cause: str
    effect: str

    # Punctuation
    full_stop: str
    comma: str
    question_mark: str
    exclamation_mark: str
    word_space: str
    sentence_space: str
    colon: str

    # Indices
    indices: list[str] = field(default_factory=lambda: LETTER_INDICES)


SUPPORTED_LANGUAGES = Literal[
    Language.english,
    Language.swahili,
    Language.french,
    Language.telugu,
    Language.hindi,
    Language.chinese,
    Language.russian,
    Language.thai,
    Language.turkish,
]
TRANSLATION_LITERALS: dict[SUPPORTED_LANGUAGES, TranslationLiterals] = {
    Language.english: TranslationLiterals(
        question="Question",
        options="Options",
        answer="Answer",
        nli_assessment="right",
        nli_entailment="Yes",
        nli_contradiction="No",
        nli_neutral="Also",
        cause="because",
        effect="therefore",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.swahili: TranslationLiterals(
        question="Swali",
        options="Chaguo",
        answer="Jibu",
        nli_assessment="sahihi",
        nli_entailment="Ndiyo",
        nli_contradiction="Hapana",
        nli_neutral="Pia",
        cause="kwa sababu",
        effect="kwa hiyo",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.french: TranslationLiterals(
        question="Question",
        options="Possibilités",
        answer="Réponse",
        nli_assessment="n'est-ce pas",
        nli_entailment="Oui",
        nli_contradiction="Non",
        nli_neutral="De plus",
        cause="parce que",
        effect="donc",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.telugu: TranslationLiterals(
        question="ప్రశ్న",
        options="ఎంపికలు",
        answer="జవాబు",
        nli_assessment="కదా",
        nli_entailment="అవును",
        nli_contradiction="కాదు",
        nli_neutral="అలాగే",
        cause="ఎందుకంటే",
        effect="అందువలన",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.hindi: TranslationLiterals(
        question="सवाल",
        options="विकल्प",
        answer="उत्तर",
        nli_assessment="है ना",
        nli_entailment="हाँ",
        nli_contradiction="नहीं",
        nli_neutral="साथ ही",
        cause="क्योंकि",
        effect="इसलिए",
        full_stop="।",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.chinese: TranslationLiterals(
        question="问题",
        options="选项",
        answer="答案",
        nli_assessment="是不是",
        nli_entailment="是的",
        nli_contradiction="不是",
        nli_neutral="而且",
        cause="因为",
        effect="所以",
        full_stop="。",
        comma="，",
        question_mark="？",
        exclamation_mark="！",
        word_space="",
        sentence_space="",
        colon="：",
    ),
    Language.russian: TranslationLiterals(
        question="Вопрос",
        options="Варианты",
        answer="ответ",
        nli_assessment="не так ли",
        nli_entailment="Да",
        nli_contradiction="Нет",
        nli_neutral="К тому же",
        cause="потому что",
        effect="поэтому",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.thai: TranslationLiterals(
        question="คำถาม",
        options="ตัวเลือก",
        answer="คำตอบ",
        nli_assessment="ใช่ไหม",
        nli_entailment="ใช่",
        nli_contradiction="ไม่",
        nli_neutral="และ",
        cause="เพราะ",
        effect="ดังนั้น",
        full_stop="",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space="",
        sentence_space=" ",
        colon=":",
    ),
    Language.turkish: TranslationLiterals(
        question="Soru",
        options="Seçenekler",
        answer="Cevap",
        nli_assessment="değil mi",
        nli_entailment="Evet",
        nli_contradiction="Hayır",
        nli_neutral="Ayrıca",
        cause="çünkü",
        effect="bu yüzden",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
}

assert set(get_args(SUPPORTED_LANGUAGES)) == set(TRANSLATION_LITERALS)
