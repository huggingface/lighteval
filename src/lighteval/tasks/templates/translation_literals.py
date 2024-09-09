from dataclasses import dataclass, field

from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.utils.language import Language


@dataclass
class TranslationLiterals:
    # This is just to create nice error messages
    language: Language

    # I hate these type errors too but we are living in python world
    # and typechecker is not respecting __getattribute__, where it would
    # find out that the value can't be None.
    question: str = None  # type: ignore
    options: str = None  # type: ignore
    answer: str = None  # type: ignore
    assessive_right: str = None  # type: ignore
    yes: str = None  # type: ignore
    no: str = None  # type: ignore
    entailment: str = None  # type: ignore
    contradiction: str = None  # type: ignore
    neutral: str = None  # type: ignore
    also: str = None  # type: ignore
    because: str = None  # type: ignore
    therefore: str = None  # type: ignore
    cause: str = None  # type: ignore
    effect: str = None  # type: ignore

    # Punctuation
    full_stop: str = None  # type: ignore
    comma: str = None  # type: ignore
    question_mark: str = None  # type: ignore
    exclamation_mark: str = None  # type: ignore
    word_space: str = None  # type: ignore
    sentence_space: str = None  # type: ignore
    colon: str = None  # type: ignore

    # Indices
    indices: list[str] = field(default_factory=lambda: LETTER_INDICES)

    def __getattribute__(self, name: str) -> str:
        value = super().__getattribute__(name)
        if value is None:
            raise AttributeError(
                f"Translation for '{name}' is needed for {self.language}. Please provide it by editing TODO"
            )
        return value


TRANSLATION_LITERALS: dict[Language, TranslationLiterals] = {
    Language.english: TranslationLiterals(
        language=Language.english,
        question="question",
        options="options",
        answer="answer",
        entailment="entailment",
        contradiction="contradiction",
        neutral="neutral",
        assessive_right="right",
        yes="yes",
        no="no",
        also="also",
        because="because",
        therefore="therefore",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.swahili: TranslationLiterals(
        language=Language.swahili,
        question="swali",
        options="chaguo",
        answer="jibu",
        assessive_right="sahihi",
        yes="ndiyo",
        no="hapana",
        also="pia",
        because="kwa sababu",
        therefore="kwa hiyo",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.french: TranslationLiterals(
        language=Language.french,
        question="question",
        options="possibilités",
        answer="réponse",
        assessive_right="n'est-ce pas",
        yes="oui",
        no="non",
        also="de plus",
        because="parce que",
        therefore="donc",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.telugu: TranslationLiterals(
        language=Language.telugu,
        question="ప్రశ్న",
        options="ఎంపికలు",
        answer="జవాబు",
        assessive_right="కదా",
        yes="అవును",
        no="కాదు",
        also="అలాగే",
        because="ఎందుకంటే",
        therefore="అందువలన",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.hindi: TranslationLiterals(
        language=Language.hindi,
        question="सवाल",
        options="विकल्प",
        answer="उत्तर",
        assessive_right="है ना",
        yes="हाँ",
        no="नहीं",
        also="साथ ही",
        because="क्योंकि",
        therefore="इसलिए",
        full_stop="।",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.chinese: TranslationLiterals(
        language=Language.chinese,
        question="问题",
        options="选项",
        answer="答案",
        assessive_right="是不是",
        yes="是的",
        no="不是",
        also="而且",
        because="因为",
        therefore="所以",
        full_stop="。",
        comma="，",
        question_mark="？",
        exclamation_mark="！",
        word_space="",
        sentence_space="",
        colon="：",
    ),
    Language.russian: TranslationLiterals(
        language=Language.russian,
        question="вопрос",
        options="варианты",
        answer="ответ",
        assessive_right="не так ли",
        yes="да",
        no="нет",
        also="к тому же",
        because="потому что",
        therefore="поэтому",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.thai: TranslationLiterals(
        language=Language.thai,
        question="คำถาม",
        options="ตัวเลือก",
        answer="คำตอบ",
        assessive_right="ใช่ไหม",
        yes="ใช่",
        no="ไม่",
        also="และ",
        because="เพราะ",
        therefore="ดังนั้น",
        full_stop="",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space="",
        sentence_space=" ",
        colon=":",
    ),
    Language.turkish: TranslationLiterals(
        language=Language.turkish,
        question="soru",
        options="seçenekler",
        answer="cevap",
        assessive_right="değil mi",
        yes="evet",
        no="hayır",
        also="ayrıca",
        because="çünkü",
        therefore="bu yüzden",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
}
