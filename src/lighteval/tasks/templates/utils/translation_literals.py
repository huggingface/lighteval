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

from dataclasses import dataclass, field

from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.utils.language import Language


# TODO(hynky1999): The typing still is not great, it should be able to infer that you can't access the
# attributes that are not defined in the class. Don't want to waste time on this though.
@dataclass
class TranslationLiterals:
    # This is just to create nice error messages
    language: Language

    # I hate these type errors too but we are living in python world
    # and typechecker is not respecting __getattribute__ all the time, where it would
    # find out that the value can't be None.
    question_word: str = None  # type: ignore
    options: str = None  # type: ignore
    answer: str = None  # type: ignore
    confirmation_word: str = None  # type: ignore
    yes: str = None  # type: ignore
    no: str = None  # type: ignore
    also: str = None  # type: ignore
    cause_word: str = None  # type: ignore
    effect_word: str = None  # type: ignore
    or_word: str = None  # type: ignore

    # NLI
    true: str = None  # type: ignore
    false: str = None  # type: ignore
    neither: str = None  # type: ignore

    # Punctuation
    full_stop: str = None  # type: ignore
    comma: str = None  # type: ignore
    question_mark: str = None  # type: ignore
    exclamation_mark: str = None  # type: ignore
    word_space: str = None  # type: ignore
    sentence_space: str = None  # type: ignore
    colon: str = ":"  # type: ignore
    semicolon: str = ";"  # type: ignore

    # Indices
    indices: list[str] = field(default_factory=lambda: LETTER_INDICES)

    def __getattribute__(self, name: str) -> str:
        value = super().__getattribute__(name)
        if value is None:
            raise AttributeError(
                f"""
Translation for '{name}' is needed for {self.language}. Please provide it's implementation by editing
the 'src/lighteval/tasks/templates/utils/translation_literals.py'
"""
            )
        return value


TRANSLATION_LITERALS: dict[Language, TranslationLiterals] = {
    Language.ENGLISH: TranslationLiterals(
        language=Language.ENGLISH,
        question_word="question",
        options="options",
        answer="answer",
        confirmation_word="right",
        yes="yes",
        no="no",
        also="also",
        cause_word="because",
        effect_word="therefore",
        true="true",
        false="false",
        neither="neither",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        or_word="or",
    ),
    Language.SWAHILI: TranslationLiterals(
        language=Language.SWAHILI,
        question_word="swali",
        options="chaguo",
        answer="jibu",
        confirmation_word="sahihi",
        yes="ndiyo",
        no="hapana",
        also="pia",
        cause_word="kwa sababu",
        effect_word="kwa hiyo",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.FRENCH: TranslationLiterals(
        language=Language.FRENCH,
        question_word="question",
        options="possibilités",
        answer="réponse",
        confirmation_word="n'est-ce pas",
        yes="oui",
        no="non",
        also="de plus",
        cause_word="parce que",
        effect_word="donc",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.TELUGU: TranslationLiterals(
        language=Language.TELUGU,
        question_word="ప్రశ్న",
        options="ఎంపికలు",
        answer="జవాబు",
        confirmation_word="కదా",
        yes="అవును",
        no="కాదు",
        also="అలాగే",
        cause_word="ఎందుకంటే",
        effect_word="అందువలన",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.HINDI: TranslationLiterals(
        language=Language.HINDI,
        question_word="सवाल",
        options="विकल्प",
        answer="उत्तर",
        confirmation_word="है ना",
        yes="हाँ",
        no="नहीं",
        also="साथ ही",
        cause_word="क्योंकि",
        effect_word="इसलिए",
        full_stop="।",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.CHINESE: TranslationLiterals(
        language=Language.CHINESE,
        question_word="问题",
        options="选项",
        answer="答案",
        confirmation_word="是不是",
        yes="是的",
        no="不是",
        also="而且",
        cause_word="因为",
        effect_word="所以",
        full_stop="。",
        comma="，",
        question_mark="？",
        exclamation_mark="！",
        word_space="",
        sentence_space="",
        colon="：",
    ),
    Language.RUSSIAN: TranslationLiterals(
        language=Language.RUSSIAN,
        question_word="вопрос",
        options="варианты",
        answer="ответ",
        confirmation_word="не так ли",
        yes="да",
        no="нет",
        also="к тому же",
        cause_word="потому что",
        effect_word="поэтому",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.THAI: TranslationLiterals(
        language=Language.THAI,
        question_word="คำถาม",
        options="ตัวเลือก",
        answer="คำตอบ",
        confirmation_word="ใช่ไหม",
        yes="ใช่",
        no="ไม่",
        also="และ",
        cause_word="เพราะ",
        effect_word="ดังนั้น",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space="",
        sentence_space=" ",
        colon=":",
    ),
    Language.TURKISH: TranslationLiterals(
        language=Language.TURKISH,
        question_word="soru",
        options="seçenekler",
        answer="cevap",
        confirmation_word="değil mi",
        yes="evet",
        no="hayır",
        also="ayrıca",
        cause_word="çünkü",
        effect_word="bu yüzden",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
}
