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

    question_word: str = None  # type: ignore
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
    full_stop: str = "."
    comma: str = ","
    question_mark: str = "?"
    exclamation_mark: str = "!"
    word_space: str = " "
    sentence_space: str = " "
    colon: str = ":"
    semicolon: str = ";"

    # Indices
    indices: list[str] = field(default_factory=lambda: LETTER_INDICES)

    def __getattribute__(self, name: str) -> str:
        value = super().__getattribute__(name)
        if value is None:
            raise AttributeError(
                f"""
Translation for '{name}' is needed for {self.language}. Please provide its implementation by editing
the 'src/lighteval/tasks/templates/utils/translation_literals.py'
"""
            )
        return value


TRANSLATION_LITERALS: dict[Language, TranslationLiterals] = {
    Language.ENGLISH: TranslationLiterals(
        language=Language.ENGLISH,
        question_word="question",
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
    Language.ARABIC: TranslationLiterals(
        language=Language.ARABIC,
        question_word="سؤال",
        answer="إجابة",
        confirmation_word="صحيح",
        yes="نعم",
        no="لا",
        also="كذلك",
        cause_word="لأن",
        effect_word="لذلك",
        full_stop=".",
        comma="،",
        question_mark="؟",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.SWAHILI: TranslationLiterals(
        language=Language.SWAHILI,
        question_word="swali",
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
    Language.SPANISH: TranslationLiterals(language=Language.SPANISH),
    Language.PORTUGUESE: TranslationLiterals(language=Language.PORTUGUESE),
    Language.ITALIAN: TranslationLiterals(language=Language.ITALIAN),
    Language.ROMANIAN: TranslationLiterals(language=Language.ROMANIAN),
    Language.GERMAN: TranslationLiterals(language=Language.GERMAN),
    Language.LATIN: TranslationLiterals(language=Language.LATIN),
    Language.CZECH: TranslationLiterals(language=Language.CZECH),
    Language.DANISH: TranslationLiterals(language=Language.DANISH),
    Language.FINNISH: TranslationLiterals(language=Language.FINNISH),
    Language.GREEK: TranslationLiterals(language=Language.GREEK),
    Language.NORWEGIAN: TranslationLiterals(language=Language.NORWEGIAN),
    Language.POLISH: TranslationLiterals(language=Language.POLISH),
    Language.SLOVENIAN: TranslationLiterals(language=Language.SLOVENIAN),
    Language.DUTCH: TranslationLiterals(language=Language.DUTCH),
    Language.JAPANESE: TranslationLiterals(language=Language.JAPANESE),
    Language.VIETNAMESE: TranslationLiterals(language=Language.VIETNAMESE),
    Language.INDONESIAN: TranslationLiterals(language=Language.INDONESIAN),
    Language.PERSIAN: TranslationLiterals(language=Language.PERSIAN),
    Language.KOREAN: TranslationLiterals(language=Language.KOREAN),
    Language.BENGALI: TranslationLiterals(language=Language.BENGALI),
    Language.TAMIL: TranslationLiterals(language=Language.TAMIL),
    Language.HUNGARIAN: TranslationLiterals(language=Language.HUNGARIAN),
    Language.UKRAINIAN: TranslationLiterals(language=Language.UKRAINIAN),
    Language.SLOVAK: TranslationLiterals(language=Language.SLOVAK),
    Language.BULGARIAN: TranslationLiterals(language=Language.BULGARIAN),
    Language.CATALAN: TranslationLiterals(language=Language.CATALAN),
    Language.CROATIAN: TranslationLiterals(language=Language.CROATIAN),
    Language.SERBIAN: TranslationLiterals(language=Language.SERBIAN),
    Language.LITHUANIAN: TranslationLiterals(language=Language.LITHUANIAN),
    Language.ESTONIAN: TranslationLiterals(language=Language.ESTONIAN),
    Language.HEBREW: TranslationLiterals(language=Language.HEBREW),
    Language.LATVIAN: TranslationLiterals(language=Language.LATVIAN),
    Language.SERBOCROATIAN: TranslationLiterals(language=Language.SERBOCROATIAN),  # Deprecated
    Language.ALBANIAN: TranslationLiterals(language=Language.ALBANIAN),
    Language.AZERBAIJANI: TranslationLiterals(language=Language.AZERBAIJANI),
    Language.ICELANDIC: TranslationLiterals(language=Language.ICELANDIC),
    Language.MACEDONIAN: TranslationLiterals(language=Language.MACEDONIAN),
    Language.GEORGIAN: TranslationLiterals(language=Language.GEORGIAN),
    Language.GALICIAN: TranslationLiterals(language=Language.GALICIAN),
    Language.ARMENIAN: TranslationLiterals(language=Language.ARMENIAN),
    Language.BASQUE: TranslationLiterals(language=Language.BASQUE),
    Language.MALAY: TranslationLiterals(language=Language.MALAY),
    Language.TAGALOG: TranslationLiterals(language=Language.TAGALOG),
    Language.JAVANESE: TranslationLiterals(language=Language.JAVANESE),
    Language.PUNJABI: TranslationLiterals(language=Language.PUNJABI),
    Language.BIHARI: TranslationLiterals(language=Language.BIHARI),  # Deprecated
    Language.GUJARATI: TranslationLiterals(language=Language.GUJARATI),
    Language.YORUBA: TranslationLiterals(language=Language.YORUBA),
    Language.MARATHI: TranslationLiterals(language=Language.MARATHI),
    Language.URDU: TranslationLiterals(language=Language.URDU),
    Language.AMHARIC: TranslationLiterals(language=Language.AMHARIC),
    Language.MALAYALAM: TranslationLiterals(language=Language.MALAYALAM),
    Language.KANNADA: TranslationLiterals(language=Language.KANNADA),
    Language.NEPALI: TranslationLiterals(language=Language.NEPALI),
    Language.KAZAKH: TranslationLiterals(language=Language.KAZAKH),
    Language.BELARUSIAN: TranslationLiterals(language=Language.BELARUSIAN),
    Language.BURMESE: TranslationLiterals(language=Language.BURMESE),
    Language.ESPERANTO: TranslationLiterals(language=Language.ESPERANTO),
    Language.UZBEK: TranslationLiterals(language=Language.UZBEK),
    Language.KHMER: TranslationLiterals(language=Language.KHMER),
    Language.TAJIK: TranslationLiterals(language=Language.TAJIK),
    Language.WELSH: TranslationLiterals(language=Language.WELSH),
    Language.NORWEGIAN_NYNORSK: TranslationLiterals(language=Language.NORWEGIAN_NYNORSK),
    Language.BOSNIAN: TranslationLiterals(language=Language.BOSNIAN),
    Language.SINHALA: TranslationLiterals(language=Language.SINHALA),
    Language.TATAR: TranslationLiterals(language=Language.TATAR),
    Language.AFRIKAANS: TranslationLiterals(language=Language.AFRIKAANS),
    Language.ORIYA: TranslationLiterals(language=Language.ORIYA),
    Language.KIRGHIZ: TranslationLiterals(language=Language.KIRGHIZ),
    Language.IRISH: TranslationLiterals(language=Language.IRISH),
    Language.OCCITAN: TranslationLiterals(language=Language.OCCITAN),
    Language.KURDISH: TranslationLiterals(language=Language.KURDISH),
    Language.LAO: TranslationLiterals(language=Language.LAO),
    Language.LUXEMBOURGISH: TranslationLiterals(language=Language.LUXEMBOURGISH),
    Language.BASHKIR: TranslationLiterals(language=Language.BASHKIR),
    Language.WESTERN_FRISIAN: TranslationLiterals(language=Language.WESTERN_FRISIAN),
    Language.PASHTO: TranslationLiterals(language=Language.PASHTO),
    Language.MALTESE: TranslationLiterals(language=Language.MALTESE),
    Language.BRETON: TranslationLiterals(language=Language.BRETON),
    Language.ASSAMESE: TranslationLiterals(language=Language.ASSAMESE),
    Language.MALAGASY: TranslationLiterals(language=Language.MALAGASY),
    Language.DIVEHI: TranslationLiterals(language=Language.DIVEHI),
    Language.YIDDISH: TranslationLiterals(language=Language.YIDDISH),
    Language.SOMALI: TranslationLiterals(language=Language.SOMALI),
    Language.SANSKRIT: TranslationLiterals(language=Language.SANSKRIT),
    Language.SINDHI: TranslationLiterals(language=Language.SINDHI),
    Language.TURKMEN: TranslationLiterals(language=Language.TURKMEN),
    Language.SOUTH_AZERBAIJANI: TranslationLiterals(language=Language.SOUTH_AZERBAIJANI),
    Language.SORANI: TranslationLiterals(language=Language.SORANI),
    Language.CEBUANO: TranslationLiterals(language=Language.CEBUANO),
    Language.WAR: TranslationLiterals(language=Language.WAR),
    Language.SWEDISH: TranslationLiterals(language=Language.SWEDISH),
}
