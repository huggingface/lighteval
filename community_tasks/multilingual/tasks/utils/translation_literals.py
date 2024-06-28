from typing import Literal


LANGS = Literal["ar", "en", "fr", "hi", "ru", "sw", "te", "th", "tr", "zh"]

LANG_NAMES = {
    "arabic": "ar",
    "bengali": "bn",
    "english": "en",
    "finnish": "fi",
    "indonesian": "id",
    "korean": "ko",
    "russian": "ru",
    "swahili": "sw",
    "telugu": "te",
    "afrikaans": "af",
    "chinese": "zh",
    "italian": "it",
    "javanese": "jv",
    "portuguese": "pt",
    "thai": "th",
    "vietnamese": "vi",
}

LANG_NAMES_INVERTED = {v: k for k, v in LANG_NAMES.items()}


QUESTION = {
    "ar": "سؤال",
    "en": "Question",
    "fr": "Question",
    "hi": "सवाल",
    "ru": "Вопрос",
    "sw": "Swali",
    "te": "ప్రశ్న",
    "th": "คำถาม",
    "tr": "Soru",
    "zh": "问题",
}
ANSWER = {
    "ar": "إجابة",
    "en": "Answer",
    "fr": "Réponse",
    "hi": "उत्तर",
    "ru": "ответ",
    "sw": "Jibu",
    "te": "సమాధానం",
    "th": "คำตอบ",
    "tr": "Cevap",
    "zh": "答案",
}

# Harness cfg
# TODO: AUTOGENERATE CHECK THESE
# We sure the XNLI definitions from eval harness
NLI_QUESTION = {
    "ar": "صحيح",
    "bg": "правилно",
    "de": "richtig",
    "el": "σωστός",
    "en": "right",
    "es": "correcto",
    "fr": "correct",
    "hi": "सही",
    "ru": "правильно",
    "sw": "sahihi",
    "th": "ถูกต้อง",
    "tr": "doğru",
    "ur": "صحیح",
    "vi": "đúng",
    "zh": "正确",
}

ENTAILMENT_LABELS = {
    "ar": "نعم",
    "bg": "да",
    "de": "Ja",
    "el": "Ναί",
    "en": "Yes",
    "es": "Sí",
    "fr": "Oui",
    "hi": "हाँ",
    "ru": "Да",
    "sw": "Ndiyo",
    "th": "ใช่",
    "tr": "Evet",
    "ur": "جی ہاں",
    "vi": "Vâng",
    "zh": "是的",
}

NEUTRAL_LABELS = {
    "ar": "لذا",
    "bg": "така",
    "de": "Auch",
    "el": "Έτσι",
    "en": "Also",
    "es": "Asi que",
    "fr": "Aussi",
    "hi": "इसलिए",
    "ru": "Так",
    "sw": "Hivyo",
    "th": "ดังนั้น",
    "tr": "Böylece",
    "ur": "اس لئے",
    "vi": "Vì vậy",
    "zh": "所以",
}

CONTRADICTION_LABELS = {
    "ar": "رقم",
    "bg": "не",
    "de": "Nein",
    "el": "όχι",
    "en": "No",
    "es": "No",
    "fr": "Non",
    "hi": "नहीं",
    "ru": "Нет",
    "sw": "Hapana",
    "th": "ไม่",
    "tr": "Hayır",
    "ur": "نہیں",
    "vi": "Không",
    "zh": "不是的",
}

IMPOSSIBLE = {
    "fr": "Impossible",
}
