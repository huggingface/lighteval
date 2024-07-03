from typing import Literal


LANGS = Literal["ar", "en", "fr", "hi", "ru", "sw", "te", "th", "tr", "zh"]

LANG_NAMES = {
    "arabic": "ar",
    "bengali": "bn",
    "english": "en",
    "finnish": "fi",
    "french": "fr",
    "indonesian": "id",
    "korean": "ko",
    "russian": "ru",
    "swahili": "sw",
    "telugu": "te",
    "afrikaans": "af",
    "chinese": "zh",
    "italian": "it",
    "javanese": "jv",
    "hindi": "hi",
    "portuguese": "pt",
    "thai": "th",
    "vietnamese": "vi",
}

NO_PUNCT_LANGS = ["th"]

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
    "sw": "sawa",
    "te": "సరియైనదా",
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
    "te": "అవును",
    "th": "ใช่",
    "tr": "Evet",
    "ur": "جی ہاں",
    "vi": "Vâng",
    "zh": "是的",
}

NEUTRAL_LABELS = {
    "ar": "و",
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
    "te": "అలాగే",
    "ur": "اس لئے",
    "vi": "Vì vậy",
    "zh": "所以",
}

CONTRADICTION_LABELS = {
    "ar": "لا",
    "bg": "не",
    "de": "Nein",
    "el": "όχι",
    "en": "No",
    "es": "No",
    "fr": "Non",
    "hi": "नहीं",
    "ru": "Нет",
    "sw": "Hapana",
    "te": "నం",
    "th": "ไม่",
    "tr": "Hayır",
    "ur": "نہیں",
    "vi": "Không",
    "zh": "不是的",
}

IMPOSSIBLE = {
    "fr": "Impossible",
}


CORRECT_LABELS = {
    "ar": "صح"
}

INCORRECT_LABELS = {
    "ar": "خطأ"
}

YES_LABELS = {
    "ar": "نعم",
    "hi": "हाँ",
    "fr": "Oui"
}

NO_LABELS = {
    "ar": "لا",
    "hi": "नहीं",
    "fr": "Non"
}

CAUSE_LABELS = {
    "ar": "لأن",
    "en": "because",
    "hi": "क्योंकि",
    "et": "sest",
    "ht": "poukisa",
    "it": "perché",
    "id": "karena",
    "qu": "imataq",
    "sw": "kwa sababu",
    "zh": "因为",
    "ta": "காரணமாக",
    "te": "ఎందుకంటే",
    "th": "เพราะ",
    "tr": "çünkü",
    "ru": "потому что",
    "vi": "bởi vì"
}

EFFECT_LABELS = {
    "ar": "لذلك",
    "en": "therefore",
    "hi": "इसलिए",
    "et": "seetõttu",
    "ht": "donk sa",
    "it": "quindi",
    "id": "maka",
    "qu": "chaymi",
    "sw": "kwa hiyo",
    "zh": "所以",
    "ta": "எனவே",
    "te": "అందువలన",
    "th": "ดังนั้น",
    "tr": "bu yüzden",
    "ru": "поэтому",
    "vi": "vì vậy"
}
