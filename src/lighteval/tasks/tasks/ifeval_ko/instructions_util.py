# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from Allganize IFEval-Ko at revision 54199e3801116897697babf341865741dcd06fc8.
# https://huggingface.co/datasets/allganize/IFEval-Ko/blob/54199e3801116897697babf341865741dcd06fc8/ifeval_ko/instructions_util.py
# Ported to the Lighteval package.

"""Utility library of instructions."""

import os
import re

import immutabledict


RANK = os.environ.get("LOCAL_RANK", "0")

# ISO 639-1 codes to language names.
LANGUAGE_CODES = immutabledict.immutabledict(
    {
        "en": "English",
        "es": "Spanish",
        "pt": "Portuguese",
        "ar": "Arabic",
        "hi": "Hindi",
        "fr": "French",
        "ru": "Russian",
        "de": "German",
        "ja": "Japanese",
        "it": "Italian",
        "bn": "Bengali",
        "uk": "Ukrainian",
        "th": "Thai",
        "ur": "Urdu",
        "ta": "Tamil",
        "te": "Telugu",
        "bg": "Bulgarian",
        "ko": "Korean",
        "pl": "Polish",
        "he": "Hebrew",
        "fa": "Persian",
        "vi": "Vietnamese",
        "ne": "Nepali",
        "sw": "Swahili",
        "kn": "Kannada",
        "mr": "Marathi",
        "gu": "Gujarati",
        "pa": "Punjabi",
        "ml": "Malayalam",
        "fi": "Finnish",
    }
)

_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
# _STARTERS = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
# _ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"

_MIXED_ALPHABETS = "([A-Za-z가-힣])"  # 한글과 영어 모두 포함
_KOREAN_LIST = "([가나다라마바사])"  # 한글 리스트 마커


def split_into_sentences(text):
    """Split the text into sentences. (답변을 문장 단위로 분리합니다.)
    기존 함수를 이용합니다. 한국어 문장 생성에서도 중간에 약어 등은 영어로 표기될 수 있습니다.

    Args:
      text: A string that consists of more than or equal to one sentences.

    Returns:
      A list of strings where each string is a sentence.
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(
        _MULTIPLE_DOTS,
        lambda match: "<prd>" * len(match.group(0)) + "<stop>",
        text,
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")

    # text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(
        _MIXED_ALPHABETS + "[.]" + _MIXED_ALPHABETS + "[.]" + _MIXED_ALPHABETS + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )  # 영어/한국어 약어 처리
    text = re.sub(
        _MIXED_ALPHABETS + "[.]" + _MIXED_ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text
    )  # 영어/한국어 약어 처리

    # 기존 영어 약어 처리
    # text = re.sub(
    #     _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
    #     "\\1<prd>\\2<prd>\\3<prd>",
    #     text,
    # )
    # text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
    # text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text) # _STARTERS는 사용하지 않음

    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    text = re.sub(
        r"\s" + _ALPHABETS + r"[.]\s+(?=[가-힣])", " \\1<prd> ", text
    )  # 영어 약어 + 직후 한글이 적힐 시 온점 아님 처리
    text = re.sub(r"\s" + _KOREAN_LIST + r"[.]\s+", " \\1<prd> ", text)  # 한글로 된 리스트 마커 처리

    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def count_words(text):
    """Counts the number of words for Korean text.
    띄어쓰기를 기준으로 한국어 문장의 단어를 분리합니다."""
    # 기존 코드
    # tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    # tokens = tokenizer.tokenize(text)
    # num_words = len(tokens)

    text = text.strip()
    text = " ".join(text.split())
    if not text:
        return 0

    return len(text.split())


def count_sentences(text):
    """Count the number of sentences."""
    # tokenizer = _get_sentence_tokenizer()
    # tokenized_sentences = tokenizer.tokenize(text)
    tokenized_sentences = split_into_sentences(text)
    return len(tokenized_sentences)


# 제거된 원본 IFEval 함수
# def generate_keywords(num_keywords):
#     """Randomly generates a few keywords."""
#     return random.sample(WORD_LIST, k=num_keywords)

# @functools.lru_cache(maxsize=None)
# def _get_sentence_tokenizer():
#     return nltk.data.load("nltk:tokenizers/punkt/english.pickle")
