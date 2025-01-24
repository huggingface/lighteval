# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Callable, Iterator

from lighteval.utils.imports import (
    NO_SPACY_TOKENIZER_ERROR_MSG,
    NO_STANZA_TOKENIZER_ERROR_MSG,
    can_load_spacy_tokenizer,
    can_load_stanza_tokenizer,
)
from lighteval.utils.language import Language


logger = logging.getLogger(__name__)


# Copy of https://github.com/huggingface/datatrove/blob/main/src/datatrove/utils/tokenization.py
def strip_strings(els: list[str]) -> list[str]:
    return [el.strip() for el in els if len(el.strip()) > 0]


def simple_span_tokenize(text: str, sents: list[str]) -> Iterator[tuple[int, int]]:
    start_index = 0
    for sent in sents:
        start_char = text.index(sent, start_index)
        end_char = start_char + len(sent)
        start_index = end_char
        yield start_char, end_char


class WordTokenizer(ABC):
    @abstractmethod
    def word_tokenize(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def sent_tokenize(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        pass


class WhitespaceTokenizer(WordTokenizer):
    def word_tokenize(self, text: str) -> list[str]:
        return text.split()

    def sent_tokenize(self, text: str) -> list[str]:
        return text.split()

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        return list(simple_span_tokenize(text, text.split()))


class NLTKTokenizer(WordTokenizer):
    def __init__(self, punkt_language: str):
        super().__init__()
        self.punkt_language = punkt_language
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from nltk import download, load

            download("punkt_tab")
            self._tokenizer = load(f"tokenizers/punkt/{self.punkt_language}.pickle")
        return self._tokenizer

    def word_tokenize(self, text) -> list[str]:
        from nltk.tokenize import word_tokenize

        tokens = word_tokenize(text, language=self.punkt_language)
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from nltk.tokenize import sent_tokenize

        sents = sent_tokenize(text, language=self.punkt_language)
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        return list(self.tokenizer.span_tokenize(text))


class SpaCyTokenizer(WordTokenizer):
    def __init__(self, spacy_language: str, config=None):
        super().__init__()
        if not can_load_spacy_tokenizer(spacy_language):
            raise ImportError(NO_SPACY_TOKENIZER_ERROR_MSG)
        self.spacy_language = spacy_language
        self.config = config
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import spacy

            if self.config is None:
                self._tokenizer = spacy.blank(self.spacy_language)
            else:
                self._tokenizer = spacy.blank(self.spacy_language, config=self.config)
            self._tokenizer.add_pipe("sentencizer")
        return self._tokenizer

    def word_tokenize(self, text: str) -> list[str]:
        self.tokenizer.max_length = len(text) + 10
        tokens = [token.text for token in self.tokenizer(text, disable=["parser", "tagger", "ner"])]
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        self.tokenizer.max_length = len(text) + 10
        sents = [sent.text for sent in self.tokenizer(text, disable=["parser", "tagger", "ner"]).sents]
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        return [
            (sent.start_char, sent.end_char)
            for sent in self.tokenizer(text, disable=["parser", "tagger", "ner"]).sents
        ]


class StanzaTokenizer(WordTokenizer):
    def __init__(self, stanza_language: str, **stanza_kwargs):
        super().__init__()
        if not can_load_stanza_tokenizer():
            raise ImportError(NO_STANZA_TOKENIZER_ERROR_MSG)
        self.stanza_language = stanza_language
        self.stanza_kwargs = stanza_kwargs
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import stanza
            from stanza.pipeline.core import DownloadMethod

            self._tokenizer = stanza.Pipeline(
                self.stanza_language,
                processors="tokenize",
                download_method=DownloadMethod.REUSE_RESOURCES,
                **self.stanza_kwargs,
            )

        return self._tokenizer

    def word_tokenize(self, text: str) -> list[str]:
        doc = self.tokenizer(text)
        tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        doc = self.tokenizer(text)
        sents = [sentence.text for sentence in doc.sentences]
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        doc = self.tokenizer(text)
        return [(sent.tokens[0].start_char, sent.tokens[-1].end_char) for sent in doc.sentences]


# If you know a better tokenizer or better proxy language, please submit a PR
TOKENIZER_FACTORY: dict[Language, Callable[[], WordTokenizer]] = {
    Language.ENGLISH: lambda: SpaCyTokenizer("en"),
    Language.KOREAN: lambda: SpaCyTokenizer("ko"),
    Language.GERMAN: lambda: SpaCyTokenizer("de"),
    Language.FRENCH: lambda: SpaCyTokenizer("fr"),
    Language.CZECH: lambda: SpaCyTokenizer("cz"),
    Language.DANISH: lambda: SpaCyTokenizer("da"),
    Language.DUTCH: lambda: SpaCyTokenizer("nl"),
    Language.ESTONIAN: lambda: SpaCyTokenizer("et"),
    Language.FINNISH: lambda: SpaCyTokenizer("fi"),
    Language.GREEK: lambda: SpaCyTokenizer("el"),
    Language.ITALIAN: lambda: SpaCyTokenizer("it"),
    Language.MALAYALAM: lambda: SpaCyTokenizer("ml"),
    Language.NORWEGIAN: lambda: SpaCyTokenizer("nb"),
    Language.POLISH: lambda: SpaCyTokenizer("pl"),
    Language.PORTUGUESE: lambda: SpaCyTokenizer("pt"),
    Language.RUSSIAN: lambda: SpaCyTokenizer("ru"),
    Language.SLOVENIAN: lambda: SpaCyTokenizer("sl"),
    Language.SPANISH: lambda: SpaCyTokenizer("es"),
    Language.SWEDISH: lambda: SpaCyTokenizer("sv"),
    Language.TURKISH: lambda: SpaCyTokenizer("tr"),
    Language.CHINESE: lambda: SpaCyTokenizer("zh", {"nlp": {"tokenizer": {"segmenter": "jieba"}}}),
    Language.JAPANESE: lambda: SpaCyTokenizer("ja"),  # note that there are some issues for >50k chars text
    Language.VIETNAMESE: lambda: SpaCyTokenizer("vi"),
    Language.INDONESIAN: lambda: SpaCyTokenizer("id"),
    Language.PERSIAN: lambda: SpaCyTokenizer("fa"),
    Language.ARABIC: lambda: SpaCyTokenizer("ar"),
    Language.HINDI: lambda: SpaCyTokenizer("hi"),
    Language.TAMIL: lambda: SpaCyTokenizer("ta"),
    Language.URDU: lambda: SpaCyTokenizer("ur"),
    Language.MARATHI: lambda: SpaCyTokenizer("mr"),
    Language.TELUGU: lambda: SpaCyTokenizer("te"),
    Language.HUNGARIAN: lambda: SpaCyTokenizer("hu"),
    Language.ROMANIAN: lambda: SpaCyTokenizer("ro"),
    Language.UKRAINIAN: lambda: SpaCyTokenizer("uk"),
    Language.SLOVAK: lambda: SpaCyTokenizer("sk"),
    Language.BULGARIAN: lambda: SpaCyTokenizer("bg"),
    Language.CATALAN: lambda: SpaCyTokenizer("ca"),
    Language.CROATIAN: lambda: SpaCyTokenizer("hr"),
    Language.LATIN: lambda: SpaCyTokenizer("la"),
    Language.SERBIAN: lambda: SpaCyTokenizer("sr"),
    Language.LITHUANIAN: lambda: SpaCyTokenizer("lt"),
    Language.HEBREW: lambda: SpaCyTokenizer("he"),
    Language.LATVIAN: lambda: SpaCyTokenizer("lv"),
    Language.ICELANDIC: lambda: SpaCyTokenizer("is"),
    Language.ARMENIAN: lambda: SpaCyTokenizer("hy"),
    Language.BASQUE: lambda: SpaCyTokenizer("eu"),
    Language.THAI: lambda: SpaCyTokenizer("th"),
    Language.TAGALOG: lambda: SpaCyTokenizer("tl"),
    Language.ALBANIAN: lambda: SpaCyTokenizer("sq"),
    Language.MACEDONIAN: lambda: SpaCyTokenizer("mk"),
    Language.AZERBAIJANI: lambda: SpaCyTokenizer("az"),
    Language.AMHARIC: lambda: SpaCyTokenizer("am"),
    Language.BENGALI: lambda: SpaCyTokenizer("bn"),
    Language.MALAY: lambda: SpaCyTokenizer("ms"),
    Language.NEPALI: lambda: SpaCyTokenizer("ne"),
    Language.KAZAKH: lambda: StanzaTokenizer("kk"),
    Language.GUJARATI: lambda: SpaCyTokenizer("gu"),
    Language.KANNADA: lambda: SpaCyTokenizer("kn"),
    Language.WELSH: lambda: StanzaTokenizer("cy"),
    Language.NORWEGIAN_NYNORSK: lambda: SpaCyTokenizer("nn"),
    Language.SINHALA: lambda: SpaCyTokenizer("si"),
    Language.TATAR: lambda: SpaCyTokenizer("tt"),
    Language.AFRIKAANS: lambda: SpaCyTokenizer("af"),
    Language.KIRGHIZ: lambda: SpaCyTokenizer("ky"),
    Language.IRISH: lambda: SpaCyTokenizer("ga"),
    Language.LUXEMBOURGISH: lambda: SpaCyTokenizer("lb"),
    Language.MALTESE: lambda: StanzaTokenizer("mt"),
    Language.SANSKRIT: lambda: SpaCyTokenizer("sa"),
    Language.YORUBA: lambda: SpaCyTokenizer("yo"),
    Language.SERBOCROATIAN: lambda: SpaCyTokenizer("sr"),
    Language.BELARUSIAN: lambda: StanzaTokenizer("be"),
    # proxies
    Language.BOSNIAN: lambda: SpaCyTokenizer("hr"),  # Proxy
    Language.ESPERANTO: lambda: SpaCyTokenizer("en"),  # Proxy
    Language.OCCITAN: lambda: SpaCyTokenizer("ca"),  # Proxy
    Language.SWAHILI: lambda: SpaCyTokenizer("en"),  # Proxy
    Language.JAVANESE: lambda: SpaCyTokenizer("en"),  # Proxy
    Language.UZBEK: lambda: SpaCyTokenizer("tr"),  # Proxy, alternative ru
    Language.TAJIK: lambda: SpaCyTokenizer("ru"),  # Proxy
    Language.KURDISH: lambda: SpaCyTokenizer("en"),  # Proxy, multiple scripts!
    Language.BASHKIR: lambda: SpaCyTokenizer("tt"),  # Proxy
    Language.WESTERN_FRISIAN: lambda: SpaCyTokenizer("nl"),  # Proxy
    Language.BRETON: lambda: StanzaTokenizer("cy"),  # Proxy
    Language.MALAGASY: lambda: SpaCyTokenizer("en"),  # Proxy
    Language.YIDDISH: lambda: SpaCyTokenizer("he"),  # Proxy
    Language.SOMALI: lambda: SpaCyTokenizer("en"),  # Proxy
    Language.TURKMEN: lambda: SpaCyTokenizer("tr"),  # Proxy
    Language.PASHTO: lambda: SpaCyTokenizer("xx"),  # Proxy. xx is "multi-language"
}


@lru_cache
def get_word_tokenizer(language: Language) -> WordTokenizer:
    tokenizer = TOKENIZER_FACTORY.get(language)
    if tokenizer is None:
        logger.warning(f"No word tokenizer found for language {language}, will split on spaces.")
        return WhitespaceTokenizer()
    return tokenizer()
