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

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Callable, Iterator

from lighteval.logging.hierarchical_logger import hlog_warn
from lighteval.utils.imports import check_required_dependencies
from lighteval.utils.language import Language


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
        check_required_dependencies(f"{punkt_language} word tokenizer", ["nltk"])
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
        check_required_dependencies(f"{spacy_language} word tokenizer", ["spacy"])
        if spacy_language == "vi":
            check_required_dependencies(f"{spacy_language} word tokenizer", ["pyvi"])
        elif spacy_language == "zh":
            check_required_dependencies(f"{spacy_language} word tokenizer", ["jieba"])
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
        check_required_dependencies(f"{stanza_language} word tokenizer", ["stanza"])
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


class ThaiTokenizer(WordTokenizer):
    def __init__(self):
        super().__init__()
        check_required_dependencies("th word tokenizer", ["pythainlp"])

    def word_tokenize(self, text: str) -> list[str]:
        from pythainlp.tokenize import word_tokenize as th_word_tokenize

        tokens = th_word_tokenize(text, keep_whitespace=False, engine="newmm-safe")
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from pythainlp.tokenize import sent_tokenize as th_sent_tokenize

        sents = th_sent_tokenize(text)
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        sents = self.sent_tokenize(text)
        return list(simple_span_tokenize(text, sents))


class IndicNLPTokenizer(WordTokenizer):
    def __init__(self, language: str):
        super().__init__()
        self.language = language
        check_required_dependencies(f"{language} word tokenizer", [("indicnlp", "indic-nlp-library")])

    def word_tokenize(self, text) -> list[str]:
        from indicnlp.tokenize.indic_tokenize import trivial_tokenize as indicnlp_trivial_tokenize

        tokens = indicnlp_trivial_tokenize(text, self.language)
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from indicnlp.tokenize.sentence_tokenize import sentence_split

        sents = sentence_split(text, lang=self.language)
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        sents = self.sent_tokenize(text)
        return list(simple_span_tokenize(text, sents))


class KiwiTokenizer(WordTokenizer):
    def __init__(self, model_type="sbg"):
        super().__init__()
        check_required_dependencies("ko word tokenizer", ["kiwipiepy"])
        self.model_type = model_type
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from kiwipiepy import Kiwi

            self._tokenizer = Kiwi(model_type=self.model_type)
        return self._tokenizer

    def word_tokenize(self, text: str) -> list[str]:
        tokens = [token.form for token in self.tokenizer.tokenize(text)]
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        sents = [sent.text for sent in self.tokenizer.split_into_sents(text)]
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        return [(sent.start, sent.end) for sent in self.tokenizer.split_into_sents(text)]


# If you know a better tokenizer or better proxy language, please submit a PR
TOKENIZER_FACTORY: dict[Language, Callable[[], WordTokenizer]] = {
    Language.english: lambda: SpaCyTokenizer("en"),
    Language.korean: lambda: KiwiTokenizer(),
    Language.german: lambda: SpaCyTokenizer("de"),
    Language.french: lambda: SpaCyTokenizer("fr"),
    Language.czech: lambda: SpaCyTokenizer("cz"),
    Language.danish: lambda: SpaCyTokenizer("da"),
    Language.dutch: lambda: SpaCyTokenizer("nl"),
    Language.estonian: lambda: SpaCyTokenizer("et"),
    Language.finnish: lambda: SpaCyTokenizer("fi"),
    Language.greek: lambda: SpaCyTokenizer("el"),
    Language.italian: lambda: SpaCyTokenizer("it"),
    Language.malayalam: lambda: SpaCyTokenizer("ml"),
    Language.norwegian: lambda: SpaCyTokenizer("nb"),
    Language.polish: lambda: SpaCyTokenizer("pl"),
    Language.portuguese: lambda: SpaCyTokenizer("pt"),
    Language.russian: lambda: SpaCyTokenizer("ru"),
    Language.slovenian: lambda: SpaCyTokenizer("sl"),
    Language.spanish: lambda: SpaCyTokenizer("es"),
    Language.swedish: lambda: SpaCyTokenizer("sv"),
    Language.turkish: lambda: SpaCyTokenizer("tr"),
    Language.chinese: lambda: SpaCyTokenizer("zh", {"nlp": {"tokenizer": {"segmenter": "jieba"}}}),
    Language.japanese: lambda: SpaCyTokenizer("ja"),  # note that there are some issues for >50k chars text
    Language.vietnamese: lambda: SpaCyTokenizer("vi"),
    Language.indonesian: lambda: SpaCyTokenizer("id"),
    Language.persian: lambda: SpaCyTokenizer("fa"),
    Language.arabic: lambda: SpaCyTokenizer("ar"),
    Language.hindi: lambda: SpaCyTokenizer("hi"),
    Language.tamil: lambda: SpaCyTokenizer("ta"),
    Language.urdu: lambda: SpaCyTokenizer("ur"),
    Language.marathi: lambda: SpaCyTokenizer("mr"),
    Language.telugu: lambda: SpaCyTokenizer("te"),
    Language.hungarian: lambda: SpaCyTokenizer("hu"),
    Language.romanian: lambda: SpaCyTokenizer("ro"),
    Language.ukrainian: lambda: SpaCyTokenizer("uk"),
    Language.slovak: lambda: SpaCyTokenizer("sk"),
    Language.bulgarian: lambda: SpaCyTokenizer("bg"),
    Language.catalan: lambda: SpaCyTokenizer("ca"),
    Language.croatian: lambda: SpaCyTokenizer("hr"),
    Language.latin: lambda: SpaCyTokenizer("la"),
    Language.serbian: lambda: SpaCyTokenizer("sr"),
    Language.lithuanian: lambda: SpaCyTokenizer("lt"),
    Language.hebrew: lambda: SpaCyTokenizer("he"),
    Language.latvian: lambda: SpaCyTokenizer("lv"),
    Language.icelandic: lambda: SpaCyTokenizer("is"),
    Language.armenian: lambda: SpaCyTokenizer("hy"),
    Language.basque: lambda: SpaCyTokenizer("eu"),
    Language.thai: lambda: ThaiTokenizer(),
    Language.tagalog: lambda: SpaCyTokenizer("tl"),
    Language.albanian: lambda: SpaCyTokenizer("sq"),
    Language.macedonian: lambda: SpaCyTokenizer("mk"),
    Language.azerbaijani: lambda: SpaCyTokenizer("az"),
    Language.amharic: lambda: SpaCyTokenizer("am"),
    Language.bengali: lambda: SpaCyTokenizer("bn"),
    Language.malay: lambda: SpaCyTokenizer("ms"),
    Language.nepali: lambda: SpaCyTokenizer("ne"),
    Language.kazakh: lambda: StanzaTokenizer("kk"),
    Language.gujarati: lambda: SpaCyTokenizer("gu"),
    Language.kannada: lambda: SpaCyTokenizer("kn"),
    Language.welsh: lambda: StanzaTokenizer("cy"),
    Language.norwegian_nynorsk: lambda: SpaCyTokenizer("nn"),
    Language.sinhala: lambda: SpaCyTokenizer("si"),
    Language.tatar: lambda: SpaCyTokenizer("tt"),
    Language.afrikaans: lambda: SpaCyTokenizer("af"),
    Language.kirghiz: lambda: SpaCyTokenizer("ky"),
    Language.irish: lambda: SpaCyTokenizer("ga"),
    Language.luxembourgish: lambda: SpaCyTokenizer("lb"),
    Language.maltese: lambda: StanzaTokenizer("mt"),
    Language.sanskrit: lambda: SpaCyTokenizer("sa"),
    Language.yoruba: lambda: SpaCyTokenizer("yo"),
    Language.serbocroatian: lambda: SpaCyTokenizer("sr"),
    Language.oriya: lambda: IndicNLPTokenizer("or"),
    Language.punjabi: lambda: IndicNLPTokenizer("pa"),
    Language.assamese: lambda: IndicNLPTokenizer("as"),
    Language.sindhi: lambda: IndicNLPTokenizer("sd"),
    Language.belarusian: lambda: StanzaTokenizer("be"),
    Language.tigrinya: lambda: SpaCyTokenizer("ti"),
    Language.uyghur: lambda: StanzaTokenizer("ug"),
    Language.tswana: lambda: SpaCyTokenizer("tn"),
    Language.wolof: lambda: StanzaTokenizer("wo"),
    Language.ganda: lambda: SpaCyTokenizer("lg"),
    Language.gaelic: lambda: StanzaTokenizer("gd"),
    Language.manx: lambda: StanzaTokenizer("gv"),
    Language.galician: lambda: StanzaTokenizer("gl"),
    Language.northern_sami: lambda: StanzaTokenizer("se"),
    Language.old_bulgarian: lambda: StanzaTokenizer("cu"),
    Language.faroese: lambda: SpaCyTokenizer("fo"),
    Language.norwegian_bokmal: lambda: SpaCyTokenizer("nb"),
    # proxies
    Language.bosnian: lambda: SpaCyTokenizer("hr"),  # Proxy
    Language.esperanto: lambda: SpaCyTokenizer("en"),  # Proxy
    Language.occitan: lambda: SpaCyTokenizer("ca"),  # Proxy
    Language.swahili: lambda: SpaCyTokenizer("en"),  # Proxy
    Language.javanese: lambda: SpaCyTokenizer("en"),  # Proxy
    Language.uzbek: lambda: SpaCyTokenizer("tr"),  # Proxy, alternative ru
    Language.tajik: lambda: SpaCyTokenizer("ru"),  # Proxy
    Language.kurdish: lambda: SpaCyTokenizer("en"),  # Proxy, multiple scripts!
    Language.bashkir: lambda: SpaCyTokenizer("tt"),  # Proxy
    Language.western_frisian: lambda: SpaCyTokenizer("nl"),  # Proxy
    Language.breton: lambda: StanzaTokenizer("cy"),  # Proxy
    Language.malagasy: lambda: SpaCyTokenizer("en"),  # Proxy
    Language.yiddish: lambda: SpaCyTokenizer("he"),  # Proxy
    Language.somali: lambda: SpaCyTokenizer("en"),  # Proxy
    Language.turkmen: lambda: SpaCyTokenizer("tr"),  # Proxy
    Language.pashto: lambda: SpaCyTokenizer("xx"),  # Proxy. xx is "multi-language"
}


@lru_cache
def get_word_tokenizer(language: Language) -> WordTokenizer:
    tokenizer = TOKENIZER_FACTORY.get(language)
    if tokenizer is None:
        hlog_warn(f"No word tokenizer found for language {language}, will split on spaces.")
        return WhitespaceTokenizer()
    return tokenizer()
