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


import pytest

from lighteval.metrics.utils.linguistic_tokenizers import TOKENIZER_FACTORY, SpaCyTokenizer, get_word_tokenizer
from lighteval.utils.imports import Extra, is_package_available
from lighteval.utils.language import Language


pytestmark = pytest.mark.skipif(not is_package_available(Extra.MULTILINGUAL), reason="requires the multilingual extra")


def test_spacy_tokenizers_use_language_codes_known_to_spacy():
    import spacy

    for language, factory in TOKENIZER_FACTORY.items():
        tokenizer = factory()
        if isinstance(tokenizer, SpaCyTokenizer):
            # Raises ImportError for a code spaCy does not ship, e.g. "cz" instead of "cs" for Czech
            spacy.util.get_lang_class(tokenizer.spacy_language)


def test_czech_word_tokenizer():
    tokenizer = get_word_tokenizer(Language.CZECH)
    assert tokenizer.word_tokenize("Dobrý den, jak se máš?") == ["Dobrý", "den", ",", "jak", "se", "máš", "?"]
