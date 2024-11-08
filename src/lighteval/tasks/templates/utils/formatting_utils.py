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

from lighteval.tasks.templates.utils.translation_literals import TranslationLiterals


# Contains punctuation covering most of the languages big chunk took from https://stackoverflow.com/questions/9506869/are-there-character-collections-for-all-international-full-stop-punctuations
PUNCT = "᪩？⁈𑩂．꩞𑅃﹗𑂾\u1b7d፧𑅂꡶꘎⁉࠾᪨𑊩𑱂᱿𖩮᥅\U00011f43\U00011f44﹒𑈹𑈸።܂؞꛳\U00010f88𑗍𐩖𑙂\u061d꩟᠉\u1b7e𑗗᰼𑻸؟𑪜꧉𑗉𐽙𖫵𖬷܀꓿᜵𑗏𑁇𑗓𑥄៖𑥆𑗑𑗒꯫'۔𐩗\U00010f86꡷\u2e54｡៕߹⸮.𑇅࠹𛲟꫰ល꤯𐽗᭞𑜼፨𑃁꣏𑇟𖬸𑪛𑜾࠷𝪈?𑃀𑗃！։꣎॥𑗖᭛᠃!၊𖺘⁇𑗌𑑋𖭄᭟\"𑅁𑙁⸼꩝𑗋。꧈꫱𑜽𐽖𑂿᙮។꛷\U00010f89៚᥄𑗕𑗎᪪᭚࠽𑇞𑗊𐽘\u2e53𑗔𖩯𑇍𑻷𐽕𑩃।𑗂𑇆𑁈။᱾𑱁꘏܁᜶‼𑈻‽᪫﹖𑑌𑈼\U00010f87𑗐៙᰻"


def decapitalize(word: str):
    """
    Decapitalize the first letter of the string
    """
    if len(word) == 0:
        return word
    return word[0].lower() + word[1:]


def capitalize(word: str):
    """
    Capitalize the first letter of the string
    """
    if len(word) == 0:
        return word
    return word[0].upper() + word[1:]


def fix_ending_punct(ctx: str, translation_literals: TranslationLiterals):
    """
    Fixes the ending punctuation so that it uses the correct punctuation for the language.
    E.g in Chinese the "?" is written as "？"
    """
    ctx = ctx.rstrip()
    if len(ctx) == 0:
        return ctx
    if ctx.endswith("?"):
        ctx = ctx[:-1] + translation_literals.question_mark
    elif ctx.endswith("."):
        ctx = ctx[:-1] + translation_literals.full_stop
    elif ctx.endswith(","):
        ctx = ctx[:-1] + translation_literals.comma
    elif ctx.endswith(":"):
        ctx = ctx[:-1] + translation_literals.colon
    return ctx


def punctuation_ends_sentence(text: str, translation_literals: TranslationLiterals):
    """
    Check if the string ends with a sentence-ending punctuation mark.
    That's .?!:
    It's not perfect method as some languages don't have full stops etc..
    """
    return text.rstrip().endswith(
        (
            translation_literals.question_mark,
            translation_literals.full_stop,
            translation_literals.exclamation_mark,
            translation_literals.colon,
            translation_literals.semicolon,
        )
    )


def fix_capitalization(prefix: str, text: str, translation_literals: TranslationLiterals):
    """
    Fixes the capitalization of the text based on the prefix.
    It's based on simple heuristics:
    - If the prefix ends with a sentence-ending punctuation mark, the text should be capitalized.
    - If the prefix ends with a newline, the text should be capitalized.
    """
    if len(prefix) == 0:
        return capitalize(text)

    if prefix.endswith("\n"):
        return capitalize(text)

    return capitalize(text) if punctuation_ends_sentence(prefix, translation_literals) else decapitalize(text)
