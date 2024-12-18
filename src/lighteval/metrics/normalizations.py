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

import re
import string
import sys
import unicodedata
from dataclasses import dataclass
from typing import Callable

from lighteval.metrics.utils.linguistic_tokenizers import get_word_tokenizer
from lighteval.utils.language import Language


UNICODE_TO_LATEX = {
    "α": "\\alpha",
    "β": "\\beta",
    "γ": "\\gamma",
    "δ": "\\delta",
    "ε": "\\epsilon",
    "ζ": "\\zeta",
    "η": "\\eta",
    "θ": "\\theta",
    "ι": "\\iota",
    "κ": "\\kappa",
    "λ": "\\lambda",
    "μ": "\\mu",
    "ν": "\\nu",
    "ξ": "\\xi",
    "π": "\\pi",
    "ρ": "\\rho",
    "σ": "\\sigma",
    "τ": "\\tau",
    "υ": "\\upsilon",
    "φ": "\\phi",
    "χ": "\\chi",
    "ψ": "\\psi",
    "ω": "\\omega",
    "Γ": "\\Gamma",
    "Δ": "\\Delta",
    "Θ": "\\Theta",
    "Λ": "\\Lambda",
    "Ξ": "\\Xi",
    "Π": "\\Pi",
    "Σ": "\\Sigma",
    "Υ": "\\Upsilon",
    "Φ": "\\Phi",
    "Ψ": "\\Psi",
    "Ω": "\\Omega",
    "∀": "\\forall",
    "∂": "\\partial",
    "∃": "\\exists",
    "∄": "\\nexists",
    "∅": "\\emptyset",
    "∇": "\\nabla",
    "∈": "\\in",
    "�����": "\\notin",
    "∋": "\\ni",
    "∏": "\\prod",
    "∑": "\\sum",
    "−": "-",
    "∗": "\\ast",
    "√": "\\sqrt",
    "∛": "\\sqrt[3]",
    "∝": "\\propto",
    "∞": "\\infty",
    "∠": "\\angle",
    "∧": "\\wedge",
    "∨": "\\vee",
    "∩": "\\cap",
    "∪": "\\cup",
    "∫": "\\int",
    "∬": "\\iint",
    "∭": "\\iiint",
    "∮": "\\oint",
    "∴": "\\therefore",
    "∵": "\\because",
    "∼": "\\sim",
    "≃": "\\simeq",
    "≅": "\\cong",
    "≈": "\\approx",
    "≠": "\\neq",
    "≡": "\\equiv",
    "≤": "\\leq",
    "≥": "\\geq",
    "⊂": "\\subset",
    "⊃": "\\supset",
    "⊄": "\\nsubset",
    "⊆": "\\subseteq",
    "⊇": "\\supseteq",
    "⊕": "\\oplus",
    "⊗": "\\otimes",
    "⊥": "\\bot",
    "⋅": "\\cdot",
    "⌈": "\\lceil",
    "⌉": "\\rceil",
    "⌊": "\\lfloor",
    "⌋": "\\rfloor",
    "⟂": "\\perp",
    "±": "\\pm",
    "∓": "\\mp",
    "×": "\\times",
    "÷": "\\div",
    "↑": "\\uparrow",
    "↓": "\\downarrow",
    "↔": "\\leftrightarrow",
    "→": "\\rightarrow",
    "←": "\\leftarrow",
    "⇒": "\\Rightarrow",
    "⇐": "\\Leftarrow",
    "⇔": "\\Leftrightarrow",
    "ℵ": "\\aleph",
    "ℏ": "\\hbar",
    "ℑ": "\\Im",
    "ℓ": "\\ell",
    "ℕ": "\\mathbb{N}",
    "ℙ": "\\mathbb{P}",
    "ℚ": "\\mathbb{Q}",
    "ℝ": "\\mathbb{R}",
    "ℤ": "\\mathbb{Z}",
    "ℂ": "\\mathbb{C}",
    "∘": "\\circ",
    "∙": "\\bullet",
    "∎": "\\blacksquare",
    "∐": "\\coprod",
    "∖": "\\setminus",
    "≊": "\\approxeq",
    "≪": "\\ll",
    "≫": "\\gg",
    "⊅": "\\nsupset",
    "⊈": "\\nsubseteq",
    "⊉": "\\nsupseteq",
    "⊊": "\\subsetneq",
    "⊋": "\\supsetneq",
    "⊖": "\\ominus",
    "⊘": "\\oslash",
    "⊙": "\\odot",
    "⊚": "\\circledcirc",
    "⊛": "\\circledast",
    "⊝": "\\circleddash",
    "⊞": "\\boxplus",
    "⊟": "\\boxminus",
    "⊠": "\\boxtimes",
    "⊡": "\\boxdot",
    "⊢": "\\vdash",
    "⊣": "\\dashv",
    "⊤": "\\top",
    "⊨": "\\models",
    "⊴": "\\unlhd",
    "⊵": "\\unrhd",
    "⋀": "\\bigwedge",
    "⋁": "\\bigvee",
    "⋂": "\\bigcap",
    "⋃": "\\bigcup",
    "⋆": "\\star",
    "⋈": "\\bowtie",
    "⌶": "\\APLsquish",
    "⌷": "\\APLquad",
    "⌸": "\\APL1",
    "⌹": "\\APLlefttack",
    "⌺": "\\APLrighttack",
    "⌻": "\\APLbottomcircle",
    "⌼": "\\APLtopunderscore",
    "⌽": "\\APLcircle",
    "⌿": "\\APLslash",
    "⍀": "\\APLbackslash",
    "⍼": "\\eqdef",
    "⎰": "\\ulcorner",
    "⎱": "\\urcorner",
    "⎲": "\\llcorner",
    "⎳": "\\lrcorner",
    "⏜": "\\frown",
    "⏝": "\\smile",
    "▷": "\\triangleright",
    "◁": "\\triangleleft",
    "◊": "\\diamond",
    "○": "\\circ",
    "●": "\\bullet",
    "★": "\\bigstar",
    "♠": "\\spadesuit",
    "♡": "\\heartsuit",
    "♢": "\\diamondsuit",
    "♣": "\\clubsuit",
    "♭": "\\flat",
    "♮": "\\natural",
    "♯": "\\sharp",
    "ℒ": "\\mathcal{L}",
    "℘": "\\wp",
    "ℬ": "\\mathcal{B}",
    "ℰ": "\\mathcal{E}",
    "ℱ": "\\mathcal{F}",
    "ℋ": "\\mathcal{H}",
    "ℐ": "\\mathcal{I}",
    "ℳ": "\\mathcal{M}",
    "ℛ": "\\mathcal{R}",
    "⟨": "\\langle",
    "⟩": "\\rangle",
    "⟦": "\\llbracket",
    "⟧": "\\rrbracket",
    "⟪": "\\llangle",
    "⟫": "\\rrangle",
    "⟬": "\\lBrace",
    "⟭": "\\rBrace",
    "⟵": "\\longleftarrow",
    "⟶": "\\longrightarrow",
    "⟷": "\\longleftrightarrow",
    "⟸": "\\Longleftarrow",
    "⟹": "\\Longrightarrow",
    "⟺": "\\Longleftrightarrow",
    "⟼": "\\longmapsto",
    "⟿": "\\leadsto",
    "⤂": "\\looparrowleft",
    "⤃": "\\looparrowright",
    "⤄": "\\leftrightharpoons",
    "⤅": "\\rightleftharpoons",
    "⤌": "\\curvearrowleft",
    "⤍": "\\curvearrowright",
    "⤎": "\\circlearrowleft",
    "⤏": "\\circlearrowright",
    "⤑": "\\rightsquigarrow",
    "⤒": "\\upuparrows",
    "⤓": "\\downdownarrows",
    "⤝": "\\leftarrowtail",
    "⤞": "\\rightarrowtail",
    "⥰": "\\downfishtail",
    "⥱": "\\upfishtail",
    "⦙": "\\Vert",
    "⦚": "\\lVert",
    "⦛": "\\rVert",
    "⦜": "\\lgroup",
    "⦝": "\\rgroup",
    "⦵": "\\bigodot",
    "⦶": "\\bigotimes",
    "⦷": "\\bigoplus",
    "⦸": "\\biguplus",
    "⦹": "\\bigsqcup",
    "⧀": "\\trianglelefteq",
    "⧁": "\\trianglerighteq",
    "⧂": "\\ntriangleleft",
    "⧃": "\\ntriangleright",
    "⧄": "\\trianglelefteqslant",
    "⧅": "\\trianglerighteqslant",
    "⧉": "\\square",
    "⧮": "\\downdownarrows",
    "⧯": "\\upuparrows",
    "⨀": "\\bigodot",
    "⨁": "\\bigoplus",
    "⨂": "\\bigotimes",
    "⨃": "\\bigcup",
    "⨄": "\\biguplus",
    "⨅": "\\bigcap",
    "⨆": "\\bigsqcup",
    "⨏": "\\fint",
    "⨑": "\\iint",
    "⨒": "\\iiint",
    "⨓": "\\iiiint",
    "⨔": "\\idotsint",
    "⨕": "\\oiint",
    "⨖": "\\oiiint",
    "⨗": "\\ointctrclockwise",
    "⨘": "\\ointclockwise",
    "⨙": "\\sqint",
    "⨚": "\\intlarhk",
    "⨛": "\\intx",
    "⨜": "\\intcap",
    "⨝": "\\intcup",
    "⨞": "\\upint",
    "⨟": "\\lowint",
    "ℶ": "\\beth",
    "ℷ": "\\gimel",
    "ℸ": "\\daleth",
    "ℼ": "\\varpi",
    "ℽ": "\\digamma",
    "ℾ": "\\Gamma",
    "ℿ": "\\Pi",
    "⅀": "\\Sigma",
    "⅁": "\\Game",
    "⅂": "\\turnediota",
    "⅃": "\\Finv",
    "⅄": "\\Yup",
    "⅋": "\\multimap",
    "ↀ": "\\mho",
    "ↁ": "\\eth",
    "ↂ": "\\diagup",
    "↽": "\\leftharpoondown",
    "↾": "\\upharpoonright",
    "↿": "\\upharpoonleft",
    "⇀": "\\rightharpoonup",
    "⇁": "\\rightharpoondown",
    "⇂": "\\downharpoonright",
    "⇃": "\\downharpoonleft",
    "⇄": "\\rightleftarrows",
    "⇅": "\\updownarrows",
    "⇆": "\\leftrightarrows",
    "⇇": "\\leftleftarrows",
    "⇈": "\\upuparrows",
    "⇉": "\\rightrightarrows",
    "⇊": "\\downdownarrows",
    "⇋": "\\leftrightharpoons",
    "⇌": "\\rightleftharpoons",
    "⇍": "\\nLeftarrow",
    "⇎": "\\nRightarrow",
    "⇏": "\\nLeftrightarrow",
    "⇑": "\\Uparrow",
    "⇓": "\\Downarrow",
    "⇕": "\\Updownarrow",
    "⇖": "\\nwarrow",
    "⇗": "\\nearrow",
    "⇘": "\\searrow",
    "⇙": "\\swarrow",
    "⇚": "\\Lleftarrow",
    "⇛": "\\Rrightarrow",
    "⇝": "\\rightsquigarrow",
    "⇞": "\\upuparrows",
    "⇟": "\\downdownarrows",
    "⇠": "\\leftarrowtail",
    "⇡": "\\uparrowtail",
    "⇢": "\\rightarrowtail",
    "⇣": "\\downarrowtail",
    "⇤": "\\mapsfrom",
    "⇥": "\\mapsto",
    "⇵": "\\updownarrowbar",
    "⇶": "\\hookuparrow",
    "⇷": "\\hookdownarrow",
    "⇸": "\\looparrowleft",
    "⇹": "\\looparrowright",
    "⇺": "\\leftrightsquigarrow",
    "⇻": "\\nleftarrow",
    "⇼": "\\nrightarrow",
    "⇽": "\\leftharpoonup",
    "⇾": "\\rightharpoonup",
    "⇿": "\\leftrightharpoons",
    "∁": "\\complement",
    "∆": "\\triangle",
    "∊": "\\in",
    "∌": "\\notni",
    "∍": "\\ni",
    "∔": "\\dotplus",
    "∕": "\\slash",
    "∜": "\\sqrt[4]",
    "∟": "\\angle",
    "∡": "\\measuredangle",
    "∢": "\\sphericalangle",
    "∣": "\\mid",
    "∤": "\\nmid",
    "∥": "\\parallel",
    "∦": "\\nparallel",
    "∯": "\\oiint",
    "∰": "\\oiiint",
    "∱": "\\intclockwise",
    "∲": "\\varointclockwise",
    "∳": "\\ointctrclockwise",
    "∶": ":",
    "∷": "::",
    "∸": "\\dotminus",
    "∹": "\\excess",
    "∺": "\\geomequiv",
    "∻": "\\homothetic",
    "∽": "\\backsim",
    "∾": "\\lazysinv",
    "∿": "\\sinewave",
    "≀": "\\wr",
    "≁": "\\nsim",
    "≂": "\\eqsim",
    "≄": "\\nsimeq",
    "≆": "\\simneqq",
    "≇": "\\ncong",
    "≉": "\\napprox",
    "≋": "\\approxident",
    "≌": "\\backcong",
    "≍": "\\asymp",
    "≎": "\\Bumpeq",
    "≏": "\\bumpeq",
    "≐": "\\doteq",
    "≑": "\\Doteq",
    "≒": "\\fallingdotseq",
    "≓": "\\risingdotseq",
    "≔": ":=",
    "≕": "=:",
    "≖": "\\eqcirc",
    "≗": "\\circeq",
    "≙": "\\corresponds",
    "≚": "\\triangleq",
    "≛": "\\triangleq",
    "≜": "\\def",
    "≝": "\\stackrel{\\mathrm{def}}{=}",
    "≞": "\\questeq",
    "≟": "\\eqsim",
    "≢": "\\nequiv",
    "≣": "\\Equiv",
    "≦": "\\leqq",
    "≧": "\\geqq",
    "≨": "\\lneqq",
    "≩": "\\gneqq",
    "≬": "\\between",
    "≭": "\\nasymp",
    "≮": "\\nless",
    "≯": "\\ngtr",
    "≰": "\\nleq",
    "≱": "\\ngeq",
    "≲": "\\lesssim",
    "≳": "\\gtrsim",
    "≴": "\\nlesssim",
    "≵": "\\ngtrsim",
    "≶": "\\lessgtr",
    "≷": "\\gtrless",
    "≸": "\\notlessgreater",
    "≹": "\\notgreaterless",
    "≺": "\\prec",
    "≻": "\\succ",
    "≼": "\\preceq",
    "≽": "\\succeq",
    "≾": "\\preccurlyeq",
    "≿": "\\succcurlyeq",
    "⊀": "\\nprec",
    "⊁": "\\nsucc",
    "⋉": "\\ltimes",
    "⋊": "\\rtimes",
    "��": "\\leftthreetimes",
    "⋌": "\\rightthreetimes",
    "⋍": "\\backsimeq",
    "⋎": "\\curlyvee",
    "⋏": "\\curlywedge",
    "⋐": "\\Subset",
    "⋑": "\\Supset",
    "⋒": "\\Cap",
    "⋓": "\\Cup",
    "⋔": "\\pitchfork",
    "⋕": "\\equalparallel",
    "⋖": "\\lessdot",
    "⋗": "\\gtrdot",
    "⋘": "\\lll",
    "⋙": "\\ggg",
    "⋚": "\\lesseqgtr",
    "⋛": "\\gtreqless",
    "⋜": "\\eqslantless",
    "⋝": "\\eqslantgtr",
    "⋞": "\\curlyeqprec",
    "⋟": "\\curlyeqsucc",
    "⋠": "\\npreccurlyeq",
    "⋡": "\\nsucccurlyeq",
    "⋢": "\\nsqsubseteq",
    "⋣": "\\nsqsupseteq",
    "⋦": "\\lnsim",
    "⋧": "\\gnsim",
    "⋨": "\\precnsim",
    "⋩": "\\succnsim",
    "⋪": "\\ntriangleleft",
    "⋫": "\\ntriangleright",
    "⋬": "\\ntrianglelefteq",
    "⋭": "\\ntrianglerighteq",
    "⋮": "\\vdots",
    "⋯": "\\cdots",
    "⋰": "\\adots",
    "⋱": "\\ddots",
    "⋲": "\\disin",
    "⋳": "\\varisins",
    "⋴": "\\isins",
    "⋵": "\\isindot",
    "⋶": "\\varisinobar",
    "⋷": "\\isinobar",
    "⋸": "\\isinvb",
    "⋹": "\\isinE",
    "⋺": "\\nisd",
    "⋻": "\\varnis",
    "⋼": "\\nis",
    "⋽": "\\varniobar",
    "⋾": "\\niobar",
    "⋿": "\\bagmember",
    "⌀": "\\diameter",
    "⌁": "\\house",
    "⌂": "\\varhouse",
    "⌃": "\\upanglearrow",
    "⌄": "\\downanglearrow",
    "⌅": "\\uparrowbarred",
    "⌆": "\\downarrowbarred",
    "⌇": "\\varmapsfrom",
    "⌌": "\\invneg",
    "⌍": "\\wasylozenge",
    "⌎": "\\ocircle",
    "⌏": "\\oturnedcircle",
    "⌐": "\\neg",
    "⌑": "\\boxquestion",
    "⌒": "\\overset{\\frown}{}",
    "⌓": "\\trianglecdot",
    "⌔": "\\triangledown",
    "⌕": "\\boxadd",
    "⌖": "\\diamondmath",
    "⌗": "\\boxbar",
    "⌘": "\\Command",
    "⌙": "\\triangleminus",
    "⌚": "\\watch",
    "⌛": "\\hourglass",
    "⌜": "\\ulcorner",
    "⌝": "\\urcorner",
    "⌞": "\\llcorner",
    "⌟": "\\lrcorner",
    "⌠": "\\intop",
    "⌡": "\\intbot",
    "⌢": "\\frown",
    "⌣": "\\smile",
    "⌤": "\\varhexagonblack",
    "⌥": "\\Option",
    "⌦": "\\deleq",
    "���": "\\squares",
    "⌨": "\\keyboard",
    "⌫": "\\leftdelete",
    "⌬": "\\upslice",
    "⌭": "\\downslice",
    "⌮": "\\logof",
    "⌯": "\\recordright",
    "⌰": "\\APLinput",
    "⌱": "\\APLbox",
    "⌲": "\\APLcomment",
    "⌳": "\\APLupstile",
    "⌴": "\\APLdownstile",
    "⌵": "\\APLinput",
    "⌾": "\\APLcirclestile",
    "⍁": "\\APLdowncaret",
    "⍂": "\\APLupcaret",
    "⍃": "\\APLleftcaret",
    "⍄": "\\APLrightcaret",
    "⍅": "\\APLrightshoe",
    "⍆": "\\APLleftshoe",
    "⍇": "\\APLtilde",
    "⍈": "\\APLhighcircle",
    "⍉": "\\APLlowcircle",
    "⍊": "\\APLrightfloor",
    "⍋": "\\APLupshoe",
    "⍌": "\\APLdownshoe",
    "⍍": "\\APLleftshoe",
    "⍎": "\\APLzilde",
    "⍏": "\\APLsquiggleright",
    "⍐": "\\APLsquiggleleft",
    "⍑": "\\APLboxupcaret",
    "⍒": "\\APLboxdowncaret",
    "⍓": "\\APLupstile",
    "⍔": "\\APLdownstile",
    "⍕": "\\APLboxquestion",
    "⍖": "\\APLboxexclamation",
    "⍗": "\\APLboxbar",
    "⍘": "\\APLrighttack",
    "⍙": "\\APLlefttack",
    "⍚": "\\APLdowntackjot",
    "⍛": "\\APLuptackjot",
    "⍜": "\\APLsquiggledelete",
    "⍝": "\\APLsquiggleselect",
    "⍞": "\\APLquadcolon",
    "⍟": "\\APLstar",
    "⍠": "\\APLdiamond",
    "⍡": "\\APLequal",
    "⍢": "\\APLquad",
    "⍣": "\\APLuparrow",
    "⍤": "\\APLdownarrow",
    "⍥": "\\APLupstile",
    "⍦": "\\APLdownstile",
    "⍧": "\\APLsquare",
    "⍨": "\\APLnotequal",
    "⍩": "\\APLcolon",
    "⍪": "\\APLhook",
    "⍫": "\\APLquadequal",
    "⍬": "\\APLnotless",
    "⍭": "\\APLnotgreater",
    "⍮": "\\APLnotequal",
    "⍯": "\\APLcirclestar",
    "⍰": "\\APLdowncaret",
    "⍱": "\\APLupcaret",
    "⍲": "\\APLleftcaret",
    "⍳": "\\APLrightcaret",
    "⍴": "\\APLleftfloor",
    "⍵": "\\APLrightfloor",
    "⍶": "\\APLleftceil",
    "⍷": "\\APLrightceil",
    "⍸": "\\APLbottomcircle",
    "⍹": "\\APLtopunderscore",
    "⍺": "\\APLrightshoe",
    "⎀": "\\APLcomment",
    "⎁": "\\APLboxcomment",
    "⎂": "\\APLminus",
    "⎃": "\\APLcirc",
    "⎄": "\\APLtilde",
    "⎅": "\\APLdiaeresis",
    "⎆": "\\APLoverbar",
    "⎇": "\\APLzilde",
    "⎈": "\\APLquadcircle",
    "⎉": "\\APLdownstile",
}


UNICODE_TO_LATEX_REG = re.compile("|".join(map(re.escape, UNICODE_TO_LATEX.keys())))


def unicode_to_latex(text: str) -> str:
    return UNICODE_TO_LATEX_REG.sub(lambda m: UNICODE_TO_LATEX[m.group(0)], text)


# From HELM
def helm_normalizer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def homogeneize_numbers(text: str) -> str:
        """Casts text to float to test if it's a number, then casts back to string.
        This allows equal numbers formatted differently (1.0 vs 1 for ex) to be considered
        equal. This comes from Harness DROP - check if it causes a discrep in QuAC
        """
        try:
            return str(float(text))
        except ValueError:
            return text

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    def _tokenize(text):
        return re.split(" |-", text)

    tokens = [white_space_fix(remove_articles(homogeneize_numbers(remove_punc(lower(t))))) for t in _tokenize(text)]
    return " ".join([t for t in tokens if t != ""]).strip()


def harness_triviaqa_normalizer(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def bigbench_normalizer(text: str):
    return text.replace(" . ", ".\n")


def remove_braces(text: str) -> str:
    if text.startswith("{"):
        text = text[1:]
    if text.endswith("}"):
        text = text[:-1]
    return text


def remove_braces_and_strip(text: str) -> str:
    text = text.strip()
    if text.startswith("{"):
        text = text[1:]
    if text.endswith("}"):
        text = text[:-1]
    return text


units = [
    "integer" "point",
    "feet",
    "sue",
    "digit",
    "pound",
    "meal",
    "edge",
    "student",
    "children ticket",
    "multiple",
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m square",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "cent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

# We sort here to that when matching from right the longest units are matched first
# E.g "percent" is matched before "cent"

units_regex = re.compile("|".join([f"(^|\\W)(?:{unit}(?:s|es)?)($|\\W)" for unit in units]))

to_remove_regex = re.compile(
    r"\\mathrm\{th\}|"
    r"\\;|"
    r"\{,\}|"
    r"\\!|"  # inverse spaces (already present in original)
    r"\\left|\\right|"  # \left and \right (already present in original)
    r"\^{\\circ}|\^\\circ|"  # degrees symbols (already present in original)
    r"\\\$|\$|"  # dollar signs
    r",\\!|"  # comma with inverse space (already present in original)
    r"\{,\}|"  # braced comma (already present in original)
    r"(?<=\s)(and|an|a)(?=\s)|"  # "an" with whitespace
    r"\\\s|"  # backslash with whitespace
    # Percentage symbol
    r"\\\%|\%|%|"  # percentage symbol
    r"[xyzk](?:=|\\in|\\to)|"
    # Quote
    r'"|\''
)

to_replace_patterns = [
    # (name, pattern, replacement)
    ("frac", r"\\tfrac", r"\frac"),
    ("cfrac", r"\\cfrac", r"\frac"),
    ("dfrac", r"\\dfrac", r"\frac"),
    ("array", r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}"),
    ("array_end", r"\\end\{array\}", r"\\end{pmatrix}"),
    ("bmatrix", r"bmatrix", r"pmatrix"),
    ("textbf", r"\\textbf", r"\text"),
    ("mbox", r"\\mbox", r"\text"),
    ("decimal_space", r"\s\.", r" 0."),
    ("decimal_brace", r"\{\.", r"{0."),
    ("neq", r"\\neq", r"\ne"),
    ("leq", r"\\leq", r"\le"),
    ("geq", r"\\geq", r"\ge"),
    ("brace_open", r"\\\{", r"{"),
    ("brace_close", r"\\\}", r"}"),
    ("paren_open", r"\\\(", r"("),
    ("paren_close", r"\\\)", r")"),
    ("emptyset", r"\\emptyset", r"{}"),
    ("real_line", r"\(-\\infty,\\infty\)", r"\mathbb{R}"),
    ("infinity", r"infinity", r"\infty"),
    ("inf", r"((?<!\\)inf(?!inity))", r"\infty"),
]

# Create regex with named groups
pattern = "|".join(f"(?P<{name}>{pattern})" for name, pattern, _ in to_replace_patterns)
to_replace_regex = re.compile(pattern)

# Create lookup dictionary for replacements
replacements = {name: replacement for name, _, replacement in to_replace_patterns}


def replace(match):
    # Find which group matched
    # Get corresponding replacement from dict
    return replacements[match.lastgroup]


def replace_in_latex(text: str) -> str:
    return to_replace_regex.sub(replace, text)


def extract_last_boxed_content(text: str) -> str:
    """
    Find and extract the content of the last \\boxed{...} or \\fbox{...} element from a string.

    Example:
    >>> extract_last_boxed_content("Some text \\boxed{\\frac{2}{3}}")
    "\\frac{2}{3}"
    >>> extract_last_boxed_content("\\boxed 123")
    "123"
    >>> extract_last_boxed_content("No box here")
    ""
    """

    # Then look for \\boxed{...} or \\fbox{...}
    env = "\\boxed"
    left_idx = text.rfind(env)
    if left_idx < 0:
        env = "\\fbox"
        left_idx = text.rfind(env)
        if left_idx < 0:
            return text
    left_idx += len(env)

    # If the next character is a brace remove it, otherwise it's a \\boxed {content}
    if len(text) > left_idx and text[left_idx] != "{":
        # If there is no opening brace, it's a \\boxed {content}
        return text[left_idx:].lstrip()

    # Find matching closing brace
    i = left_idx
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                # Extract content between braces (+1 to remove the opening brace)
                return text[left_idx+1:i]
        i += 1

    # Otherwise, it's no a valid latex
    return text


def math_normalizer(text: str, skip_unit: bool = False) -> str:  # noqa C901
    """Source: https://github.com/hendrycks/math"""

    def _fix_fracs(text: str) -> str:
        """
        Fix the formatting of fractions in the given text.
        Copied from: https://github.com/hendrycks/math/blob/357963a7f5501a6c1708cf3f3fb0cdf525642761/modeling/math_equivalence.py#L1

        Args:
            text (str): The input text.

        Returns:
            str: The text with properly formatted fractions.

        Examples:
            >>> _fix_fracs("\\frac12")
            "\\frac{1}{2}"
            >>> _fix_fracs("\\frac{3}{4}")
            "\\frac{3}{4}"
            >>> _fix_fracs("\\frac1{2}")
            "\\frac{1}{2}"
        """

        substrs = text.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return text
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        text = new_str
        return text

    def _fix_a_slash_b(text: str) -> str:
        """Source: https://github.com/hendrycks/math
        Reformat fractions formatted as a/b to \\frac{a}{b}.
        Example:
        >>> _fix_a_slash_b("2/3")
        \frac{2}{3}
        """
        if len(text.split("/")) != 2:
            return text
        a_str = text.split("/")[0]
        b_str = text.split("/")[1]
        try:
            a = int(a_str)
            b = int(b_str)
            assert text == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except Exception:
            return text

    def _fix_sqrt(text: str) -> str:
        """Source: https://github.com/hendrycks/math
        Reformat square roots.
        Example:
        >>> _fix_sqrt("\\sqrt3")
        \\sqrt{3}
        """
        if "\\sqrt" not in text:
            return text
        splits = text.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    def _remove_text_formatting(text: str) -> str:
        """Remove text formatting commands like \text{}, \textbf{}, \\overline{}, and \boxed{}.
        Also ensures math expressions are properly wrapped in single $ signs.

        Args:
            text (str): The text to process

        Returns:
            str: Text with formatting commands removed and math properly delimited

        Examples:
            - Input: 'outer $\\text{inner}$ text'
            Output: 'outer $inner$ text'
            - Input: '$\\textbf{bold math}$'
            Output: '$bold math$'

            - Input: '$\\overline{x + y}$'
            Output: '$x + y$'
        """
        text = re.sub(r"(\\text\{)(.*?)(\})", "\\2", text)  # remove \text{...}
        text = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", text)  # remove \overline{...}
        return text

    # First extract the last boxed content
    text = extract_last_boxed_content(text)

    # Sometimes the \\ are doubled so we substitute them
    text = text.replace("\\\\", "\\")

    # Replace the unigrams
    text = unicode_to_latex(text)

    # Remoove useless latex commands
    text = to_remove_regex.sub("", text)

    text = replace_in_latex(text)

    # Remove the units and possibly the superscript (for things like m^2)
    _text = re.sub(r"\\text{.*?}(^{\d})?$", "", text).strip()
    if _text != "" and _text != text:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        text = _text

    if not skip_unit:
        # Remove unit: texts, we do thiss twice too remove stuff like meter square
        for _ in range(2):
            _text = units_regex.sub(r"\1\2", text)
    # Remove all text formatting
    text = _remove_text_formatting(text)

    if text[0] == ".":
        text = "0" + text

    # fix sqrt3 --> sqrt{3}
    text = _fix_sqrt(text)

    # remove spaces
    text = text.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    text = _fix_fracs(text)

    # manually change 0.5 --> \frac{1}{2}
    if text == "0.5":
        text = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    text = _fix_a_slash_b(text)

    return text.strip()


def gsm8k_normalizer(text: str) -> str:
    """
    from https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28

    Args:
        text (str): input text

    Returns:
        str: Output text, either the number found in the text or "[invalid]" if
        no number was found
    """
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"

    match = ANS_RE.search(text)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


PUNCT = {chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")}.union(
    string.punctuation
)

_ARTICLE_PATTERNS = {
    Language.ENGLISH: r"\b(a|an|the)\b",
    Language.SPANISH: r"\b(el|la|los|las|un|una|unos|unas)\b",
    Language.PORTUGUESE: r"\b(o|a|os|as|um|uma|uns|umas)\b",
    Language.ITALIAN: r"\b(il|lo|la|i|gli|le|un|uno|una)\b",
    Language.FRENCH: r"\b(le|la|les|l'|un|une|des)\b",
    Language.GERMAN: r"\b(der|die|das|den|dem|des|ein|eine|einer|eines|einem|einen)\b",
    Language.FINNISH: r"\b(se|yksi|yks)\b",
    Language.GREEK: r"\b(ὁ|οἱ|τοῦ|τῶν|τόν|τούς|ὦ|ἡ|αἱ|τῆς|τῶν|τήν|τάς|τό|τά|τοῦ|τῶν|τό|τά)\b",
    Language.NORWEGIAN: r"\b(en|ei|et|den|det|de)\b",
    Language.SWEDISH: r"\b(en|ett|den|det|de)\b",
    Language.TURKISH: r"\b(bir)\b",
    Language.DUTCH: r"\b(de|het|een)\b",
    Language.HUNGARIAN: r"\b(a|az|egy)\b",
    Language.CATALAN: r"\b(el|la|els|les|un|una|uns|unes)\b",
    Language.HEBREW: r"\b(ה)\b",
    Language.GALICIAN: r"\b(o|a|os|as|un|unha|uns|unhas)\b",
}


def remove_articles(text: str, lang: Language) -> str:
    """
    Removes definite and indefinite articles from the text.
    Generated using LLM then manually checked by non-expert.
    We currently only support languages that don't blend articles.
    If you are a native speaker of a language where articles are blended,
    we would appreciate your contribution!
    """
    pattern = _ARTICLE_PATTERNS.get(lang)
    return re.sub(pattern, " ", text) if pattern else text


def remove_punc(text: str) -> str:
    return "".join(ch for ch in text if ch not in PUNCT)


def get_multilingual_normalizer(lang: Language, lower: bool = True) -> Callable[[str], str]:
    tokenizer = get_word_tokenizer(lang)

    def _inner_normalizer(text: str) -> str:
        text = remove_articles(text, lang)
        text = remove_punc(text)
        if lower:
            text = text.lower()

        tokens = tokenizer.word_tokenize(text)
        return " ".join(tokens)

    return _inner_normalizer


# Loglikelihood normalization
@dataclass
class LogProbPMINorm:
    """
    Performs Pointwise mutual information normalization. log_likelihood_conditioned - log_likelihood_unconditioned.
    Useful when answer contains generally unlikely tokens.
    """

    name: str = "norm_pmi"

    pass


@dataclass
class LogProbTokenNorm:
    """
    Performs token level normalization. log_likelihood/token_length.
    Useful for non-english languages.
    """

    name: str = "norm_token"
    pass


@dataclass
class LogProbCharNorm:
    """
    Performs character level normalization. log_likelihood/char_length
    ignore_first_space (bool, optional): Whether to ignore the first token's log prob (if it's a space only). Defaults to False.
        The only case when it should be True is when the possible choices (for example `A`,`B` ...) have an extra
        space added in front of them to manage tokenization issues (` A`, ` B`, ...) for some models.
    """

    name: str = "norm"

    ignore_first_space: bool = False


LogProbNormalization = LogProbCharNorm | LogProbTokenNorm | LogProbPMINorm


def normalize_log_probs(
    normalization: LogProbNormalization,
    choices_logprob: list[float],
    unconditioned_logprob: list[float] | None,
    choices_text: list[str] | None,
    choices_tokens: list[list[int]] | None,
) -> list[float]:
    normalized_log_probs = choices_logprob
    match normalization:
        case LogProbCharNorm(ignore_first_space=True):
            assert choices_text is not None, "choices_text must be provided for character normalization"
            normalized_log_probs = [
                choices_logprob[ix] / (len(choice) - 1 if choice[0] == " " else len(choice))
                for ix, choice in enumerate(choices_text)
            ]
        case LogProbCharNorm(ignore_first_space=False):
            assert choices_text is not None, "choices_text must be provided for character normalization"
            normalized_log_probs = [choices_logprob[ix] / len(choice) for ix, choice in enumerate(choices_text)]
        case LogProbTokenNorm():
            assert choices_tokens is not None, "choices_tokens must be provided for token normalization"
            normalized_log_probs = [
                choices_logprob[ix] / len(choices_tokens[ix]) for ix in range(len(choices_logprob))
            ]
        case LogProbPMINorm():
            assert unconditioned_logprob is not None, "unconditioned_logprob must be provided for PMI normalization"
            normalized_log_probs = [
                choices_logprob[ix] - unconditioned_logprob[ix] for ix in range(len(choices_logprob))
            ]

    return normalized_log_probs
