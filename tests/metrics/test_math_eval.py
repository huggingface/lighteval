import pytest
import sympy

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
    sympy_expr_eq,
)
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


def compare_en(
    gold: str,
    pred: str,
    language: Language = Language.ENGLISH,
    match_types: list[str] = ["latex", "expr"],
    precision: int = 6,
):
    # Convert string match_types to ExtractionTarget objects
    extraction_targets = []
    for match_type in match_types:
        if match_type == "latex":
            extraction_targets.append(LatexExtractionConfig())
        elif match_type == "expr":
            extraction_targets.append(ExprExtractionConfig())
        elif match_type == "NativeLetters":
            extraction_targets.append(IndicesExtractionConfig(prefix_for_extraction="NativeLetters"))

    extraction_targets = tuple(extraction_targets)  # Convert to tuple

    return multilingual_extractive_match_metric(
        language=language,
        gold_extraction_target=extraction_targets,
        pred_extraction_target=extraction_targets,
        precision=precision,
    ).sample_level_fn(
        golds=[gold],
        predictions=[pred],
        formatted_doc=Doc(choices=["", "", "", ""], query="", gold_index=0),
    )


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        ("C", "thinking about I think the correct answer is C", 1),
        ("B", "Let's think step by step. It's not A because it doesn't make sense, therefore I think it's B", 1),
        ("D", "The answer is for sure D, it can't be A or B", 1),
        ("D", "The answer: D, doesn't makese nsense for answer to be A or B", 1),
        ("D", "D. it can't be A or B", 1),
    ],
)
def test_extraction_abc(gold, pred, expected):
    assert compare_en(gold, pred, match_types=["NativeLetters"]) == expected


@pytest.mark.parametrize(
    "gold,pred,language,expected",
    [
        ("C", "réponse est C non A", Language.FRENCH, 1),
        ("B", "B。 不是 A", Language.CHINESE, 1),
        ("B", "B。不是 A", Language.CHINESE, 1),
        ("B", "B不是 A", Language.CHINESE, 1),
    ],
)
def test_multilingual_extraction_abc(gold, pred, language, expected):
    assert compare_en(gold, pred, language, match_types=["NativeLetters"]) == expected


@pytest.mark.parametrize(
    "gold,pred,language,match_type,expected",
    [
        ("105", "réponse est (35 + 70 = 105).", Language.FRENCH, ["latex", "expr"], True),
        ("79", "donc 353 g - 79 g = 274 g. Donc, il a déjà 79 g de cire.", Language.FRENCH, ["latex", "expr"], True),
        ("220", "Réponse: Janeth aura encore 220 $ à payer d'ici 12 mois.", Language.FRENCH, ["latex", "expr"], True),
        (
            "2/5",
            "} \\times \\frac{1}{3} = \\frac{6}{15} = \\frac{2}{5} ] 所以，每份应该是 (\\frac{2}{5}) 吨。 答案：每份应该是 (\\frac{2}{5}) 吨。",
            Language.CHINESE,
            ["latex", "expr"],
            True,
        ),
        ("4000", " 地块面积 = 72000 / 18 = 4000千克", Language.CHINESE, ["latex", "expr"], True),
        (
            "300",
            "来计算水池中水的流出时间：12000升/40升/分钟=300分钟。因此，水池中水将在300分钟内被放完。",
            Language.CHINESE,
            ["latex", "expr"],
            True,
        ),
        ("13/28", "计划的比例为13/28", Language.CHINESE, ["latex", "expr"], True),
        ("8/46", "\\frac{4}{23}", Language.CHINESE, ["latex", "expr"], True),
        ("\\frac{9.5}{3.14159}", "\\frac{9.5}{3.14159} \\approx 3.01", Language.CHINESE, ["latex", "expr"], True),
        (
            "1314",
            "ا��باقي: 4 ÷ 3 = 1 بباقي 1 نكتب 1 فوق الخط ونضع الباقي 1 تحت الرقم الرابع. 6. نجمع الأرقام فوق الخط: 438 7. نتحقق من النتيجة: 438 × 3 = 1314 لذا، فإن ناتج 1314 ÷ 3 هو 438. الباقي من القسمة هو 0، مما يعني أن 1314 قابل للقسمة على 3 تمامًا.",
            Language.ARABIC,
            ["latex", "expr"],
            True,
        ),
        (
            "67",
            " ा गणना**: दुकान में शुरूआत में 56 कमीजें थीं। 2. जोड़ने वाली संख्या गणना: बाद में 11 और कमीजें मिलीं। 3. कुल संख्या गणना: मूल संख्या और जोड़ी गई संख्या को जोड़ने पर दुकान में अब कितनी कमीजें हैं ज्ञात कर सकते हैं। इसलिए, गणना करें: [ 56 + 11 = 67 ] इसलिए, दुकान में अब 67 कमीजें हैं। ",
            Language.HINDI,
            ["latex", "expr"],
            True,
        ),
        ("0", "So the os then when we 9/3 we get 8 so the answer is 0", Language.ENGLISH, ["latex", "expr"], True),
    ],
)
def test_multilingual_extraction_math(gold, pred, language, match_type, expected):
    assert compare_en(gold, pred, language, match_type) == expected


def test_multilingual_extraction_math_latex_numbers():
    assert compare_en("1", "so $x+y = 1000$ therefore answer is $1$", Language.FRENCH)

    # Ensure latex has precedence over numbers
    assert compare_en("1000", "so answer is $x+y = 1000$ not 1", Language.FRENCH)
    # If latex fails, numbers are extracted
    assert compare_en("1", "how many $? just about 1$", Language.ENGLISH)


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        # Test negative numbers
        ("-5", "-5", 1),
        # Test for thousands separator
        ("7425000", "7,425,000", 1),
        ("1000", "1 000", 1),
        ("1000", "1000.0", 1),
        # Test thousand separator with floating point number
        ("1000.0", "1,000.0", 1),
        # Test decimal separator as ,
        ("1000.99", "1000,99", 1),
        ("1,22", "1.22", 1),
        ("2.74", "Soucis : 2,74 $ a..", 1),
        # Test .4
        ("0.4", ".4", 1),
        # Test decimals
        ("1000.99", "1,000.99", 1),
    ],
)
def test_number_extraction(gold, pred, expected):
    assert compare_en(gold, pred, match_types=["expr"]) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        ("10/9", "\\frac{10}{9}", 1),
        ("-10/9", "-\\frac{10}{9}", 1),
    ],
)
def test_simple_fraction_notation(gold, pred, expected):
    assert compare_en(gold, pred, match_types=["latex", "expr"]) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        ("$[0,1)$", "$[0,1)$", 1),
        ("$[0,1)$", "$[0,1)$", 1),
        ("$[0,9)$", "$[0,1)$", 0),
        ("$(0,9)$", "$[0,9)$", 0),
        ("$1$", "$-[0,1)$", 0),
    ],
)
def test_fallback(gold, pred, expected):
    assert compare_en(gold, pred, match_types=["latex", "expr"]) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        # Notations
        ("$9$", "Answer don't parse me \\[ 9 \\]", 1),
        ("$9$", "Answer don't parse me $ 9 $", 1),
        ("$9$", "Answer don't parse me $$ 9 $$", 1),
        ("$9$", "Answer don't parse me \\( 9 \\)", 1),
        # Separate line shouldn't work for inline latex
        ("$9$", "Answer don't parse me $ \n 9 \n $", 0),
        ("$9$", "Answer don't parse me \\( \n 9 \n \\)", 0),
        # Separate line should work for block latex
        ("$9$", "Answer don't parse me \\[ \n 9 \n \\]", 1),
        ("$9$", "Answer don't parse me $$ \n 9 \n $$", 1),
        # the $ can appear in the middle of the string
        ("$10/9$", "Answer don't parse me $ \\frac{1}{2} \\$ = \\frac{10}{9} $", 1),
        # Incorrect fractions work
        ("$1/3$", "$\\frac13 $", 1),
        ("$1$", "$\\frac3{3} $", 1),
        # Incorrect sqrt works
        ("$\\sqrt{3}$", "$\\sqrt3 $", 1),
        # frac variants work like frac
        ("$1/3$", "$\\cfrac{1}{3} $", 1),
        ("$1/3$", "$\\dfrac{1}{3} $", 1),
        ("$1/3$", "$\\tfrac{1}{3} $", 1),
        # Simple fractions are parsed
        ("$1/3$", "$ 1/3 $", 1),
        # Styling is removed
        ("$1/3$", "$\\left( \\frac{1}{3} \\right)$", 1),
        ("$1/3$", "$\\boxed{\\frac{1}{3}}$", 1),
        ("$1/3$", "$\\frac{1}{3} \\text{x}$", 1),
        ("$1/3$", "$\\frac{1}{3} \\textbf{x}$", 1),
        # Last = is considered
        ("$1/3$", "$\\k = \\frac{1}{3}$", 1),
        ("$1/3$", "$\\frac{1}{3} \\textbf{x}$", 1),
    ],
)
def test_latex_notation(gold, pred, expected):
    assert compare_en(gold, pred, match_types=["latex"]) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        (
            "$\\frac{2}{3}$",
            " \\$\\dfrac{4}{6} =\\boxed{\\$\\dfrac23 \\approx \\$0.67}",
            1,
        ),
        (
            "$2-2p$",
            "Since $x<2$, it follows that $|x-2|=2-x$. If $2-x=p$, then $x=2-p$. Thus $x-p=\\boxed{2-2p}$.",
            1,
        ),
        (
            "\\boxed{\n\\begin{pmatrix} 0 & 3 \\\\ 0 & -1 \\end{pmatrix}\n}.\n\\end{align*}",
            "\\boxed{\n\\begin{pmatrix} 0 & 3 \\\\ 0 & -1 \\end{pmatrix}\n}.\n\\end{align*}",
            1,
        ),
        # Notations
        (
            "$(3, \\frac{\\pi}{2})$",
            r"We have that $r = \\sqrt{0^2 + 3^2} = 3.$  Also, if we draw the line connecting the origin and $(0,3),$ this line makes an angle of $\\frac{\\pi}{2}$ with the positive $x$-axis.\n\n[asy]\nunitsize(0.8 cm);\n\ndraw((-0.5,0)--(3.5,0));\ndraw((0,-0.5)--(0,3.5));\ndraw(arc((0,0),3,0,90),red,Arrow(6));\n\ndot((0,3), red);\nlabel(\"$(0,3)$\", (0,3), W);\ndot((3,0), red);\n[/asy]\n\nTherefore, the polar coordinates are $\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}.$",
            1,
        ),
        (
            "$\\frac{14}{3}$",
            r"$f(-2)+f(-1)+f(0)=\frac{3(-2)-2}{-2-2}+\frac{3(-1)-2}{-1-2}+\frac{3(0)-2}{0-2}=\frac{-8}{-4}+\frac{-5}{-3}+\frac{-2}{-2}=2+\frac{5}{3}+1=\boxed{\frac{14}{3}}$",
            1,
        ),
        (
            "$\\text{Evelyn}$",
            r"Evelyn covered more distance in less time than Briana, Debra and Angela, so her average speed is greater than any of their average speeds. Evelyn went almost as far as Carla in less than half the time that it took Carla, so Evelyn's average speed is also greater than Carla's. Therefore, $\boxed{\text{Evelyn}}$ is our answer.",
            1,
        ),
        # Test cases from math problems
        (
            "$90^\\circ$",
            r"For the first line, let $t = 2x = 3y = -z.$  Then \[\begin{pmatrix} x \\ y \\ z \end{pmatrix} = \begin{pmatrix} t/2 \\ t/3 \\ -t \end{pmatrix} = \frac{t}{6} \begin{pmatrix} 3 \\ 2 \\ -6 \end{pmatrix}.\]Thus, the direction vector of the first line is $\begin{pmatrix} 3 \\ 2 \\ -6 \end{pmatrix}.$ For the second line, let $t = 6x = -y = -4z.$  Then \[\begin{pmatrix} x \\ y \\ z \end{pmatrix} = \begin{pmatrix} t/6 \\ -t \\ -t/4 \end{pmatrix} = \frac{t}{12} \begin{pmatrix} 2 \\ -12 \\ -3 \end{pmatrix}.\]Thus, the direction vector of the first line is $\begin{pmatrix} 2 \\ -12 \\ -3 \end{pmatrix}.$ Note that \[\begin{pmatrix} 3 \\ 2 \\ -6 \end{pmatrix} \cdot \begin{pmatrix} 2 \\ -12 \\ -3 \end{pmatrix} = 0.\]Hence, the angle between the lines is $\boxed{90^\circ}.$",
            1,
        ),
        (
            "$3\\sqrt{13}$",
            r"We use the distance formula:  \begin{align*} \sqrt{(2 - (-4))^2 + ((-6) - 3)^2} &= \sqrt{6^2 + (-9)^2}\\ & = \sqrt{36 + 81}\\ & = \sqrt{117} = \boxed{3\sqrt{13}}. \end{align*}",
            1,
        ),
        (
            "$\\frac{3}{56}$",
            r"We also know that $q(-1) = ((-1)^2 - 1)p(-1) + 1 = 1.$  Setting $x = -1$ in the equation above, we get \[q(-1) = 20160(-a + b),\]so $-a + b = \frac{1}{20160}.$  Solving for $a$ and $b,$ we find $a = -\frac{29}{40320}$ and $b = -\frac{3}{4480}.$  Hence, \begin{align*} q(x) &= \left( -\frac{29}{40320} x - \frac{3}{4480} \right) (x - 2)(x - 3) \dotsm (x - 7) \\ &= -\frac{(29x + 27)(x - 2)(x - 3) \dotsm (x - 7)}{40320}. \end{align*}In particular, \[q(8) = -\frac{(29 \cdot 8 + 27)(6)(5) \dotsm (1)}{40320} = -\frac{37}{8},\]so \[p(8) = \frac{q(8) + 8}{8^2 - 1} = \boxed{\frac{3}{56}}.\]",
            1,
        ),
        (
            "$2$",
            r"Of the two-digit perfect squares, only $4^2=16$ and $6^2=36$ end in $6$. Thus, there are $\boxed{2}$ distinct possible values for $B$.",
            1,
        ),
        (
            "$15\\mbox{ cm}^2$",
            r"The shaded triangle has a base of length $10\text{ cm}.$ Since the triangle is enclosed in a rectangle of height $3\text{ cm},$ then the height of the triangle is $3\text{ cm}.$ (We know that the enclosing shape is a rectangle, because any figure with four sides, including two pairs of equal opposite sides, and two right angles must be a rectangle.) Therefore, the area of the triangle is $$\frac{1}{2}\times 3 \times 10 = \boxed{15\mbox{ cm}^2}.$$",
            1,
        ),
        (
            "$-2,1$",
            r"By the Integer Root Theorem, the possible integer roots are all the divisors of 14 (including negative divisors), which are $-14,$ $-7,$ $-2,$ $-1,$ $1,$ $2,$ $7,$ and $14.$  Checking, we find that the only integer roots are $\boxed{-2,1}.$",
            1,
        ),
        (
            "$9$",
            r"We use the property that $a \equiv b \pmod{m}$ implies $a^c \equiv b^c \pmod{m}$. Since $129 \equiv -3 \pmod{11}$ and $96 \equiv -3 \pmod{11}$, we have  $$129^{34}+96^{38} \equiv (-3)^{34}+(-3)^{38} \equiv 3^{34}+3^{38} \pmod{11}.$$ Since $3^5 \equiv 1 \pmod{11},$ we can see that $3^{34} = (3^5)^{6} \cdot 3^4$ and $3^{38} = (3^5)^{7} \cdot 3^3.$ Then, $129^{34}+96^{38} \equiv \boxed{9} \pmod{11}.$",
            1,
        ),
        (
            "$90^\\circ$",
            "Therefore, \\begin{align*} \\angle BAC &= \\angle BAD + \\angle DAC \\\\ &= 50^\\circ+40^\\circ \\\\ &= \\boxed{90^\\circ}. \\end{align*}",
            1,
        ),
        (
            "$0$",
            "Note that $p(x)$ has degree at most 2.  Also, $p(a) = p(b) = p(c) = 1.$  Thus, the polynomials $p(x)$ and 1 agree at three different values, so by the Identity Theorem, they are the same polynomial.  Hence, the degree of $p(x)$ (which is the constant polynomial 1) is $\\boxed{0}.$",
            1,
        ),
        # Test long division in base 5
        (
            "$204_5$",
            r"We may carry out long division in base 5 just as in base 10. We have  \[ \begin{array}{c|ccc} \multicolumn{2}{r}{2} & 0 & 4 \\ \cline{2-4} 2 & 4 & 1 & 3 \\ \multicolumn{2}{r}{4} & \downarrow & \\ \cline{2-2} \multicolumn{2}{r}{0} & 1 & \\ \multicolumn{2}{r}{} & 0 & \downarrow \\ \cline{3-3} \multicolumn{2}{r}{} & 1 & 3 \\ \multicolumn{2}{r}{} & 1 & 3 \\ \cline{3-4} \multicolumn{2}{r}{} & & 0 \end{array} \]for a quotient of $\boxed{204_5}$. Note that in the above calculation we have used that $13_5$ divided by $2_5$ is $4_5$, which follows from $4_5\times2_5=8_{10}=13_5$.",
            1,
        ),
        (
            "$(6,31,-1)$",
            "Let $\\alpha$ be a root of $x^3 - 3x^2 + 4x - 1 = 0,$ so $\\alpha^3 = 3 \\alpha^2 - 4 \\alpha + 1.$ Then solving the system of equations, we find $(p,q,r) = \\boxed{(6,31,-1)}.$",
            1,
        ),
        (
            "$1 \\pm \\sqrt{19}$",
            "This simplifies to $64y + 1920 = 0,$ so $y = -30.$ Then $x^2 - 2x - 48 = -30,$ or $x^2 - 2x - 18 = 0.$ By the quadratic formula, $x = \\boxed{1 \\pm \\sqrt{19}}.$",
            1,
        ),
        (
            "$3 \\pm 2 \\sqrt{2}$",
            "This gives us $x^2 + 1 = 6x,$ or $x^2 - 6x + 1 = 0.$ By the quadratic formula, the roots are $x = \\boxed{3 \\pm 2 \\sqrt{2}}.$",
            1,
        ),
        (
            "$\\{1\\pm\\sqrt{5},-2\\}$",
            "The roots of $P(x)$ are $-2$ and $1 \\pm \\sqrt{5}$, so the answer is $\\boxed{\\{1\\pm\\sqrt{5},-2\\}}.$",
            1,
        ),
        (
            "$f(2) < f(1) < f(4)$",
            'The graph of $f(x) = x^2 + bx + c$ is an upward-facing parabola, and the condition\n\\[f(2 + t) = f(2 - t)\\]tells us that the axis of symmetry of the parabola is the line $x = 2.$  Thus, $f(x)$ is an increasing function of $|x - 2|.$  In other words, the farther $x$ is from 2, the greater $f(x)$ is.\n\n[asy]\nunitsize(1.5 cm);\n\nreal parab (real x) {\n  return (x^2/4);\n}\n\ndraw(graph(parab,-2,2),red);\ndraw((0,-0.5)--(0,2),dashed);\n\nlabel("$x = 2$", (0,2), N);\ndot("$(2,f(2))$", (0,0), SE);\ndot("$(1,f(1))$", (-0.8,parab(-0.8)), SW);\ndot("$(4,f(4))$", (1.6,parab(1.6)), SE);\n[/asy]\n\nHence, $\\boxed{f(2) < f(1) < f(4)}.$',
            1,
        ),
        (
            "$2 \\sin b \\cos a$",
            "By sum-to-product,\n\\[\\sin (a + b) - \\sin (a - b) = \\boxed{2 \\sin b \\cos a}.\\]",
            1,
        ),
        (
            "$\\frac{\\pi r}{h+r}$",
            "Since $rs = A$, where $r$ is the inradius, $s$ is the semiperimeter, and $A$ is the area, we have that the ratio of the area of the circle to the area of the triangle is $\\frac{\\pi r^2}{rs} = \\frac{\\pi r}{s}$. Now we try to express $s$ as $h$ and $r$. Denote the points where the incircle meets the triangle as $X,Y,Z$, where $O$ is the incenter, and denote $AX = AY = z, BX = BZ = y, CY = CZ = x$. Since $XOZB$ is a square (tangents are perpendicular to radius), $r = BX = BZ = y$. The perimeter can be expressed as $2(x+y+z)$, so the semiperimeter is $x+y+z$. The hypotenuse is $AY+CY = z+x$. Thus we have $s = x+y+z = (z+x)+y = h+r$. The answer is $\\boxed{\\frac{\\pi r}{h+r}}$.'], Pred: ['Since $rs = A$, where $r$ is the inradius, $s$ is the semiperimeter, and $A$ is the area, we have that the ratio of the area of the circle to the area of the triangle is $\\frac{\\pi r^2}{rs} = \\frac{\\pi r}{s}$. Now we try to express $s$ as $h$ and $r$. Denote the points where the incircle meets the triangle as $X,Y,Z$, where $O$ is the incenter, and denote $AX = AY = z, BX = BZ = y, CY = CZ = x$. Since $XOZB$ is a square (tangents are perpendicular to radius), $r = BX = BZ = y$. The perimeter can be expressed as $2(x+y+z)$, so the semiperimeter is $x+y+z$. The hypotenuse is $AY+CY = z+x$. Thus we have $s = x+y+z = (z+x)+y = h+r$. The answer is $\\boxed{\\frac{\\pi r}{h+r}}$.",
            1,
        ),
        ("$125$ miles", "The distance is $\\boxed{125\\textnormal{ miles}}.$", 1),
        (
            "$[-1, -\\frac{1}{2}) \\cup (-\\frac{1}{2}, 0) \\cup (0, 1) \\cup (1, \\infty)$",
            "The solution set is $\\boxed{[-1, -\\tfrac12) \\cup (-\\tfrac12, 0) \\cup (0, 1) \\cup (1, \\infty)}.$",
            1,
        ),
        ("$\\sqrt{2}+\\sqrt{5}$", "The answer is $\\boxed{\\sqrt 2+\\sqrt 5}$", 1),
        ("$\\frac{9}{4}\\pi$", "Therefore $\\boxed{\\frac94\\pi}$.", 1),
        ("x \\in \\boxed{\\{-1\\} \\cup [0,7)}.$", "x \\in \\boxed{\\{-1\\} \\cup [0,7)}.$", 1),
    ],
)
def test_latex_notation_math(gold, pred, expected):
    assert compare_en(gold, pred, match_types=["latex"]) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        # Basic support for all relations
        (
            "$x >= 5$",
            "Therefore $x \\geq 5$ is the solution.",
            1,
        ),
        (
            "$x < 3$",
            "We find that $x \\lt 3$.",
            1,
        ),
        (
            "$x \\leq 2$",
            "Thus $x <= 2$ is our answer.",
            1,
        ),
        (
            "$x > 5$",
            "Therefore $x \\gt 5$ is the solution.",
            1,
        ),
        (
            "$x != 3$",
            "We find that $x \\neq 3$.",
            1,
        ),
        # Incorrect cases
        (
            "$x > 5$",
            "Therefore $x < 5$ is the solution.",
            0,
        ),
        (
            "$x \\geq 5$",
            "The solution is $x \\leq 5$",
            0,
        ),
        (
            "$x \\neq 5$",
            "The solution is $x = 5$",
            0,
        ),
        # Test flipped inequalities
        (
            "$x \\leq 5$",
            "$5 \\geq x$",
            1,
        ),
        (
            "$x \\geq 5$",
            "$5 \\leq x$",
            1,
        ),
    ],
)
def test_relations_math(gold, pred, expected):
    assert compare_en(gold, pred, match_types=["latex"]) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        # Test Identity Matrix
        (
            r"$\begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}$",
            r"The identity matrix is $ \begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix} $.",
            1,
        ),
        # Test bmatrix
        (
            r"$\begin{bmatrix}0 & 0 \\0 & 0\end{bmatrix}$",
            r"Here is a zero matrix: $ \begin{pmatrix}0 & 0 \\0 & 0\end{pmatrix} $",
            1,
        ),
        # Test Matrix with Special Formatting
        (
            r"$\begin{pmatrix}1 & 2 \\3 & 4\end{pmatrix}$",
            r"Special matrix: $ \left[\begin{array}{cc}1 & 2 \\3 & 4\end{array}\right] $",
            1,
        ),
        # Test Matrix with Fraction Entries
        (
            r"$\begin{pmatrix}\frac{1}{2} & \frac{3}{4} \\ \frac{5}{6} & \frac{7}{8}\end{pmatrix}$",
            r"Matrix with fractions: $ \begin{pmatrix}\frac{1}{2} & \frac{3}{4} \\ \frac{5}{6} & \frac{7}{8}\end{pmatrix} $",
            1,
        ),
        # Test matrix addition
        (
            r"$\begin{pmatrix}6 & 8 \\ 10 & 12\end{pmatrix}$",
            r"The sum is $\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix} + \begin{pmatrix}5 & 6 \\ 7 & 8\end{pmatrix}$",
            1,
        ),
        # Test matrix multiplication
        (
            r"$\begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}$",
            r"When multiplying by identity: $\begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix} \begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}$",
            1,
        ),
        # Test incorrect matrix
        (
            r"$\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}$",
            r"The matrix is $\begin{pmatrix}1 & 2 \\ 3 & 5\end{pmatrix}$",  # Different value in bottom right
            0,
        ),
    ],
)
def test_matrix_extraction(gold, pred, expected):
    assert compare_en(gold, pred, match_types=["latex"]) == expected


def test_precision():
    assert sympy_expr_eq(sympy.Rational(1, 3), sympy.Float(0.333), precision=3)
    assert not sympy_expr_eq(sympy.Rational(1, 3), sympy.Float(0.333), precision=4)

    # It should work with more nuanced pairs
    assert sympy_expr_eq(sympy.Rational(1, 3) + 1, sympy.Float(1.333), precision=3)
    assert not sympy_expr_eq(sympy.Rational(1, 3) + 1, sympy.Float(1.333), precision=4)

    # From latex
    assert compare_en("$\\frac{1}{3}$", "0.3333$", match_types=["latex", "expr"], precision=4) == 1


# Tests from qwen parser
@pytest.mark.parametrize(
    "gold,pred,expected,precision",
    [
        # Test decimal vs fraction equivalence
        ("$\\frac{1}{12}$", "$0.0833333333333333$", 1, 6),
        ("$(1,\\frac{9}{2})$", "$(1,4.5)$", 1, 6),
        # Test algebraic expressions
        ("$\\frac{x+2}{7}$", "$\\frac{x}{7}+\\frac{2}{7}$", 1, 6),
        ("$\\tan^2(y)+1$", "$\\sec^2(y)$", 1, 6),
        # Test complex matrices
        (
            "$\\begin{pmatrix}-\\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\\\\\end{pmatrix}$",
            "$\\begin{pmatrix}-\\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\\\\\end{pmatrix}$",
            1,
            6,
        ),
        (
            "$\\begin{pmatrix}\\frac{1}{3\\sqrt[3]{x}^2}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\\\\\end{pmatrix}$",
            "$\\begin{pmatrix}\\frac{1}{3x^{2/3}}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\end{pmatrix}$",
            1,
            6,
        ),
        # Test equations
        ("$34x+45y-20z+100=0$", "$-34x-45y+20z-100=0$", 1, 6),
        # Test matrix with decimals
        (
            "$(\\begin{pmatrix}\\frac{1}{3}\\\\ \\frac{1}{5} \\end{pmatrix})$",
            "$\\begin{pmatrix}0.33\\\\0.2 \\end{pmatrix}$",
            1,
            2,
        ),
        # Test expression order invariance
        (
            "$\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{15+2\\sqrt{33}}$",
            "$\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{2\\sqrt{33}+15}$",
            1,
            6,
        ),
        # Test non-equivalent expressions
        ("$(a+5)(b+2)$", "$(+5)(b+2)$", 0, 6),
        ("$2$", "$\\frac{1+\\sqrt{5}}{2}$", 0, 6),
        ("$4$", "$\\frac{34}{16}+\\frac{\\sqrt{1358}}{16}$", 0, 6),
        ("$1\\sqrt{19}$", "$1$", 0, 6),
        # Test intervals
        ("$(\\frac{3}{5},\\frac{8}{3}]$", "$(0.6,2.6667]$", 1, 2),
        # Test non-equivalent algebraic expressions
        ("$x+2n+1$", "$x+1$", 0, 6),
    ],
)
def test_complex_math_expressions(gold, pred, expected, precision):
    assert compare_en(gold, pred, match_types=["latex", "expr"], precision=precision) == expected
