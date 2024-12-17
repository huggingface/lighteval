import pytest

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


def compare_en(
    gold: str, pred: str, language: Language = Language.ENGLISH, match_types: list[str] = ["latex", "expr"]
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
            "الباقي: 4 ÷ 3 = 1 بباقي 1 نكتب 1 فوق الخط ونضع الباقي 1 تحت الرقم الرابع. 6. نجمع الأرقام فوق الخط: 438 7. نتحقق من النتيجة: 438 × 3 = 1314 لذا، فإن ناتج 1314 ÷ 3 هو 438. الباقي من القسمة هو 0، مما يعني أن 1314 قابل للقسمة على 3 تمامًا.",
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
        ("$\\frac{a}{b}*c$", "$\\fracabc $", 1),
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
