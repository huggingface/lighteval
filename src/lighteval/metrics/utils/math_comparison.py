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

# Heavily inspired by https://github.com/QwenLM/Qwen2.5-Math and https://github.com/huggingface/lm-evaluation-harness
import re
from itertools import product

from sympy import (
    Basic,
    Eq,
    FiniteSet,
    Float,
    GreaterThan,
    Interval,
    LessThan,
    MatrixBase,
    MatrixExpr,
    Mul,
    Number,
    Rational,
    Set,
    StrictGreaterThan,
    StrictLessThan,
    Symbol,
    simplify,
)
from sympy.core.relational import Relational

from lighteval.utils.timeout import timeout


def safe_sympy_doit(a: Basic | MatrixBase):
    """Safely execute doit() on a sympy expression, catching exceptions.
      Doit in sympy will evaluate expressions it will pass the expression tree and evluate nodes.
      For example for 1+1+1 it will evaluate the additions and return 3. One issue with it is that it maybe
      evaluates too much as integrals will also be evaluated.

      As we are using latex2sympy2_extended, evaluates are

    Args:
        a: A sympy Basic or MatrixBase expression to evaluate

    Returns:
        The result of a.doit() if successful, otherwise returns the original expression
    """
    try:
        return a.doit()
    except TimeoutError:
        raise
    except:  # noqa: E722
        pass
    return a


def is_atomic_or_pct_atomic(expr: Basic | MatrixBase, atomic_type: type) -> bool:
    """Check if expression is either an atomic type or percentage atomic type.

    Args:
        expr: The sympy expression to check
        atomic_type: The atomic type to check for

    Returns:
        True if expr is atomic_type or percentage atomic type, False otherwise
    """
    return isinstance(expr, atomic_type) or (
        # Check for percentage representation: latex2sympy_extended converts "X%" into X*Rational(1,100)
        # So we detect percentages by looking for this multiplication structure
        isinstance(expr, Mul)
        and len(expr.args) == 2
        and expr.args[1] == Rational(1, 100)
        and isinstance(expr.args[0], atomic_type)
    )


def sympy_numeric_eq(a: Basic | MatrixBase, b: Basic | MatrixBase, precision: int):
    """Compare two sympy expressions numerically with given precision.

    Args:
        a: First sympy expression
        b: Second sympy expression
        precision: Number of decimal places to compare

    Returns:
        True if expressions are numerically equal within precision, False otherwise
    """
    # Only do this when one of the two is a float, in other cases use symbolic equality as this could lead to false positives
    # E.g we want 1/3 == 0.333333 to work
    if isinstance(a, (MatrixBase, MatrixExpr)) and isinstance(b, (MatrixBase, MatrixExpr)):
        a = safe_sympy_doit(a)
        b = safe_sympy_doit(b)

        # If we have matrices and one of them is only made of floats, we can use the same logic as above
        if isinstance(a, (MatrixBase)) and isinstance(b, (MatrixBase)) and a.shape == b.shape:
            return all(sympy_numeric_eq(a_elem, b_elem, precision) for a_elem, b_elem in zip(a.flat(), b.flat()))

    # Ensure this also works for percentage numbers so that 0.333333% = 0.33333333333 with precision 4
    elif is_atomic_or_pct_atomic(a, Number) or is_atomic_or_pct_atomic(b, Number):
        # If one of them is a float or a negative atomic number, we can try to use precision
        if is_atomic_or_pct_atomic(a, Float) or is_atomic_or_pct_atomic(b, Float):
            a = safe_sympy_doit(a)
            b = safe_sympy_doit(b)
            # Now if both are numbers, we can use precision
            if isinstance(a, (Number)) and isinstance(b, (Number)):
                return a.round(precision) == b.round(precision)
        else:
            return safe_sympy_doit(a) == safe_sympy_doit(b)

    else:
        try:
            return (a - b).evalf(chop=True) == 0  # type: ignore
        except TimeoutError:
            raise
        except:  # noqa: E722
            pass

    return False


def sympy_symbolic_eq(a: Basic | MatrixBase, b: Basic | MatrixBase) -> bool:
    """Compare two sympy expressions symbolically.

    Args:
        a: First sympy expression
        b: Second sympy expression

    Returns:
        True if expressions are symbolically equal, False otherwise
    """
    try:
        a_b_diff = simplify((a - b))  # type: ignore
        if isinstance(a_b_diff, MatrixBase) and a_b_diff.is_zero_matrix:
            return True
        elif isinstance(a_b_diff, Basic) and a_b_diff.is_zero:
            return True
    except TimeoutError:
        raise
    except:  # noqa: E722
        pass

    return False


def sympy_deep_compare_finite_set(a: FiniteSet, b: FiniteSet, precision: int) -> bool:
    """Compare two finite sets by comparing each element with given precision.

    Args:
        a: First finite set
        b: Second finite set
        precision: Number of decimal places to compare

    Returns:
        True if sets contain equal elements within precision, False otherwise
    """
    # This ensures it works for {1/3} and {0.333333}
    if len(a) == len(b) and all(sympy_expr_eq(a, b, precision) for a, b in zip(a, b)):
        return True

    return False


def sympy_compare_set_interval(a: FiniteSet, b: Interval, precision: int) -> bool:
    """Compare a finite set with an interval.

    Args:
        a: Finite set to compare
        b: Interval to compare
        precision: Number of decimal places to compare

    Returns:
        True if set and interval are equivalent, False otherwise
    """
    # Only compare if it's the special case of 2 elements
    if len(a) == 2 and b.is_open:
        return sympy_deep_compare_finite_set(a, FiniteSet(b.start, b.end), precision)

    return False


def sympy_compare_interval(a: Interval, b: Interval, precision: int) -> bool:
    """Compare two intervals.

    Args:
        a: First interval
        b: Second interval
        precision: Number of decimal places to compare endpoints

    Returns:
        True if intervals are equal, False otherwise
    """
    return (
        a.left_open == b.left_open
        and a.right_open == b.right_open
        and sympy_expr_eq(a.start, b.start, precision)
        and sympy_expr_eq(a.end, b.end, precision)
    )


def sympy_compare_relational(gold: Relational, pred: Relational, precision: int) -> bool:
    """Compare two relational expressions.

    Args:
        gold: First relational expression
        pred: Second relational expression
        precision: Number of decimal places to compare

    Returns:
        True if relations are equivalent, False otherwise
    """

    # Helper to check if expressions are equivalent when flipped
    def are_flipped_inequalities_equal(a: Relational, b: Relational) -> bool:
        try:
            return sympy_expr_eq(a.lhs - a.rhs, b.rhs - b.lhs, precision)  # type: ignore
        except TimeoutError:
            raise
        except:  # noqa: E722
            pass
        return False

    # Same type of relation (e.g. both <= or both >=)

    try:
        if type(gold) == type(pred) and sympy_expr_eq(gold.lhs - gold.rhs, pred.lhs - pred.rhs, precision):  # type: ignore
            return True
    except TimeoutError:
        raise
    except:  # noqa: E722
        pass

    # Check flipped inequalities (a <= b equals b >= a)
    if (
        isinstance(gold, GreaterThan)
        and isinstance(pred, LessThan)
        or isinstance(gold, LessThan)
        and isinstance(pred, GreaterThan)
        or isinstance(gold, StrictGreaterThan)
        and isinstance(pred, StrictLessThan)
        or isinstance(gold, StrictLessThan)
        and isinstance(pred, StrictGreaterThan)
        or isinstance(gold, Eq)
        and isinstance(pred, Eq)
    ) and are_flipped_inequalities_equal(gold, pred):
        return True

    return False


def sympy_str_eq(a: Basic | MatrixBase, b: Basic | MatrixBase) -> bool:
    """Compare two sympy expressions by string representation.

    Args:
        a: First sympy expression
        b: Second sympy expression

    Returns:
        True if string representations are equal, False otherwise
    """
    a_doit = safe_sympy_doit(a)
    b_doit = safe_sympy_doit(b)

    try:
        # Structural equality, the cheapest but the dumbest one, it will fail for a + b vs b + a
        if a_doit == b_doit:
            return True
        # Then do a simple str comparison
        if str(a_doit).strip() == str(b_doit).strip():
            return True
    except TimeoutError:
        raise
    except:  # noqa: E722
        pass
    return False


def sympy_compare_sets(gold: Set | Basic | MatrixBase, pred: Set | Basic | MatrixBase, precision: int) -> bool:
    """Compare two sympy sets for equality using multiple methods.

    Args:
        gold: First sympy set (expected)
        pred: Second sympy set (predicted)
        precision: Number of decimal places to compare

    Returns:
        True if sets are equal by any comparison method, False otherwise
    """
    # Convert non-sets to singleton sets
    a_set = gold if isinstance(gold, Set) else FiniteSet(gold)
    b_set = pred if isinstance(pred, Set) else FiniteSet(pred)

    # If both are intervals, use interval comparison
    if isinstance(a_set, Interval) and isinstance(b_set, Interval):
        return sympy_compare_interval(a_set, b_set, precision)

    # Try direct set equality
    if a_set == b_set:
        return True
    if a_set.symmetric_difference(b_set).is_empty:
        return True

    # For finite sets, compare elements
    if isinstance(a_set, FiniteSet) and isinstance(b_set, FiniteSet):
        return sympy_deep_compare_finite_set(a_set, b_set, precision)

    # Handle interval vs finite set cases
    if isinstance(a_set, Interval) and isinstance(b_set, FiniteSet):
        return sympy_compare_set_interval(b_set, a_set, precision)
    if isinstance(a_set, FiniteSet) and isinstance(b_set, Interval):
        return sympy_compare_set_interval(a_set, b_set, precision)

    return False


def sympy_expr_eq(gold: Basic | MatrixBase, pred: Basic | MatrixBase, precision: int) -> bool:
    """Compare two sympy expressions for equality using multiple methods.

    Args:
        gold: First sympy expression (expected)
        pred: Second sympy expression (predicted)
        precision: Number of decimal places to compare

    Returns:
        True if expressions are equal by any comparison method, False otherwise
    """
    # If the reference is relational, but the target is not, it's possible it's a case of answer=x+1+z, so we just take x+1+z
    # We assume that the gold never needs to be simplified, so we don't handle that case
    # e.g 1+1+1=3 will never be simplified to 3; it would be possible to do so with lhs-rhs == 0, but we assume the gold is at its most simplified form.
    # The new latex2sympy2 will actually convert such cases automatically, but so this is in theory not needed
    if isinstance(gold, Eq) and not isinstance(pred, Relational) and isinstance(gold.lhs, Symbol):
        gold = gold.rhs

    # Here we respect the gold and simplify accordingly, thus any of
    # k=x+1+z or 1+1+1=3 will be simplified to rhs
    if isinstance(pred, Eq) and not isinstance(gold, Eq):
        pred = pred.rhs

    # Start with simple str and expr comparisson as it's the fastest
    # str comparison is better, than simple eq, because it will also handle missarangments
    if sympy_str_eq(gold, pred):
        return True

    # Support for equations
    if isinstance(gold, Relational) and isinstance(pred, Relational):
        return sympy_compare_relational(gold, pred, precision)

    elif isinstance(gold, Set) or isinstance(pred, Set):
        return sympy_compare_sets(gold, pred, precision)

    elif isinstance(gold, (Basic, MatrixBase)) and isinstance(pred, (Basic, MatrixBase)):
        # Mostly so that 0.333333 = 1/3
        if sympy_numeric_eq(gold, pred, precision):
            return True
        # Then try symbolic equality
        if sympy_symbolic_eq(gold, pred):
            return True

    return False


complex_number_pattern = re.compile(
    r"""
    # Complex number indicators
    \\mathbb\{C\}|        # Complex number set ℂ
    \\i\b|                # Complex i
    \bi\b|                # Standalone i
    \\text\{i\}|          # Text i
    \\mathrm\{i\}|        # Roman i
    \\imath\b|            # Alternative i notation

    # Matrix operations
    \\det|                # Determinant
    \\operatorname\{tr\}| # Trace
    \\operatorname\{rank\}| # Rank
    \\text\{rank\}|
    \\arg\{|              # Complex argument
    \\Re\{|               # Real part
    \\Im\{|               # Imaginary part
    \\operatorname\{Re\}| # Real part alternate
    \\operatorname\{Im\}| # Imaginary part alternate
    \\text\{Re\}|         # Real part text
    \\text\{Im\}          # Imaginary part text
""",
    re.VERBOSE,
)


def should_treat_as_complex(latex_str: str) -> bool:
    """
    Returns True if the latex string likely contains complex numbers, matrices, or vectors.
    """

    return bool(complex_number_pattern.search(latex_str))


def compare_gold_target(
    gold: list[Basic | MatrixBase | str], target: list[Basic | MatrixBase | str], precision: int
) -> bool:
    @timeout(timeout_seconds=10)
    def compare_single_extraction(gold: Basic | MatrixBase | str, target: Basic | MatrixBase | str) -> bool:
        # If both are sympy expressions, we can use sympy to compare them
        if isinstance(gold, (Basic, MatrixBase)) and isinstance(target, (Basic, MatrixBase)):
            return sympy_expr_eq(gold, target, precision)

        # We don't support str / sympy.Expr comparison. Imo there is no point in doing this, as chances
        # of this happening are very low.  The only why one of them is not converted to sympy expression
        # is usually because the parsing logic failed in this case we should improve the parsing logic
        # instead of somehow fixing adhoc.
        elif isinstance(gold, str) and isinstance(target, str):
            # We just do string comparison for everything else
            gold = gold.strip()
            target = target.strip()

            # Ensure it's both not empty and equal
            return len(gold) > 0 and len(target) > 0 and gold == target

        return False

    def compare_single_extraction_wrapper(g, t):
        try:
            return compare_single_extraction(g, t)
        except TimeoutError:
            return False

    return any(compare_single_extraction_wrapper(g, t) for g, t in product(gold, target))
