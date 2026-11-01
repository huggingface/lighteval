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


"""Checks that the type annotations of the package are valid types.

Some annotations used functions instead of types, e.g. `callable` or `np.array`.
Those are rejected by type checkers (`mypy` reports them as `valid-type` errors)
and cannot be resolved at runtime by `typing.get_type_hints`.

See https://github.com/huggingface/lighteval/issues/759.
"""

import ast
import builtins
import importlib
import inspect
import typing
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "lighteval"

# Modules that are cheap to import and are therefore resolved when looking up
# an annotation such as `np.array`. Annotations referring to anything else are
# left unchecked.
RESOLVABLE_MODULES = ("typing", "typing_extensions", "numpy")

try:  # `typing_extensions` is only used to detect a bare `Annotated`.
    import typing_extensions
except ImportError:  # pragma: no cover
    typing_extensions = None


def _is_function(obj) -> bool:
    """Whether `obj` is a function, i.e. something that cannot be used as a type."""
    return inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.ismethod(obj)


def _is_annotated(obj) -> bool:
    """Whether `obj` is `typing.Annotated`, which requires a type argument."""
    if obj is typing.Annotated:
        return True
    return typing_extensions is not None and obj is typing_extensions.Annotated


def _module_bindings(tree: ast.Module) -> dict[str, tuple[str, object]]:
    """Map the names defined at module level to what they refer to."""
    bindings: dict[str, tuple[str, object]] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                bindings[name] = ("module", alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                bindings[name] = ("import", (node.module or "", alias.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bindings[node.name] = ("function", node)
    return bindings


def _resolve(module_name: str, attribute: str):
    """Resolve `module_name.attribute`, returning None when it cannot be done."""
    if module_name.split(".")[0] not in RESOLVABLE_MODULES:
        return None
    try:
        return getattr(importlib.import_module(module_name), attribute, None)
    except ImportError:
        return None


def _annotations(tree: ast.Module):
    """Yield every annotation of the module (arguments, return values, variables)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            all_arguments = (
                list(arguments.posonlyargs)
                + list(arguments.args)
                + list(arguments.kwonlyargs)
                + ([arguments.vararg] if arguments.vararg else [])
                + ([arguments.kwarg] if arguments.kwarg else [])
            )
            for argument in all_arguments:
                if argument.annotation is not None:
                    yield argument.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def _check_resolved(resolved, label: str, lineno: int, problems: list[str], is_root: bool) -> None:
    """Report `resolved` when it is not something that can be used as a type."""
    if _is_function(resolved):
        problems.append(f"line {lineno}: `{label}` is a function, not a type")
    elif is_root and _is_annotated(resolved):
        problems.append(f"line {lineno}: `Annotated` requires a type argument")


def _check_builtin(name: str, lineno: int, problems: list[str]) -> None:
    """Report `name` when it is a builtin function, such as `callable`."""
    if hasattr(builtins, name) and _is_function(getattr(builtins, name)):
        problems.append(f"line {lineno}: `builtins.{name}` is a function, not a type")


def _check_name(node: ast.Name, bindings: dict[str, tuple[str, object]], problems: list[str], is_root: bool) -> None:
    """Check an annotation that is a bare name."""
    name = node.id
    if name not in bindings:
        _check_builtin(name, node.lineno, problems)
        return

    kind, value = bindings[name]
    if kind == "function":
        problems.append(f"line {node.lineno}: `{name}` is a function, not a type")
    elif kind == "module":
        problems.append(f"line {node.lineno}: `{name}` is a module, not a type")
    elif kind == "import":
        module_name, attribute = value
        resolved = _resolve(module_name, attribute)
        if resolved is not None:
            _check_resolved(resolved, f"{module_name}.{attribute}", node.lineno, problems, is_root)


def _check_attribute(
    node: ast.Attribute, bindings: dict[str, tuple[str, object]], problems: list[str], is_root: bool
) -> None:
    """Check an annotation that is an attribute access, such as `np.array`."""
    base = node.value
    if not isinstance(base, ast.Name):
        _check_annotation(base, bindings, problems, is_root=False)
        return

    kind, value = bindings.get(base.id, (None, None))
    resolved = None
    if kind == "module":
        resolved = _resolve(value, node.attr)
    elif kind == "import":
        module_name, attribute = value
        imported = _resolve(module_name, attribute)
        resolved = getattr(imported, node.attr, None) if imported is not None else None
    if resolved is None:
        return

    _check_resolved(resolved, f"{base.id}.{node.attr}", node.lineno, problems, is_root)


def _check_annotation(
    node: ast.AST, bindings: dict[str, tuple[str, object]], problems: list[str], is_root: bool = True
) -> None:
    """Collect the problems of a single annotation.

    `is_root` tells whether the annotation is used as a type on its own, in which
    case `Annotated` is missing its type argument.
    """
    if isinstance(node, ast.Name):
        _check_name(node, bindings, problems, is_root)
    elif isinstance(node, ast.Attribute):
        _check_attribute(node, bindings, problems, is_root)
    elif isinstance(node, ast.Subscript):
        # `Annotated[str, ...]` is valid, only a bare `Annotated` is not.
        _check_annotation(node.value, bindings, problems, is_root=False)
    else:
        for child in ast.iter_child_nodes(node):
            _check_annotation(child, bindings, problems, is_root=False)


def _invalid_annotations(path: Path) -> list[str]:
    """Return the annotations of `path` that are not valid types."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bindings = _module_bindings(tree)
    problems: list[str] = []

    for annotation in _annotations(tree):
        _check_annotation(annotation, bindings, problems)
    return problems


def _package_files() -> list[Path]:
    return sorted(path for path in PACKAGE_ROOT.rglob("*.py"))


@pytest.mark.parametrize("path", _package_files(), ids=lambda path: str(path.relative_to(PACKAGE_ROOT)))
def test_annotations_are_valid_types(path: Path):
    """No annotation of the package may refer to something that is not a type."""
    problems = _invalid_annotations(path)

    assert not problems, f"invalid type annotations in {path}: " + ", ".join(problems)
