# MIT License
#
# Copyright (c) 2024 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest

from lighteval.utils.imports import Extra, is_package_available, requires


def test_requires():
    @requires("sglang")
    class RandomModel:
        pass

    assert RandomModel.__name__ == "RandomModel"
    assert RandomModel.__class__.__name__ == "DummyObject"

    with pytest.raises(
        ImportError,
        match="Through the use of RandomModel, you requested the use of `sglang` for this evaluation, but it is not available in your current environment. Please install it using pip.",
    ):
        RandomModel()


def test_requires_with_extra():
    @requires(Extra.TGI)
    class RandomModel:
        pass

    with pytest.raises(
        ImportError,
        match=r"Through the use of RandomModel, you are trying to run an evaluation requiring tgi capabilities. Please install the required extra: `pip install lighteval\[tgi\]`",
    ):
        RandomModel()


def test_requires_with_wrong_dependency():
    with pytest.raises(
        RuntimeError,
        match="A dependency was specified with @requires, but it is not defined in the possible dependencies defined in the pyproject.toml: `random_dependency`",
    ):

        @requires("random_dependency")
        class RandomModel:
            pass


def test_is_package_available():
    assert is_package_available("torch")


def test_is_package_unavailable():
    assert not is_package_available("tensorboardX")


def test_is_package_is_not_specified_in_pyproject_toml():
    with pytest.raises(
        RuntimeError,
        match="Package tensorflow was tested against, but isn't specified in the pyproject.toml file. Please specifyit as a potential dependency or an extra for it to be checked.",
    ):
        is_package_available("tensorflow")
