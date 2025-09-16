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

# tests/utils/pretend_missing.py
import functools
import importlib

import pytest

import lighteval.utils.imports as imports


def pretend_missing(*names):
    """
    Decorator: pretend that certain packages are missing
    by patching mypkg.utils.is_package_available.
    """

    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            from unittest.mock import patch

            def fake(name):
                return False if name in names else (importlib.util.find_spec(name) is not None)

            with patch.object(imports, "is_package_available", side_effect=fake):
                # If your module caches results at import time, reload here
                import lighteval

                importlib.reload(lighteval)

                return test_func(*args, **kwargs)

        return wrapper

    return decorator


@pretend_missing("langdetect")
def test_langdetect_required_for_ifeval():
    from lighteval.main_accelerate import accelerate

    with pytest.raises(
        ImportError,
        match="Through the use of ifeval_prompt, you requested the use of `langdetect` for this evaluation, but it is not available in your current environment. Please install it using pip.",
    ):
        accelerate(model_args="model_name=gpt2,batch_size=1", tasks="extended|ifeval|0", max_samples=0)


@pretend_missing("spacy", "stanza")
def test_multilingual_required_for_xnli():
    from lighteval.main_accelerate import accelerate

    with pytest.raises(
        ImportError,
        match="Through the use of get_multilingual_normalizer, you are trying to run an evaluation requiring multilingual capabilities. Please install the required extra: `pip install lighteval[multilingual]`",
    ):
        accelerate(model_args="model_name=gpt2,batch_size=1", tasks="lighteval|xnli_zho_mcf|0", max_samples=0)


@pretend_missing("vllm")
def test_vllm_required_for_vllm_usage():
    from lighteval.main_vllm import vllm

    with pytest.raises(
        ImportError,
        match="You requested the use of `vllm` for this evaluation, but it is not available in your current environment. Please install it using pip.'",
    ):
        vllm(model_args="model_name=gpt2,batch_size=1", tasks="lighteval|xnli_zho_mcf|0", max_samples=0)
