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

import importlib

import torch

import lighteval.tasks.multilingual.tasks.swiss_legal.metrics as swiss_legal_metrics


def test_import_does_not_change_global_torch_settings(monkeypatch):
    """
    Importing the swiss legal metrics (done for every run with --load-tasks-multilingual)
    must not lower the float32 matmul precision for the whole process.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (8, 0))
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", torch.backends.cudnn.benchmark)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)
    precision = torch.get_float32_matmul_precision()
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32

    try:
        importlib.reload(swiss_legal_metrics)
        assert torch.get_float32_matmul_precision() == precision
        assert torch.backends.cuda.matmul.allow_tf32 == allow_tf32
    finally:
        torch.set_float32_matmul_precision(precision)
