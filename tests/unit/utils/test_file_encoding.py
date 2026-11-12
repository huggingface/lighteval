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

"""Files that users write (model configs, task lists) are UTF-8 and must be read as UTF-8 on every OS.

Without an explicit encoding, `open()` uses the locale's code page on Windows (e.g. cp1252), which garbles
non-ASCII text or raises UnicodeDecodeError.
"""

from lighteval.models.abstract_model import ModelConfig
from lighteval.tasks.registry import Registry


SYSTEM_PROMPT = "Réponds en français. 日本語でも可。"


def test_model_config_from_path_reads_utf8(tmp_path):
    config_file = tmp_path / "model.yaml"
    config_file.write_text(
        f'model_parameters:\n  model_name: "my-model"\n  system_prompt: "{SYSTEM_PROMPT}"\n', encoding="utf-8"
    )

    config = ModelConfig.from_path(str(config_file))

    assert config.system_prompt == SYSTEM_PROMPT


def test_task_list_file_reads_utf8(tmp_path):
    tasks_file = tmp_path / "tasks.txt"
    tasks_file.write_text(f"# {SYSTEM_PROMPT}\nmy_task|0\n", encoding="utf-8")

    registry = object.__new__(Registry)  # only the file-parsing helper is under test

    assert registry._get_full_task_list_from_input_string(str(tasks_file)) == ["my_task|0"]
