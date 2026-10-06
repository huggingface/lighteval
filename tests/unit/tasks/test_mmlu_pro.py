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


from lighteval.tasks.tasks.mmlu_pro import mmlu_pro_prompt_function


def _line(num_options, answer_index=0, question="What is 2 + 2?"):
    return {
        "question": question,
        "options": [f"option {i}" for i in range(num_options)],
        "answer_index": answer_index,
    }


def test_mmlu_pro_choices_track_option_count():
    # MMLU-Pro questions have up to 10 options, so choices must be one letter per
    # option. Regression for choices=ascii_uppercase[: len(choices)], where
    # `choices` was the joined prompt string, returning all 26 letters A-Z.
    doc = mmlu_pro_prompt_function(_line(10, answer_index=3), task_name="mmlu_pro")
    assert doc.choices == list("ABCDEFGHIJ")
    assert len(doc.choices) == 10


def test_mmlu_pro_instruction_lists_real_letters():
    # The instruction must enumerate the actual answer letters, not a hardcoded
    # "ABCD", otherwise the model is told only A-D are valid for a 10-option
    # question.
    doc = mmlu_pro_prompt_function(_line(10), task_name="mmlu_pro")
    assert "one of ABCDEFGHIJ" in doc.query
    assert "one of ABCD." not in doc.query


def test_mmlu_pro_four_options():
    # Letters track the option count for shorter questions too.
    doc = mmlu_pro_prompt_function(_line(4), task_name="mmlu_pro")
    assert doc.choices == list("ABCD")
    assert "one of ABCD." in doc.query


def test_mmlu_pro_gold_index_letter_alignment():
    # gold_index still points at the correct answer letter.
    doc = mmlu_pro_prompt_function(_line(10, answer_index=3), task_name="mmlu_pro")
    assert doc.gold_index == 3
    assert doc.choices[doc.gold_index] == "D"


def test_mmlu_pro_choices_block_enumerates_each_option():
    doc = mmlu_pro_prompt_function(_line(10), task_name="mmlu_pro")
    for letter in "ABCDEFGHIJ":
        assert f"{letter}: " in doc.query
