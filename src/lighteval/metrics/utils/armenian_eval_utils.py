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

import ast
import re


def extract_answer_for_numeric_choices(text):
    pattern = r"answer is \(?([1-7])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None


def extract_again_for_numeric_choices(text):
    match = re.search(r".*[aA]nswer:\s*([1-7])", text)
    if match:
        return match.group(1)
    else:
        return None


def extract_final_for_numeric_choices(text):
    pattern = r"\b[1-7]\b(?!.*\b[1-7]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def extract_answer_for_numeric_answer(text):
    pattern = r"answer is \(?(\d+)\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None


def extract_again_for_numeric_answer(text):
    match = re.search(r".*[aA]nswer:\s*(\d+)", text)
    if match:
        return match.group(1)
    else:
        return None


def extract_final_for_numeric_answer(text):
    pattern = r"\b(\d+)\b(?!.*\b(\d+)\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None


def extract_answer_for_letter_choices(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None


def extract_again_for_letter_choices(text):
    match = re.search(r".*[aA]nswer:\s*([A-J])", text, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None


def extract_final_for_letter_choices(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0)
    else:
        return None


def extract_correct_answers_list(text):
    pattern = r"\[([^\]]+)\]"

    match = re.search(pattern, text)
    if match:
        extracted_list_str = "[" + match.group(1) + "]"
        try:
            extracted_list = ast.literal_eval(extracted_list_str)
            return extracted_list
        except (ValueError, SyntaxError):
            return []
    else:
        return []


def extract_correct_answers_dict(text):
    pattern = r"\{.*?\}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            extracted_dict = ast.literal_eval(match.group())
            return list(extracted_dict.values())
        except Exception:
            return []
    else:
        return []
