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


from unittest.mock import patch

from lighteval.metrics import apply_metric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.tasks.hle import ExtractedAnswer, hle_metrics, hle_text_only


def test_hle_judge_metric_runs_through_apply_metric():
    docs = [hle_text_only({"image": "", "question": f"question {i}", "answer": "42"}, "hle") for i in range(3)]
    responses = [ModelResponse(text=[f"Answer: {answer}\nConfidence: 90%"]) for answer in ["42", "41", "42"]]

    def fake_judge(prompts):
        return [
            ExtractedAnswer(
                extracted_final_answer="",
                reasoning="",
                correct="yes" if "Answer: 42" in prompt[0]["content"] else "no",
                confidence=90,
                strict=True,
            )
            for prompt in prompts
        ]

    with patch("lighteval.metrics.utils.llm_as_judge.JudgeLM._JudgeLM__lazy_load_client", return_value=fake_judge):
        outputs = apply_metric(responses=responses, docs=docs, metrics=[hle_metrics])

    assert [output["accuracy"]["correct"] for output in outputs] == ["yes", "no", "yes"]
    corpus_fn = hle_metrics.get_corpus_aggregations()["accuracy"]
    assert corpus_fn([output["accuracy"] for output in outputs])["accuracy"] == 66.67
