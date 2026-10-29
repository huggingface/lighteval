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

import json
import re

import numpy as np
import torch

from lighteval.metrics.imports.bert_scorer import BERTScorer
from lighteval.metrics.utils.armenian_eval_utils import (
    extract_again_for_letter_choices,
    extract_again_for_numeric_answer,
    extract_again_for_numeric_choices,
    extract_answer_for_letter_choices,
    extract_answer_for_numeric_answer,
    extract_answer_for_numeric_choices,
    extract_correct_answers_dict,
    extract_correct_answers_list,
    extract_final_for_letter_choices,
    extract_final_for_numeric_answer,
    extract_final_for_numeric_choices,
)
from lighteval.metrics.utils.metric_utils import (
    SampleLevelComputation,
    SampleLevelMetric,
    SamplingMethod,
)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import Doc


class NERSpanComputation(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs):
        gold_entities = [(t.lower().strip(), ty.lower().strip()) for (t, ty) in doc.specific.get("gold_entities", [])]

        if hasattr(model_response, "text_post_processed") and model_response.text_post_processed:
            pred_text = model_response.text_post_processed[0]
        elif hasattr(model_response, "text") and model_response.text:
            pred_text = model_response.text[0]
        else:
            pred_text = ""

        pred_entities = self.parse_pred(pred_text)

        pred_by_tag = {}
        for p_text, p_tag in pred_entities:
            pred_by_tag.setdefault(p_tag, []).append(p_text)

        correct = 0
        for g_text, g_tag in gold_entities:
            if g_tag in pred_by_tag:
                for p_text in pred_by_tag[g_tag]:
                    if self.char_overlap_ratio(g_text, p_text) >= 0.5:
                        correct += 1
                        break

        return correct / len(gold_entities) if gold_entities else 0.0

    def parse_pred(self, pred: str):  # noqa: C901
        pred = pred.strip()
        if pred.startswith("```"):
            pred = pred.strip("`").lstrip("json").strip()

        entities = []

        bracket_count = 0
        start_idx = -1
        json_candidates = []

        for i, char in enumerate(pred):
            if char == "[":
                if bracket_count == 0:
                    start_idx = i
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
                if bracket_count == 0 and start_idx != -1:
                    json_candidates.append(pred[start_idx : i + 1])
                    start_idx = -1

        for json_str in json_candidates:
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    for e in parsed:
                        if isinstance(e, dict) and "entity" in e and "tag" in e:
                            entity_text = e["entity"].strip()
                            entity_tag = e["tag"].strip()
                            if entity_text and entity_tag:
                                entities.append((entity_text.lower(), entity_tag.lower()))
                    if entities:
                        break
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        if not entities:
            try:
                parsed = json.loads(pred)
                if isinstance(parsed, list):
                    for e in parsed:
                        if "entity" in e and "tag" in e:
                            entity_text = e["entity"].strip()
                            entity_tag = e["tag"].strip()
                            if entity_text and entity_tag:
                                entities.append((entity_text.lower(), entity_tag.lower()))
            except Exception:
                parts = [p for p in pred.replace(";", ",").split(",") if ":" in p]
                for part in parts:
                    try:
                        t, tag = part.split(":", 1)
                        t = t.strip()
                        tag = tag.strip()
                        if t and tag:
                            entities.append((t.lower(), tag.lower()))
                    except ValueError:
                        continue

        return entities

    def char_overlap_ratio(self, a: str, b: str) -> float:
        gold_words = a.split()
        pred_words = b.split()

        matches = 0
        total = sum(len(w) for w in gold_words)

        for g, p in zip(gold_words, pred_words):
            for gc, pc in zip(g, p):
                if gc == pc:
                    matches += 1

        if total == 0:
            return 0.0
        return matches / total


class UDPosComputation(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs):
        gold_entities = [
            (str(t).lower().strip(), str(ty).lower().strip())
            for (t, ty) in doc.specific.get("gold_entities", [])
            if t is not None and ty is not None
        ]

        if hasattr(model_response, "text_post_processed") and model_response.text_post_processed:
            pred_text = model_response.text_post_processed[0]
        elif hasattr(model_response, "text") and model_response.text:
            pred_text = model_response.text[0]
        else:
            pred_text = ""

        pred_entities = []

        m_hy = re.search(r"խոսքի մասն է\s+([^\s.,;]+)", pred_text, flags=re.IGNORECASE)
        m_en = re.search(
            r"part of speech of this word is\s+([^\s.,;]+)",
            pred_text,
            flags=re.IGNORECASE,
        )

        if m_hy:
            tag = m_hy.group(1).lower().strip()
        elif m_en:
            tag = m_en.group(1).lower().strip()
        else:
            tag = None

        if tag:
            if gold_entities:
                word = gold_entities[0][0]
            else:
                word = ""
            pred_entities.append((word.lower(), tag))

        correct = 0
        for g_text, g_tag in gold_entities:
            for _, p_tag in pred_entities:
                if g_tag == p_tag:
                    correct += 1
                    break

        return {
            "correct": correct,
            "gold": len(gold_entities),
            "pred": len(pred_entities),
        }


def pos_agg(items):
    total_correct = sum(i["correct"] for i in items)
    total_gold = sum(i["gold"] for i in items)
    return total_correct / total_gold if total_gold > 0 else 0.0


class BertScoreArm(SampleLevelComputation):
    def __init__(self):
        self.scorer = BERTScorer(
            model_type="Metric-AI/armenian-text-embeddings-1",
            num_layers=9,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs):
        pred = (
            model_response.text_post_processed[0]
            if hasattr(model_response, "text_post_processed") and model_response.text_post_processed
            else (model_response.text[0] if hasattr(model_response, "text") and model_response.text else "")
        ).strip()

        golds = doc.choices or []
        if not golds:
            return 0.0

        P, R, F = self.scorer.score([pred], [golds])
        return F[0].item()


class ArmenianExamComputation(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs):  # noqa: C901

        task_type = doc.specific.get("task_type")
        true_answer = doc.specific.get("label")
        text = model_response.final_text[0]
        score = 0.0

        if task_type == 1:
            extracted_answer = extract_answer_for_numeric_choices(text)
            if extracted_answer is None:
                extracted_answer = extract_again_for_numeric_choices(text)
            if extracted_answer is None:
                extracted_answer = extract_final_for_numeric_choices(text)

            if extracted_answer is not None and extracted_answer == true_answer[0]:
                score = 0.25

        elif task_type == 2:
            extracted_answer = extract_correct_answers_list(text)
            extracted_answer = [str(i) for i in extracted_answer]
            if extracted_answer is not None and set(true_answer) == set(extracted_answer):
                score = 0.25

        elif task_type == 3:
            extracted_answer = extract_correct_answers_list(text)
            chsy_score = 0
            for a, b in zip(extracted_answer, true_answer):
                if a == b:
                    chsy_score += 0.25
                elif a != b and a != "Չգիտեմ":
                    chsy_score -= 0.25
            if chsy_score < 0:
                chsy_score = 0
            score = chsy_score

        elif task_type == 4:
            extracted_answer = extract_correct_answers_dict(text)
            extracted_answer = [str(i) for i in extracted_answer]
            if extracted_answer == true_answer:
                score = 0.25

        elif task_type == 5:
            extracted_answer = extract_correct_answers_list(text)
            extracted_answer = [str(i) for i in extracted_answer]
            if extracted_answer == true_answer:
                score = 0.25

        elif task_type == 6:
            extracted_answer = extract_answer_for_letter_choices(text)
            if extracted_answer is None:
                extracted_answer = extract_again_for_letter_choices(text)
            if extracted_answer is None:
                extracted_answer = extract_final_for_letter_choices(text)

            if extracted_answer is not None and extracted_answer == true_answer[0]:
                score = 0.25

        elif task_type == 7:
            extracted_answer = extract_answer_for_numeric_answer(text)
            if extracted_answer is None:
                extracted_answer = extract_again_for_numeric_answer(text)
            if extracted_answer is None:
                extracted_answer = extract_final_for_numeric_answer(text)

            if extracted_answer is not None and extracted_answer == true_answer[0]:
                score = 0.25

        return score


class MMLUProComputation(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs):
        true_answer = doc.specific.get("answer")

        if true_answer is None:
            true_answer = "ABCDEFGHIJ"[doc.gold_index]

        text = model_response.final_text[0]

        extracted_answer = extract_answer_for_letter_choices(text)

        if extracted_answer is None:
            extracted_answer = extract_again_for_letter_choices(text)

        if extracted_answer is None:
            extracted_answer = extract_final_for_letter_choices(text)

        if extracted_answer is not None and extracted_answer == true_answer:
            return 1.0

        return 0.0


class GeneralizedExactMatch(SampleLevelComputation):
    def __init__(self, extractors: list, choice_map: str = None):
        self.extractors = extractors
        self.choice_map = choice_map

    def compute(self, doc, model_response, **kwargs):
        true_answer = doc.specific.get("answer")

        if true_answer is None and self.choice_map is not None:
            true_answer = self.choice_map[doc.gold_index]

        if true_answer is None:
            return 0.0

        text = model_response.final_text[0]
        extracted_answer = None

        for extractor_fn in self.extractors:
            extracted_answer = extractor_fn(text)
            if extracted_answer is not None:
                break

        if extracted_answer is not None and str(extracted_answer).strip() == str(true_answer).strip():
            return 1.0

        return 0.0


ner_span_metric = SampleLevelMetric(
    metric_name="ner_accuracy",
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=NERSpanComputation(),
    corpus_level_fn=np.mean,
    higher_is_better=True,
)
pos_metric = SampleLevelMetric(
    metric_name="ud_pos_regex_acc",
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=UDPosComputation(),
    corpus_level_fn=pos_agg,
    higher_is_better=True,
)
bert_score_arm = SampleLevelMetric(
    metric_name="bert_score_arm",
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=BertScoreArm(),
    corpus_level_fn=np.mean,
    higher_is_better=True,
)
armenian_exam_metric = SampleLevelMetric(
    metric_name="armenian_exam_score",
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=ArmenianExamComputation(),
    corpus_level_fn=np.sum,
    higher_is_better=True,
)
armenian_mmlu_pro_metric = SampleLevelMetric(
    metric_name="armenian_mmlu_pro_score",
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=MMLUProComputation(),
    corpus_level_fn=np.mean,
    higher_is_better=True,
)
armenian_mcqa_metric = SampleLevelMetric(
    metric_name="exact_match_mcqa",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=GeneralizedExactMatch(
        extractors=[
            extract_answer_for_letter_choices,
            extract_again_for_letter_choices,
            extract_final_for_letter_choices,
        ]
    ),
    corpus_level_fn=np.mean,
)
