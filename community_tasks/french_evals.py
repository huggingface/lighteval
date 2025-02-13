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

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

This module implements tasks for the french specific datasets
See : https://huggingface.co/fr-gouv-coordination-ia
"""

import os
import random
from pathlib import Path
import re
import string

import numpy as np
from aenum import extend_enum

import lighteval.tasks.extended.ifeval.instructions_registry as instructions_registry
from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.extended.ifeval.main import ifeval_metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list

from typing import Callable
from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase

from lighteval.utils.language import Language

# From HELM french version normalizer
def helm_normalizer_fr(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Adapted for pr_fouras dataset"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(le |la |les |l |un |une |des )", "", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def homogeneize_numbers(text: str) -> str:
        """Casts text to float to test if it's a number, then casts back to string.
        This allows equal numbers formatted differently (1.0 vs 1 for ex) to be considered
        equal. This comes from Harness DROP - check if it causes a discrep in QuAC
        """
        try:
            return str(float(text.replace(',', '.')))
        except ValueError:
            return text

    def remove_punc(text: str) -> str:
        punctuation = string.punctuation
        # Regex: removes all punctuation except when it is between two digits (e.g., 54,7)
        regex = rf'(?<!\d)[{punctuation}](?!\d)|(?<=\D)[{punctuation}]+|[{punctuation}]+(?=\D)|(?<=^)[{punctuation}]+|[{punctuation}]+(?=$)'
        return re.sub(regex, "", text)

    def remove_plural_feminine(text: str) -> str:
        # Regex : supprime les 's', 'x', 'e', 'es' en fin de mot
        regex = r'\b(\w+?)(?:es|e|s|x)\b'
        return re.sub(regex, r'\1', text)

    def lower(text: str) -> str:
        return text.lower()

    def _tokenize(text):
        text = lower(text)
        text = re.sub(r"['\'’`]", " ", text)
        text = remove_articles(text)
        return re.split(" ", text)

    tokens = [remove_plural_feminine(remove_punc(white_space_fix(homogeneize_numbers(t)))) for t in _tokenize(text)]
    return " ".join([t for t in tokens if t != ""]).strip()

## Math normalizer
def math_normalizer(text: str) -> str:
    """Custom to bac-fr dataset"""

    def space_digit(text: str) -> str:
        return re.sub(r"(\d+[.]*\d*)", r" \1 ", text)

    def homogeneize_numbers(text: str) -> str:
        """Casts text to float to test if it's a number, then casts back to string.
        This allows equal numbers formatted differently (1.0 vs 1 for ex) to be considered
        equal. This comes from Harness DROP - check if it causes a discrep in QuAC
        """
        try:
            return str(float(text.replace(',', '.')))
        except ValueError:
            return text

    def lower(text: str) -> str:
        return text.lower()

    def remove_punc(text: str) -> str:
        punctuation = "'`*\"~.()$"
        # Regex: removes punctuation at start & end 
        regex = rf'(?<=^)[{punctuation}]+|[{punctuation}]+(?=$)'
        return re.sub(regex, "", text)

    def remove_special_chars(text: str) -> str:
        # Regex : supprime ',', '"', "'", '`', '\n' et '\t' mais pas 'n' et 't'
        regex = r"[\"'`]|(?:\n)|(?:\t)|(?:\r)|(?:\\n)|(?:\\t)|(?:\\r)"
        return re.sub(regex, "", text)

    def comma_to_point(text: str) -> str:
        return re.sub(r"(\d+),(\d+)", r"\1.\2", text)

    def clean_math_expressions(text: str) -> str:
        # Étape 1 : Supprimer toutes les accolades '{' et '}'
        text = re.sub(r"[{}]", "", text)
        # Étape 2 : Supprimer les commandes \boxed, \fbox et \underline avec plusieurs \
        regex = r"(\\*)(boxed|fbox|underline)"
        text = re.sub(regex, "", text)
    
        return text
    

    def _tokenize(text):
        text = lower(text)
        text = comma_to_point(text)
        text = clean_math_expressions(text)
        text = remove_punc(text)
        text = remove_special_chars(space_digit(text))
        return text.split()

    
    tokens = [homogeneize_numbers(t) for t in _tokenize(text)]
    return "".join([t for t in tokens if t != ""]).strip()
    


## custom Metric for bac-fr and pr-fouras
class BacPrefixSuffixExactMatch:
    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = max,
        normalize_gold: Callable[[str], str] | None = None,
        normalize_pred: Callable[[str], str] | None = None,
    ):
        """An exact match class.

        Args:
            aggregation_function (callable, optional): How to aggregate the item results. Defaults to max.
                Used if there are several golds or predictions on which scores were computed.
            normalize_gold (callable, optional): Function to use to normalize the reference strings.
                Defaults to None if no normalization is applied.
            normalize_pred (callable, optional): Function to use to normalize the predicted strings.
                Defaults to None if no normalization is applied.
        """
        self.aggregation_function = aggregation_function
        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        """Computes the metric over a list of golds and predictions for one single sample.

        Args:
            golds (list[str]): Reference targets
            predictions (list[str]): Predicted strings

        Returns:
            float: Aggregated score over the current sample's items.
        """
        
        results = []
        # We might need to flatten golds if they are a list of lists
        for gold in golds:
            for pred in predictions:
                results.append(self.compute_one_item(gold=gold, pred=pred))
        return self.aggregation_function(results)

    def compute_one_item(
        self,
        gold: str,
        pred: str,
    ) -> float:
        """Compares two strings only.

        Args:
            gold (str): One of the possible references
            pred (str): One of the possible predictions

        Returns:
            float: The exact match score. Will be 1 for a match, 0 otherwise.
        """
        if not pred:
            return 0

        gold = gold.strip()
        pred = pred.strip()
        
        if self.normalize_gold:
            gold = self.normalize_gold(gold)
        if self.normalize_pred:
            pred = self.normalize_pred(pred)

        if pred.startswith(gold) or pred.endswith(gold) or gold == pred: # prefix em # suffix em # strict em
            return 1 
        #last chance!!
        gold_lc = gold.split('=')[-1]
        pred_lc = pred.split('=')[-1]
        if pred_lc.startswith(gold_lc) or pred_lc.endswith(gold_lc) or gold_lc == pred_lc: 
            return 1 
        return 0

class PFourasPrefixSuffixExactMatch:
    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = max,
        normalize_gold: Callable[[str], str] | None = None,
        normalize_pred: Callable[[str], str] | None = None,
    ):
        """An exact match class.

        Args:
            aggregation_function (callable, optional): How to aggregate the item results. Defaults to max.
                Used if there are several golds or predictions on which scores were computed.
            normalize_gold (callable, optional): Function to use to normalize the reference strings.
                Defaults to None if no normalization is applied.
            normalize_pred (callable, optional): Function to use to normalize the predicted strings.
                Defaults to None if no normalization is applied.
        """
        self.aggregation_function = aggregation_function
        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        """Computes the metric over a list of golds and predictions for one single sample.

        Args:
            golds (list[str]): Reference targets
            predictions (list[str]): Predicted strings

        Returns:
            float: Aggregated score over the current sample's items.
        """
        # Split the multi-responses
        predictions = "".join(predictions).split('/')
        
        results = []
        # We might need to flatten golds if they are a list of lists
        for gold in golds:
            for pred in predictions:
                results.append(self.compute_one_item(gold=gold, pred=pred))
        return self.aggregation_function(results)

    def compute_one_item(
        self,
        gold: str,
        pred: str,
    ) -> float:
        """Compares two strings only.

        Args:
            gold (str): One of the possible references
            pred (str): One of the possible predictions

        Returns:
            float: The exact match score. Will be 1 for a match, 0 otherwise.
        """
        if not pred:
            return 0

        gold = gold.strip()
        pred = pred.strip()

        if self.normalize_gold:
            gold = self.normalize_gold(gold)
        if self.normalize_pred:
            pred = self.normalize_pred(pred)

        if pred.startswith(gold): # prefix em
            return 1 
        if pred.endswith(gold): # suffix em
            return 1
        return 1 if gold == pred else 0 # strict em


bac_prefixsuffix_quasi_exact_match = SampleLevelMetric(
    metric_name="bac-fr-qem",
    sample_level_fn=BacPrefixSuffixExactMatch(
        normalize_gold=math_normalizer,
        normalize_pred=math_normalizer,
    ).compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

pfouras_prefixsuffix_quasi_exact_match = SampleLevelMetric(
    metric_name="pr-fouras-qem",
    sample_level_fn=PFourasPrefixSuffixExactMatch(
        normalize_gold=helm_normalizer_fr,
        normalize_pred=helm_normalizer_fr,
    ).compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

# Ifeval-fr prompt function
def prompt_ifeval_fr(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[""],
        gold_index=0,
        instruction="",
        specific={"instructions_id_list": line["instruction_id_list"], "kwargs": line["kwargs"]},
    )


# qpqa-fr prompt function
def prompt_gpqa_fr(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Réponse incorrecte 1"], line["Réponse incorrecte 2"], line["Réponse incorrecte 3"]]
    choices.insert(gold_index, line["Réponse correcte"])

    instruction = "Choisissez la réponse correcte aux questions suivantes.\n\n"

    query = f"Question: {line['Question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, choices)])
    query += "Réponse: "
    return Doc(
        task_name=task_name,
        query=f"{instruction}{query}",
        choices=LETTER_INDICES[: len(choices)],
        gold_index=gold_index,
        instruction=instruction,
    )


# BAC-fr prompt function
def prompt_bac_fr(line, task_name: str = None):
    prompt = f"Enoncé: {line['enonce']}\n"
    if line['instruction'] is not None:
        prompt += f"{line['instruction']}\n"
    prompt += "Réponse: "
    if line["choix"] is not None:  # Multichoice evaluation
        # prompt += "\n".join([f"{LETTER_INDICES[ix]}.{choix}" for ix, choix in enumerate(line["choix"])])
        return Doc(
            task_name=task_name,
            query=prompt,
            choices=as_list(line["choix"]),
            gold_index=as_list(line["choix"]).index(line["choix correct"]),
            instruction="",
        )
    else:
        return Doc(task_name=task_name, query=prompt, choices=[line["reponse"]], gold_index=0, instruction="")

# pr-fouras prompt function
def prompt_pr_fouras(line, task_name: str = None):
    instruction = "Trouver la réponse exacte à l'énigme. Vous pouvez proposer plusieurs réponses possibles. Chaque réponse doit être séparée d'un caractère /.\n Exemple:\n Enigme: Plus je travaille, plus je raccourcis. Qui suis-je ?\nRéponses: Des ciseaux / Une paire de ciseaux / Une bougie / Une gomme / Une personne agée / Un vieux / Un vêtement / Un sécateur / Un clou / Une pause.\n\n"
    prompt = instruction+f"Enigme: {line['enigme']}\n"
    prompt += "Réponses: "
    return Doc(task_name=task_name, query=prompt, choices=[line["reponse"]], gold_index=0, instruction="")



DSDIR = Path(os.getenv("DATASETS_DIRECTORY"))
# IFEVal-fr task
ifeval_fr_task = LightevalTaskConfig(
    name="ifeval-fr",
    prompt_function=prompt_ifeval_fr,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["community"],
    hf_repo=str(DSDIR / "IFEval-fr"),
    hf_subset="default",
    metric=[ifeval_metrics],
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=1280,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",  # select your metric in Metrics
)

# GPQA-fr task
gpqa_fr_task = LightevalTaskConfig(
    name="gpqa-fr",
    suite=["community"],
    prompt_function=prompt_gpqa_fr,
    hf_repo=str(DSDIR / "gpqa-fr"),
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metric=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)

# BAC-fr task
bac_fr_task = LightevalTaskConfig(
    name="bac-fr",
    suite=["community"],
    prompt_function=prompt_bac_fr,
    hf_repo=str(DSDIR / "bac-fr"),
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=100,
    metric=[bac_prefixsuffix_quasi_exact_match],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)

# pr-fouras task
pr_fouras_task = LightevalTaskConfig(
    name="pr-fouras",
    suite=["community"],
    prompt_function=prompt_pr_fouras,
    hf_repo=str(DSDIR / "pr-fouras"),
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=100,
    metric=[pfouras_prefixsuffix_quasi_exact_match],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)

# STORE YOUR EVALS
TASKS_TABLE = [ifeval_fr_task, gpqa_fr_task, bac_fr_task, pr_fouras_task]
