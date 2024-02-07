"""
Custom evaluation tasks for lighteval
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, Union


class Metrics(Enum):
    any_target_loglikelihood_acc = auto()
    bert_score = auto()
    bias = auto()
    bits_per_byte = auto()
    bleu = auto()
    bleu_1 = auto()
    bleu_4 = auto()
    byte_perplexity = auto()
    chrf = auto()
    code_eval_APPS = auto()
    code_eval_HE = auto()
    copyright = auto()
    disinformation = auto()
    exact_match = auto()
    exact_set_match = auto()
    extractiveness = auto()
    f1_from_bags = auto()
    f1_quasi = auto()
    f1_sequence = auto()
    f1_set_match = auto()
    faithfulness = auto()
    iou_set_match = auto()
    log_prob = auto()
    loglikelihood_acc = auto()
    loglikelihood_acc_norm = auto()
    loglikelihood_acc_norm_nospace = auto()
    loglikelihood_acc_norm_single_token = auto()
    loglikelihood_acc_single_token = auto()
    loglikelihood_f1 = auto()
    loglikelihood_f1_single_token = auto()
    math_quasi_exact_match = auto()
    mc_taco = auto()
    mcc = auto()
    mcc_single_token = auto()
    mrr = auto()
    mrr_single_token = auto()
    multi_fi_numeric = auto()
    one_choice_loglikelihood_acc = auto()
    perfect_exact_match = auto()
    prediction_perplexity = auto()
    prefix_exact_match = auto()
    prefix_quasi_exact_match = auto()
    quasi_exact_match = auto()
    quasi_exact_match2 = auto()
    ranking = auto()
    recall_at_1_single_token = auto()
    recall_at_2_single_token = auto()
    recall_at_1 = auto()
    recall_at_2 = auto()
    rouge = auto()
    rouge_1 = auto()
    rouge_2 = auto()
    rouge_l = auto()
    target_perplexity = auto()
    ter = auto()
    toxicity = auto()
    truthfulqa_mc_metrics = auto()
    word_perplexity = auto()

    def __str__(self):
        return self.name.replace("_at_", "@")


NEEDS_GENERATION_ONLY = [
    "perfect_exact_match",
    "exact_match",
    "quasi_exact_match",
    "quasi_exact_match2",
    "prefix_exact_match",
    "prefix_quasi_exact_match",
    "math_quasi_exact_match",
    "iou_set_match",
    "exact_set_match",
    "f1_sequence",
    "f1_quasi",
    "f1_set_match",
    "f1_from_bags",
    "chrf",
    "ter",
    "rouge",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "faithfulness",
    "extractiveness",
    "bert_score",
    "bleu",
    "bleu_1",
    "bleu_4",
    "bias",
    "toxicity",
    "code_eval_HE",
    "code_eval_APPS",
    "copyright",
]


@dataclass(unsafe_hash=True)
class CustomEvaluationTask:
    name: str
    prompt_function: str
    hf_repo: str
    hf_subset: str
    metric: Tuple[Union[str, Metrics]]
    hf_avail_splits: Optional[Tuple[str]] = None
    evaluation_splits: Optional[Tuple[str]] = None
    few_shots_split: Optional[str] = None
    few_shots_select: Optional[str] = None
    generation_size: int = -1
    stop_sequence: Optional[Tuple[str]] = None
    output_regex: Optional[str] = None

    frozen: bool = False
    suite: Optional[Tuple[str]] = None  # we use this to know if we should use a custom lighteval or bigcode task

    def __post_init__(self):
        self.metric = [str(m) for m in self.metric]
        if self.suite is None:
            self.suite = ["custom"]
        if self.hf_avail_splits is None:
            self.hf_avail_splits = ["train", "validation", "test"]
        if self.evaluation_splits is None:
            self.evaluation_splits = ["validation"]
        if self.stop_sequence is None:
            self.stop_sequence = ["\n"]

        # Convert list to tuple for hashing
        self.metric = tuple(self.metric)
        self.hf_avail_splits = tuple(self.hf_avail_splits) if self.hf_avail_splits else None
        self.evaluation_splits = tuple(self.evaluation_splits) if self.evaluation_splits else None
        self.suite = tuple(self.suite) if self.suite else None
        self.stop_sequence = tuple(self.stop_sequence) if self.stop_sequence else None


@dataclass(unsafe_hash=True)
class BigCodeEvaluationTask:
    name: str
    bigcode_task: str
    bigcode_task_kwargs: Optional[dict] = None
    n_samples: int = 1
    prefix: Optional[str] = None

    suite: Tuple[str] = None

    def __post_init__(self):
        if self.suite is None:
            self.suite = ("bigcode",)

        # Convert list to tuple for hashing
        self.suite = tuple(self.suite)
