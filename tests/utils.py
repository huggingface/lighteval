

from typing import Optional
from lighteval.evaluator import evaluate
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import GenerateReturn, LoglikelihoodReturn, LoglikelihoodSingleTokenReturn, ModelReturn
from lighteval.tasks.lighteval_task import LightevalTask, create_requests_from_tasks
from lighteval.tasks.requests import GreedyUntilRequest, LoglikelihoodRequest, LoglikelihoodRollingRequest, LoglikelihoodSingleTokenRequest, RequestType, Request
from transformers import AutoTokenizer

class FakeModel(LightevalModel):
    """Dummy model to generate random baselines."""

    def __init__(
        self,
        greedy_until_responses: list[GenerateReturn] = [],
        loglikelihood_responses: list[LoglikelihoodReturn] = [],
        loglikelihood_rolling_responses: list[LoglikelihoodReturn] = [],
        loglikelihood_single_token_responses: list[LoglikelihoodSingleTokenReturn] = [],
    ):
        
        
        self._tokenizer = None
        self.greedy_until_responses = greedy_until_responses
        self.loglikelihood_responses = loglikelihood_responses
        self.loglikelihood_rolling_responses = loglikelihood_rolling_responses
        self.loglikelihood_single_token_responses = loglikelihood_single_token_responses
        
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return self._tokenizer

    @property
    def add_special_tokens(self):
        return False

    @property
    def max_length(self) -> int:
        return 2048

    def greedy_until(
        self, requests: list[GreedyUntilRequest], override_bs: Optional[int] = None
    ) -> list[GenerateReturn]:
        ret_resp, self.greedy_until_resp = self.greedy_until_responses[:len(requests)], self.greedy_until_responses[len(requests):]
        return ret_resp

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodReturn]:
        ret_resp, self.loglikelihood_responses = self.loglikelihood_responses[:len(requests)], self.loglikelihood_responses[len(requests):]
        return ret_resp

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodReturn]:
        ret_resp, self.loglikelihood_rolling_responses = self.loglikelihood_rolling_responses[:len(requests)], self.loglikelihood_rolling_responses[len(requests):]
        return ret_resp

    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenReturn]:
        ret_resp, self.loglikelihood_single_token_responses = self.loglikelihood_single_token_responses[:len(requests)], self.loglikelihood_single_token_responses[len(requests):]
        return ret_resp


def run_fake_task(task: LightevalTask, lm: FakeModel, max_samples: int = 1):
    # We can move these args to the function signature if they are needed
    evaluation_tracker = EvaluationTracker()
    task_dict = {task.name: task}
    evaluation_tracker.task_config_logger.log(task_dict)
    requests, docs = create_requests_from_tasks(task_dict=task_dict, fewshot_dict={task.name: [(0, False)]}, num_fewshot_seeds=1, lm=lm, max_samples=max_samples, evaluation_tracker=evaluation_tracker, use_chat_template=False, system_prompt=None)
    results = evaluate(lm=lm, requests_dict=requests, docs=docs, task_dict={task.name: task}, override_bs=None, evaluation_tracker=evaluation_tracker)
    return results.metrics_logger.metrics_values
    
    

    




