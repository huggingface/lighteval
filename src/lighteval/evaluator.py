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

# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/evaluator.py
# Adds support for Prompt templates

import collections
import copy
from typing import Dict, Union

from pytablewriter import LatexTableWriter, MarkdownTableWriter

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.logging.hierarchical_logger import hlog
from lighteval.models.base_model import BaseModel
from lighteval.models.tgi_model import ModelClient
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.requests import Doc, Request, RequestType, TaskExampleId


def evaluate(  # noqa: C901
    lm: Union[BaseModel, ModelClient],
    requests_dict: Dict[RequestType, list[Request]],
    docs: Dict[TaskExampleId, Doc],
    task_dict: Dict[str, LightevalTask],
    override_bs: int = None,
    evaluation_tracker: EvaluationTracker = None,
) -> EvaluationTracker:
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param fewshot_dict: dict
        Number of examples in few-shot context, per task
    :param max_samples: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param prompt_template: str
        A template string that can be used to wrap the prompt, for examples see "templates/prompts"
    :param num_fewshot_seeds: int
        Number of times the experiment is run (with a different sampling of the few-shot prompts) - see HELM.
    :return
        Dictionary of results
    """
    # A request output tupe is a Tuple where the first element is the index of
    # the request for one document of one task i.e.
    # task: "arc_easy", doc: "0"# request: "0" -> request_index = 0,
    # We can have multiple request per doc for multi choice tasks for example.

    # all responses for each (task, doc)
    RequestIndexModelResponseTuple = collections.namedtuple(
        "RequestIndexModelResponseTuple", ["request_index", "model_response"]
    )
    example_id_response_dict: dict[TaskExampleId, list[RequestIndexModelResponseTuple]] = collections.defaultdict(list)

    for request_type, requests in requests_dict.items():
        hlog(f"Running {request_type} requests")
        # These are all the request type from the request factory at the moment
        if request_type == RequestType.LOGLIKELIHOOD:
            full_resps = lm.loglikelihood(requests, override_bs=override_bs)
        elif request_type == RequestType.LOGLIKELIHOOD_SINGLE_TOKEN:
            full_resps = lm.loglikelihood_single_token(requests, override_bs=override_bs)
        elif request_type == RequestType.GREEDY_UNTIL:
            full_resps = lm.greedy_until(requests, override_bs=override_bs)
        elif request_type == RequestType.GREEDY_UNTIL_WITH_LOGITS:
            full_resps = lm.greedy_until_with_logits(requests, override_bs=override_bs)
        elif request_type == RequestType.LOGLIKELIHOOD_ROLLING:
            full_resps = lm.loglikelihood_rolling(requests, override_bs=override_bs)
        elif request_type == RequestType.GREEDY_UNTIL_MULTI_TURN:
            full_resps = lm.greedy_until_multi_turn(requests, override_bs=override_bs)
        else:
            raise NotImplementedError(f"Request type {request_type} not supported")

        for full_resp, request in zip(full_resps, requests):
            cur_resp = copy.deepcopy(full_resp)
            example_id_response_dict[TaskExampleId(request.task_name, request.example_index)].append(
                RequestIndexModelResponseTuple(request.request_index, cur_resp)
            )

    # ===== unpack results and sort back in order and return control to Task =====
    for task_example_id, prediction_list in example_id_response_dict.items():
        # ===== Unpack the request =====
        prediction_list.sort(
            key=lambda x: x.request_index
        )  # When we use Loglikelihood for several tokens we have all the options here
        model_responses = [x.model_response for x in prediction_list]
        cur_task_name = task_example_id.task_name.rsplit("|", 1)[0]

        task: LightevalTask = task_dict[cur_task_name]
        doc: Doc = docs[task_example_id]

        if doc.instruction is None:
            doc.instruction = ""

        # using a deep copy here because process results pops from the model responses
        metrics = task.process_results(doc, copy.deepcopy(model_responses))

        # Remove the user_prompt from the metrics in case of llm-as-judge metric
        if "user_prompt" in metrics:
            user_prompt = metrics["user_prompt"]
            del metrics["user_prompt"]
        else:
            user_prompt = None
        if "judgement" in metrics:
            judgement = metrics["judgement"]
            del metrics["judgement"]
        else:
            judgement = None

        evaluation_tracker.metrics_logger.log(task_example_id.task_name, metrics)
        evaluation_tracker.details_logger.log(
            task_example_id.task_name, task, doc, model_responses, metrics, (user_prompt, judgement)
        )

    return evaluation_tracker


def make_results_table(result_dict):
    """Generate table of results."""
    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k in sorted(result_dict["results"].keys()):
        dic = result_dict["results"][k]
        version = result_dict["versions"][k] if k in result_dict["versions"] else ""
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m, "%.4f" % v, "±", "%.4f" % se])
            else:
                values.append([k, version, m, "%.4f" % v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    return md_writer.dumps()
