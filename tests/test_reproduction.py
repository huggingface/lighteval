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

from lighteval.evaluator import EvaluationTracker, evaluate
from lighteval.metrics.metrics import Metrics
from lighteval.models.base_model import BaseModelConfig
from lighteval.models.model_config import EnvConfig
from lighteval.models.model_loader import load_model
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.requests import Doc, RequestType, TaskExampleId
from lighteval.tasks.tasks_prompt_formatting import arc


def run_evaluation() -> dict:
    evaluation_tracker = EvaluationTracker()
    model_config = BaseModelConfig("hf-internal-testing/tiny-random-LlamaForCausalLM")
    model, _ = load_model(config=model_config, env_config=EnvConfig())

    task_config = LightevalTaskConfig("test", arc, "", "", [Metrics.loglikelihood_acc])
    task = LightevalTask("test", task_config)
    task_dict = {"custom|test": task}
    evaluation_tracker.task_config_logger.log(task_dict)
    doc = Doc("Who is the GOAT?", ["CR7", "Messi", "Pele", "Zizou"], gold_index=3)
    doc.ctx = "Who is the GOAT?"
    docs = {TaskExampleId("custom|test|0", "0_0"): doc}
    requests_dict = task.construct_requests(doc, doc.ctx, "0_0", "custom|test|0")
    # Because `task.construct_requests` has empty entries causing error in `evaluate`` currently.
    requests_dict = {RequestType.LOGLIKELIHOOD: requests_dict[RequestType.LOGLIKELIHOOD]}

    evaluation_tracker = evaluate(
        lm=model,
        requests_dict=requests_dict,
        docs=docs,
        task_dict=task_dict,
        override_bs=1,
        evaluation_tracker=evaluation_tracker,
    )
    evaluation_tracker.metrics_logger.aggregate(task_dict=task_dict)
    evaluation_tracker.details_logger.aggregate()
    model.cleanup()
    return evaluation_tracker.generate_final_dict()


def test_reproduction():
    result_1 = run_evaluation()
    del result_1["config_general"]["start_time"]
    result_2 = run_evaluation()
    del result_2["config_general"]["start_time"]
    assert result_2 == result_1
