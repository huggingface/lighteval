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


from inspect_ai import Epochs, Task, eval, task
from inspect_ai.dataset import hf_dataset
from inspect_ai.solver import generate, system_message

from lighteval.tasks import default_tasks
from lighteval.tasks.lighteval_task import LightevalTaskConfig_inspect as LightevalTaskConfig


@task
def get_task(lighteval_task_config: LightevalTaskConfig):
    name = lighteval_task_config.name
    sample_fields = lighteval_task_config.prompt_function

    dataset_repo = lighteval_task_config.dataset_repo
    dataset_subset = lighteval_task_config.dataset_subset
    dataset_split = lighteval_task_config.dataset_split

    system_prompt = lighteval_task_config.system_prompt

    dataset = hf_dataset(dataset_repo, name=dataset_subset, split=dataset_split, sample_fields=sample_fields)
    solvers = lighteval_task_config.solvers or [
        system_message(system_prompt),
        generate(cache=True),
    ]
    scorers = lighteval_task_config.scorers
    epochs = lighteval_task_config.epochs
    epochs_reducer = lighteval_task_config.epochs_reducer

    return Task(dataset=dataset, solver=solvers, scorer=scorers, name=name, epochs=Epochs(epochs, epochs_reducer))


model_args = {
    "max-tokens", "system-message", "temperature", "top-p", "top-k", "frequence-penalty", 
    "presence-penalty", "logit-bias", "seed", "stop-seqs", "num-choices", "best-of", "log-probs", "top-logprobs",
    "cache-prompt", "reasoning-effort", "reasoning-tokens", "reasoning-history", "response-format", "parallel-tool-calls", "max-tool-output",
    "internal-tools", "max-retries", "timeout"
}


def main():
    MODEL = ["openai/gpt-4o"]
    all_tasks = [
        #default_tasks.gsm8k_lighteval,
        #default_tasks.aime25,
        #default_tasks.aime24,
        #default_tasks.math_500,
        default_tasks.gsm_plus,
        #default_tasks.gpqa_diamond,
        #default_tasks.gpqa_extended,
        #default_tasks.gpqa_main,
    ]  # default_tasksifeval]
    all_tasks = [get_task(task) for task in all_tasks]

    # eval_set(all_tasks, model=MODEL, display="rich", limit=10, max_tasks=3, bundle_dir="./log_static", log_dir="./log_dynamic-1")

    eval(all_tasks, model=MODEL, display="rich", limit=10, max_tasks=3, log_probs=True)


if __name__ == "__main__":
    main()
