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

import random
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import cycle
from typing import TYPE_CHECKING, Optional

from transformers import AutoTokenizer, PreTrainedTokenizer

from lighteval.logging.hierarchical_logger import hlog_warn
from lighteval.tasks.requests import Doc


if TYPE_CHECKING:
    from lighteval.tasks.lighteval_task import LightevalTask


@dataclass
class FewShotSelectionMethod:
    sorting: str  # sorting method for the overall few shot pool (balanced, random, sequential)
    with_sampling: bool  # samples item randomly from the few shot pool
    fewshotpool_unique: (
        bool
    )  # set to true if you are CERTAIN there is no intersection between the few shot pool and your evaluation set


class FewShotSelection(Enum):
    balanced = FewShotSelectionMethod(sorting="balanced", with_sampling=False, fewshotpool_unique=False)
    random = FewShotSelectionMethod(sorting="random", with_sampling=False, fewshotpool_unique=False)
    sequential = FewShotSelectionMethod(sorting="sequential", with_sampling=False, fewshotpool_unique=False)
    random_sampling_from_train = FewShotSelectionMethod(sorting="random", with_sampling=True, fewshotpool_unique=True)
    random_sampling = FewShotSelectionMethod(sorting="random", with_sampling=True, fewshotpool_unique=False)


ALLOWED_SELECTIONS = FewShotSelection._member_names_


class FewShotSampler:
    def __init__(self, few_shots_select: str = "balanced", few_shots_split: str = None):
        # If no info was selected in the config file, it will pass None by default
        if few_shots_select is None:
            few_shots_select = "balanced"

        if few_shots_select not in ALLOWED_SELECTIONS:
            raise ValueError(
                f"few_shots_select must be one of f{','.join(ALLOWED_SELECTIONS[:-1])} or {ALLOWED_SELECTIONS[-1]}, not {few_shots_select}"
            )

        self.few_shots_select = FewShotSelection[few_shots_select]
        self.few_shots_split = few_shots_split

        self._fewshot_cache = {}

    def sample_fewshot_examples(
        self,
        task: "LightevalTask",  # noqa F821
        num_fewshot: int,
        variance_seed: int,
        sampler: random.Random = None,
        formatted_doc: Doc = None,
    ):
        if num_fewshot == 0:
            return []

        # If there is no cache, we initialize it
        if variance_seed not in self._fewshot_cache:
            fewshotpool = task.fewshot_docs()
            if self.few_shots_select.value.sorting == "sequential":
                self.init_fewshot_sampling_sequential(
                    fewshotpool=fewshotpool, num_fewshot=num_fewshot, variance_seed=variance_seed
                )
            elif self.few_shots_select.value.sorting == "random":
                self.init_fewshot_sampling_random(fewshotpool=fewshotpool, variance_seed=variance_seed)
            elif self.few_shots_select.value.sorting == "balanced":
                self.init_fewshot_sampling_balanced(
                    fewshotpool=fewshotpool, num_fewshot=num_fewshot, variance_seed=variance_seed, task=task
                )
            else:
                raise Exception("No correct few shot strategy selected - but this point should not be reachable.")

        if self.few_shots_select.value.with_sampling and sampler is not None:
            if self.few_shots_select.value.fewshotpool_unique:
                # This functionality is here for compatibility with the harness few shot system.
                # It assumes (in some cases) that there is no intersection between the few shot pool and the actual
                # eval examples, and therefore samples only `num_fewshot` (see Task.fewshot_examples)
                cur_cache = sampler.sample(self._fewshot_cache[variance_seed], num_fewshot)
            else:  # we don't reach this yet but let's add it for future use cases
                cur_cache = sampler.sample(self._fewshot_cache[variance_seed], num_fewshot + 1)
        else:
            cur_cache = self._fewshot_cache[variance_seed]

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        return [x for x in cur_cache if x != formatted_doc][:num_fewshot]

    def init_fewshot_sampling_sequential(self, fewshotpool: list[Doc], num_fewshot: int, variance_seed: int):
        # No balancing of the few-shot examples, we take the first items of the set
        # We rotate by num_fewshot * seed (seed >= 0) to be able to have different series of sequential few-shots
        for _ in range(num_fewshot * variance_seed):
            fewshotpool.append(fewshotpool.pop(0))
        self._fewshot_cache[variance_seed] = fewshotpool  # Store few shot examples

    def init_fewshot_sampling_random(self, fewshotpool: list[Doc], variance_seed: int):
        if variance_seed == 0:
            self._fewshot_cache[variance_seed] = fewshotpool
        else:  # we shuffle
            rnd = random.Random(variance_seed)
            self._fewshot_cache[variance_seed] = rnd.shuffle(fewshotpool)

    def init_fewshot_sampling_balanced(
        self,
        fewshotpool: list[Doc],
        num_fewshot: int,
        variance_seed: int,
        task: "LightevalTask",
    ):
        # rnd = random.Random(variance_seed)
        random.seed(variance_seed)

        # Build up balanced selection based on labels
        # Sort by counts of labels
        label_to_instances = defaultdict(list)
        for instance in fewshotpool:
            target = task.doc_to_target(instance, few_shot=True)
            label_to_instances[target].append(instance)

        counts_to_labels = defaultdict(list)
        for label, instances in sorted(label_to_instances.items()):
            counts_to_labels[len(instances)].append(label)

        sorted_labels = []
        # Sort the labels by the number of Instances that belong to them
        for count in sorted(counts_to_labels, reverse=True):
            labels = counts_to_labels[count]
            # Break ties by randomly shuffling labels that have the same number of Instances
            random.shuffle(labels)
            sorted_labels.extend(labels)

        examples = []
        num_instances_to_sample = min(
            len(fewshotpool), num_fewshot + 1
        )  # We add 1 to be able to sample for the test set and remove later the doc we are testing on
        labels_iterable = cycle(sorted_labels)
        while num_instances_to_sample > 0:
            next_label = next(labels_iterable, None)
            if not next_label:
                break

            instances = label_to_instances[next_label]
            # If there are no instances to sample for this particular label, skip it.
            if len(instances) == 0:
                continue

            # Randomly sample without replacement
            examples.append(instances.pop(random.randrange(len(instances))))
            num_instances_to_sample -= 1

        self._fewshot_cache[variance_seed] = examples  # Store few shot examples

    def get_examples_with_chat_template(
        self,
        task: "LightevalTask",
        tokenizer: AutoTokenizer,
        example: str,
        instruction: str,
        fewshot_ex: list[str],
        system_prompt: str,
    ):
        examples = []
        if system_prompt is not None:
            examples.append({"role": "system", "content": system_prompt})
        for ex in fewshot_ex:
            examples.append({"role": "user", "content": task.doc_to_text_without_instructions(ex)})
            examples.append({"role": "assistant", "content": task.doc_to_target(ex)})
        # We add the actual example
        examples.append({"role": "user", "content": example})
        # We add the initial instruction if present, after the system prompt of before the task
        if examples[0]["role"] == "system":
            examples[0]["content"] = examples[0]["content"] + instruction
        else:
            examples[0]["content"] = instruction + examples[0]["content"]

        return tokenizer.apply_chat_template(examples, tokenize=False, add_generation_prompt=True)

    def get_examples(
        self,
        task: "LightevalTask",
        example: str,
        instruction: str,
        fewshot_ex: list[str],
    ):
        if len(fewshot_ex) == 0:
            return instruction + example

        labeled_examples = (
            "\n\n".join([task.doc_to_text_without_instructions(ex) + task.doc_to_target(ex) for ex in fewshot_ex])
            + "\n\n"
        )
        return instruction + labeled_examples + example

    def create_multi_turn_contexts(
        self, doc: Doc, use_chat_template: bool, system_prompt: Optional[str], tokenizer: PreTrainedTokenizer
    ) -> list[str]:
        """Creates N contexts (depending on the number of turn) for a tasks.
        Multi turn tasks need use chat templating.

        Args:
            doc (Doc): Formated document.
            use_chat_template (bool): wether or not to use chat template. Will fail if false.
            system_prompt (Optional[str]): The system prompt to use
            tokenizer (PreTrainedTokenizer): The tokenizer used for the chat template

        Raises:
            ValueError: If use_chat_template is set to false.

        Returns:
            list[str]: contexts for every turn
        """
        if not use_chat_template:
            raise ValueError("You need to use the chat template to create multi turn contexts")

        role_content_list = []
        if system_prompt is not None:
            role_content_list.append({"role": "system", "content": system_prompt})

        for i in doc.specific["multi_turn_queries"]:
            role_content_list.append({"role": "user", "content": i})
            role_content_list.append({"role": "assistant", "content": "{model_response}"})
        role_content_list.pop(-1)

        contexts = []
        offset = 2 if system_prompt is not None else 1
        for i in range(0, len(role_content_list), offset + 1):
            c = tokenizer.apply_chat_template(
                role_content_list[: i + offset], add_generation_prompt=True, tokenize=False, add_special_tokens=False
            )
            contexts.append(c)

        return contexts, 0

    def fewshot_context(
        self,
        task: "LightevalTask",
        doc: Doc,
        num_fewshot: int,
        seed: int,
        sampler: Optional[random.Random] = None,
        truncate_few_shots: bool = False,
        max_model_length: Optional[int] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        use_chat_template=False,
        system_prompt: str = None,
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs - should be preformatted.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param seed: seed
            The random seed used to randomly sample examples. If -1, no shuffling will occur, and the samples taken
            will be the `num_fewshot` firsts of the set.
        :returns: str
            The fewshot context.
        """
        if use_chat_template and tokenizer is None:
            raise Exception("You can't use a chat template if you don't pass the tokenizer")

        example, instruction = task.doc_to_text_and_instructions(doc)

        # will be an empty list if num_fewshot == 0
        fewshot_ex = self.sample_fewshot_examples(
            task=task, num_fewshot=num_fewshot, formatted_doc=doc, variance_seed=seed, sampler=sampler
        )

        num_effective_fewshots = num_fewshot

        if use_chat_template:
            output = self.get_examples_with_chat_template(
                task=task,
                tokenizer=tokenizer,
                example=example,
                instruction=instruction,
                fewshot_ex=fewshot_ex,
                system_prompt=system_prompt,
            )
            toks = tokenizer(output)["input_ids"]
        else:
            output = self.get_examples(task=task, example=example, instruction=instruction, fewshot_ex=fewshot_ex)
            toks = tokenizer(output)["input_ids"]

        # If we need to truncate few-shots to fit in the context
        if truncate_few_shots and max_model_length is not None and tokenizer is not None:
            # If self.generation_size is None, the maximum allowed generation size depends
            # on the model maximum context length, not on the task - we don't take it into account here
            # but we probably should
            gen_size = task.generation_size if task.generation_size is not None else 0

            while len(toks) + gen_size > max_model_length and num_effective_fewshots >= 0:
                num_effective_fewshots -= 1

                if use_chat_template:
                    output = self.get_examples_with_chat_template(
                        task=task,
                        tokenizer=tokenizer,
                        example=example,
                        instruction=instruction,
                        fewshot_ex=fewshot_ex[:num_effective_fewshots],
                        system_prompt=system_prompt,
                    )
                    toks = tokenizer(output)["input_ids"]
                else:
                    output = self.get_examples(
                        task=task,
                        example=example,
                        instruction=instruction,
                        fewshot_ex=fewshot_ex[:num_effective_fewshots],
                    )
                    toks = tokenizer(output)["input_ids"]

        return output, num_effective_fewshots

    def get_fewshot_seeds(self, few_shot_iterations: int = None) -> list[int]:
        """Return a list of seeds for sampling several times the few shots"""
        # todo @saylortwift: check which seed for bb
        if few_shot_iterations <= 1:
            return [0]
        seeds = range(few_shot_iterations)
        hlog_warn(f"Running {self.name} with {few_shot_iterations} few-shot iterations.")
        return seeds
