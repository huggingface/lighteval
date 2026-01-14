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

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import cycle
from typing import TYPE_CHECKING

from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from lighteval.tasks.lighteval_task import LightevalTask


class PromptManager:
    def __init__(self, use_chat_template: bool = False, tokenizer=None, system_prompt: str | None = None):
        self.use_chat_template = use_chat_template
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt  # System prompt to be used in chat templates

    def prepare_prompt(self, doc: Doc) -> str:
        """Prepare a prompt from a document, either using chat template or plain text format.

        Returns:
            str: The formatted prompt string
        """
        if self.use_chat_template:
            return self._prepare_chat_template(doc)
        else:
            return self._prepare_plain_text(doc)

    def prepare_prompt_multimodal(self, doc: Doc) -> str:
        if self.use_chat_template is False or self.tokenizer is None:
            raise ValueError("Multimodal prompts are only supported with chat template format.")

        if doc.images is None:
            raise ValueError("Multimodal prompts require images to be provided in the document.")

        text_content = [{"type": "text", "text": doc.query}]
        image_content = [{"type": "image", "image": image} for image in doc.images]
        message = {"role": "user", "content": text_content + image_content}

        if (
            self.system_prompt is not None or doc.instruction is not None
        ):  # We add system prompt and instruction jointly if possible
            system_prompt = self.system_prompt if self.system_prompt is not None else ""
            instruction = doc.instruction if doc.instruction is not None else ""
            system_content = [{"type": "text", "text": system_prompt + instruction}]
            system_prompt_message = {"role": "system", "content": system_content}
            message = [system_prompt_message, message]

        else:
            message = [message]

        return self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

    def prepare_prompt_api(self, doc: Doc) -> list[dict[str, str]]:
        """Prepare a prompt for API calls, using a chat-like format.
        Will not tokenize the message because APIs will usually handle this.

        Returns:
            list[dict[str, str]]: List of message dictionaries for API calls
        """
        return self._prepare_chat_template(doc, tokenize=False)

    def _prepare_chat_template(self, doc: Doc, tokenize: bool = True) -> str:
        """Prepare prompt using chat template format.

        Returns:
            str | list[dict[str, str]]: Formatted chat template string or list of messages
        """
        messages = []
        instruction_used = False  # Flag to check if instruction is used in the first few-shot example

        # Add system prompt if available
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add few-shot examples
        for ix, fewshot_sample in enumerate(doc.fewshot_samples):
            query = self._extract_query(fewshot_sample.query, fewshot_sample.instruction)
            if ix == 0 and doc.instruction is not None:
                instruction_used = True
                query = doc.instruction + query

            messages.append({"role": "user", "content": query})
            messages.append({"role": "assistant", "content": fewshot_sample.get_golds()[0]})

        # Add main query
        main_query = self._extract_query(doc.query, doc.instruction)

        if doc.instruction is not None and not instruction_used:
            # If instruction is provided, prepend it to the main query
            main_query = doc.instruction + main_query

        messages.append({"role": "user", "content": main_query})

        if tokenize:  # for local models
            assert self.tokenizer is not None, "Tokenizer must be set for chat template formatting."

            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        else:  # for apis
            return messages

    def _prepare_plain_text(self, doc: Doc) -> str:
        """Prepare prompt using plain text format.

        Returns:
            str: The formatted plain text prompt
        """
        parts = []

        # Add system prompt if available
        if self.system_prompt is not None:
            parts.append(self.system_prompt)

        if doc.instruction is not None:
            parts.append(doc.instruction)

        # Add few-shot examples
        for fewshot_sample in doc.fewshot_samples:
            query = self._extract_query(fewshot_sample.query, fewshot_sample.instruction)
            parts.append(query + " " + fewshot_sample.get_golds()[0].strip())

        # Add main query
        query = self._extract_query(doc.query, doc.instruction)
        parts.append(query)

        return "\n\n".join(parts)

    def _extract_query(self, query: str, instruction: str | None) -> str:
        """Extract query content, removing instruction prefix if appropriate.

        Returns:
            str: The extracted query content without instruction prefix if it was present
        """
        if instruction is not None:
            if query.startswith(instruction):
                return query[len(instruction) :].strip()
            else:
                return query
        return query


@dataclass
class FewShotSelectionMethod:
    sorting: str  # sorting method for the overall few shot pool (balanced, random, sequential)
    with_sampling: bool  # samples item randomly from the few shot pool
    fewshotpool_unique: bool  # set to true if you are CERTAIN there is no intersection between the few shot pool and your evaluation set


class FewShotSelection(Enum):
    balanced = FewShotSelectionMethod(sorting="balanced", with_sampling=False, fewshotpool_unique=False)
    random = FewShotSelectionMethod(sorting="random", with_sampling=False, fewshotpool_unique=False)
    sequential = FewShotSelectionMethod(sorting="sequential", with_sampling=False, fewshotpool_unique=False)
    random_sampling_from_train = FewShotSelectionMethod(sorting="random", with_sampling=True, fewshotpool_unique=True)
    random_sampling = FewShotSelectionMethod(sorting="random", with_sampling=True, fewshotpool_unique=False)


ALLOWED_SELECTIONS = FewShotSelection._member_names_


class FewShotSampler:
    def __init__(self, task: "LightevalTask"):
        self.task = task

        few_shots_select = task.fewshot_selection
        if few_shots_select is None:
            few_shots_select = "balanced"

        if few_shots_select not in ALLOWED_SELECTIONS:
            raise ValueError(
                f"few_shots_select must be one of {','.join(ALLOWED_SELECTIONS[:-1])} or {ALLOWED_SELECTIONS[-1]}, not {few_shots_select}"
            )

        self.few_shots_select = FewShotSelection[few_shots_select]
        self.few_shots_split = task.fewshot_split

        self._fewshot_cache = {}

    def sample_fewshot_examples(
        self,
        num_fewshot: int,
        variance_seed: int,
        sampler: random.Random | None = None,
        formatted_doc: Doc | None = None,
    ) -> list[Doc]:
        if num_fewshot == 0:
            return []

        self._init_fewshot_pool(num_fewshot=num_fewshot, variance_seed=variance_seed)
        samples = self._sample_from_pool(num_fewshot=num_fewshot, variance_seed=variance_seed, sampler=sampler)

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        return [x for x in samples if x != formatted_doc][:num_fewshot]

    def _init_fewshot_pool(
        self,
        num_fewshot: int,
        variance_seed: int,
    ):
        # If there is no cache, we initialize it
        if variance_seed not in self._fewshot_cache:
            if self.few_shots_select.value.sorting == "sequential":
                self._init_fewshot_sampling_sequential(num_fewshot=num_fewshot, variance_seed=variance_seed)
            elif self.few_shots_select.value.sorting == "random":
                self._init_fewshot_sampling_random(variance_seed=variance_seed)
            elif self.few_shots_select.value.sorting == "balanced":
                self._init_fewshot_sampling_balanced(num_fewshot=num_fewshot, variance_seed=variance_seed)
            else:
                raise Exception("No correct few shot strategy selected - but this point should not be reachable.")

    def _sample_from_pool(self, variance_seed: int, num_fewshot: int, sampler: random.Random) -> list:
        if self.few_shots_select.value.with_sampling and sampler is not None:
            if self.few_shots_select.value.fewshotpool_unique:
                # This functionality is here for compatibility with the harness few shot system.
                # It assumes (in some cases) that there is no intersection between the few shot pool and the actual
                # eval examples, and therefore samples only `num_fewshot` (see Task.fewshot_examples)
                return sampler.sample(self._fewshot_cache[variance_seed], num_fewshot)
            else:  # we don't reach this yet but let's add it for future use cases
                return sampler.sample(self._fewshot_cache[variance_seed], num_fewshot + 1)
        else:
            return self._fewshot_cache[variance_seed]

    def _init_fewshot_sampling_sequential(self, num_fewshot: int, variance_seed: int):
        # No balancing of the few-shot examples, we take the first items of the set
        # We rotate by num_fewshot * seed (seed >= 0) to be able to have different series of sequential few-shots
        fewshotpool = self.task.fewshot_docs()
        for _ in range(num_fewshot * variance_seed):
            fewshotpool.append(fewshotpool.pop(0))
        self._fewshot_cache[variance_seed] = fewshotpool  # Store few shot examples

    def _init_fewshot_sampling_random(self, variance_seed: int):
        fewshotpool = list(self.task.fewshot_docs())
        if variance_seed == 0:
            self._fewshot_cache[variance_seed] = fewshotpool
        else:  # we shuffle
            rnd = random.Random(variance_seed)
            rnd.shuffle(fewshotpool)
            self._fewshot_cache[variance_seed] = fewshotpool

    def _init_fewshot_sampling_balanced(
        self,
        num_fewshot: int,
        variance_seed: int,
    ):
        fewshotpool = self.task.fewshot_docs()

        random.seed(variance_seed)

        # Build up balanced selection based on fewshot_sorting_class
        # (or the gold target, if the class is undefined)
        label_to_instances = defaultdict(list)
        for instance in fewshotpool:
            target = instance.fewshot_sorting_class or as_list(instance.get_golds())[0]
            label_to_instances[target].append(instance)

        # Sort by counts of class labels
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

    def get_fewshot_seeds(self, few_shot_iterations: int = None) -> list[int]:
        """Return a list of seeds for sampling several times the few shots"""
        # todo @saylortwift: check which seed for bb
        if few_shot_iterations <= 1:
            return [0]
        seeds = range(few_shot_iterations)
        logger.warning(f"Running {self.task.name} with {few_shot_iterations} few-shot iterations.")
        return seeds
