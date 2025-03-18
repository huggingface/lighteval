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
from typing import TYPE_CHECKING, Optional, Tuple, Union

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.endpoints.inference_providers_model import InferenceProvidersClient
from lighteval.models.litellm_model import LiteLLMClient
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from lighteval.tasks.lighteval_task import LightevalTask


class PromptManager:
    def __init__(self, task: "LightevalTask", lm: LightevalModel):
        self.model = lm
        self.task = task
        self.few_shot_sampler = FewShotSampler(task)

    @staticmethod
    def doc_to_text(doc: Doc, return_instructions: bool = False) -> Union[str, Tuple[str, str]]:
        """
        Returns the query of the document without the instructions. If the
        document has instructions, it removes them from the query:

        Args:
            doc (Doc): document class, containing the query and the
                instructions.

        Returns:
            str: Query of the document without the instructions.
        """
        instructions = doc.instruction if doc.instruction is not None else ""
        if not doc.query.startswith(instructions):
            raise ValueError(f"Prompt query {doc.query} is not starting with instruction {instructions}")

        return (
            (doc.query[len(instructions) :], instructions) if return_instructions else doc.query[len(instructions) :]
        )

    @staticmethod
    def doc_to_target(formatted_doc: Doc) -> str:
        """
        Returns the target of the given document.

        Args:
            formatted_doc (Doc): Formatted document.

        Returns:
            str: Target of the document, which is the correct answer for a document.
        """
        return as_list(formatted_doc.get_golds())[0]

    @staticmethod
    def doc_to_fewshot_sorting_class(formatted_doc: Doc) -> str:
        """
        In some cases, when selecting few-shot samples, we want to use specific document classes
        which need to be specified separately from the target.
        For example, a document where the gold is a json might want to use only one of the keys of
        the json to define sorting classes in few shot samples. Else we take the gold.

        Args:
            formatted_doc (Doc): Formatted document.

        Returns:
            str: Class of the fewshot document
        """
        return formatted_doc.fewshot_sorting_class or PromptManager.doc_to_target(formatted_doc)

    def add_context_to_doc(
        self,
        doc: Doc,
        num_fewshot: int,
        seed: int,
        sampler: Optional[random.Random] = None,
        truncate_few_shots: bool = False,
        use_chat_template=False,
        system_prompt: str = None,
    ) -> Doc:
        is_multi_turn = doc.specific is not None and len(doc.specific.get("multi_turn_queries", [])) > 0
        if is_multi_turn:
            ctx, num_effective_few_shots = self._multi_turn_contexts(doc, use_chat_template, system_prompt)
            doc.specific["multi_turn_queries_context"] = ctx
        else:
            ctx, num_effective_few_shots = self._single_turn_context(
                doc=doc,
                num_fewshot=num_fewshot,
                seed=seed,
                truncate_few_shots=truncate_few_shots,
                sampler=sampler,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
            )
        doc.num_effective_few_shots = num_effective_few_shots
        doc.num_asked_few_shots = num_fewshot
        doc.ctx = ctx

        return doc

    def _multi_turn_contexts(self, doc: Doc, use_chat_template: bool, system_prompt: Optional[str]) -> list[str]:
        """Creates N contexts (depending on the number of turn) for a tasks.
        Multi turn tasks need use chat templating.

        Args:
            doc (Doc): Formatted document.
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
            c = self.model.tokenizer.apply_chat_template(
                role_content_list[: i + offset], add_generation_prompt=True, tokenize=False, add_special_tokens=False
            )
            contexts.append(c)

        return contexts, 0

    def _single_turn_context(
        self,
        doc: Doc,
        num_fewshot: int,
        seed: int,
        sampler: Optional[random.Random] = None,
        truncate_few_shots: bool = False,
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
        if use_chat_template and self.model.tokenizer is None:
            raise Exception("You can't use a chat template if your model does not have a tokenizer")

        example, instruction = self.doc_to_text(doc, return_instructions=True)

        fewshot_ex = self.few_shot_sampler.sample_fewshot_examples(
            num_fewshot=num_fewshot, formatted_doc=doc, variance_seed=seed, sampler=sampler
        )

        num_effective_fewshots = num_fewshot

        output = self.get_examples(
            example=example,
            instruction=instruction,
            fewshot_ex=fewshot_ex,
            system_prompt=system_prompt,
            use_chat_template=use_chat_template,
        )
        if not use_chat_template:
            toks = self.model.tok_encode(output)
        else:
            toks = [self.model.tok_encode(msg["content"]) for msg in output]
            toks = [t for ts in toks for t in ts]

        # If we need to truncate few-shots to fit in the context
        if truncate_few_shots and self.model.max_length is not None and self.model.tokenizer is not None:
            # If self.generation_size is None, the maximum allowed generation size depends
            # on the model maximum context length, not on the task - we don't take it into account here
            # but we probably should
            gen_size = self.task.generation_size if self.task.generation_size is not None else 0

            while len(toks) + gen_size > self.model.max_length and num_effective_fewshots >= 0:
                num_effective_fewshots -= 1
                output = self.get_examples(
                    example=example,
                    instruction=instruction,
                    fewshot_ex=fewshot_ex[:num_effective_fewshots],
                    system_prompt=system_prompt,
                    use_chat_template=use_chat_template,
                )
                if not use_chat_template:
                    toks = self.model.tok_encode(output)
                else:
                    toks = [self.model.tok_encode(msg["content"]) for msg in output]
                    toks = [t for ts in toks for t in ts]

        if type(self.model) in [LiteLLMClient, InferenceProvidersClient]:
            return output, num_effective_fewshots

        elif use_chat_template:
            return self.model.tokenizer.apply_chat_template(
                output, tokenize=False, add_generation_prompt=True
            ), num_effective_fewshots

        return output, num_effective_fewshots

    def get_examples(
        self,
        example: str,
        instruction: Union[str | None],
        fewshot_ex: list[str],
        system_prompt: Union[str | None],
        use_chat_template: bool,
    ):
        examples = []
        # Few shot examples
        for ex in fewshot_ex:
            if use_chat_template:
                examples.append({"role": "user", "content": self.doc_to_text(ex, return_instructions=False)})
                examples.append({"role": "assistant", "content": self.doc_to_target(ex)})
            else:
                examples.append(self.doc_to_text(ex, return_instructions=False) + self.doc_to_target(ex))

        # Actual example
        if use_chat_template:
            examples.append({"role": "user", "content": example})
        else:
            examples.append(example)

        # System prompt and instruction
        if use_chat_template:
            if system_prompt is not None:  # We add system prompt and instruction jointly if possible
                examples.insert(0, {"role": "system", "content": system_prompt + instruction})
            else:  # Else we add the instruction to the first example
                examples[0]["content"] = instruction + examples[0]["content"]
            return examples
        else:
            if system_prompt is not None:
                output = system_prompt + instruction + "\n\n".join(examples)
            else:
                output = instruction + "\n\n".join(examples)
            if output == "\n\n":
                return ""
            return output


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
                f"few_shots_select must be one of f{','.join(ALLOWED_SELECTIONS[:-1])} or {ALLOWED_SELECTIONS[-1]}, not {few_shots_select}"
            )

        self.few_shots_select = FewShotSelection[few_shots_select]
        self.few_shots_split = task.fewshot_split

        self._fewshot_cache = {}

    def sample_fewshot_examples(
        self,
        num_fewshot: int,
        variance_seed: int,
        sampler: random.Random = None,
        formatted_doc: Doc = None,
    ):
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
            target = PromptManager.doc_to_fewshot_sorting_class(instance)
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
