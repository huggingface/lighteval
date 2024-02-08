import os
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset, LoglikelihoodSingleTokenDataset
from lighteval.logging.hierarchical_logger import hlog, hlog_err, hlog_warn
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_config import BaseModelConfig, EnvConfig
from lighteval.models.model_output import Batch, GenerateReturn, LoglikelihoodReturn, LoglikelihoodSingleTokenReturn
from lighteval.models.utils import _get_dtype, _get_precision, _simplify_name
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    GreedyUntilWithLogitsRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
    Request,
)
from lighteval.utils import as_list, is_accelerate_available
from lighteval.utils_parallelism import find_executable_batch_size


if is_accelerate_available():
    from accelerate.utils import get_max_memory

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STARTING_BATCH_SIZE = 512


class BaseModel(LightevalModel):
    def __init__(
        self,
        config: BaseModelConfig,
        env_config: EnvConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self._config = config.init_configs(env_config)
        self.accelerator = config.accelerator
        self._batch_size = config.batch_size
        self._max_length = self._init_max_length(config.max_length)

        self._add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False
        self._tokenizer = self._create_auto_tokenizer(config, env_config)

        # If model_parallel is not set we compare the number of process with the number of GPUs
        self.model = self._create_auto_model(config, env_config)
        self.model.eval()
        torch.set_grad_enabled(False)

        self._device = config.accelerator.device if config.accelerator is not None else "cpu"
        self.multichoice_continuations_start_space = config.multichoice_continuations_start_space

        # We are in DP (and launch the script with `accelerate launch`)
        if not config.model_parallel and not config.load_in_4bit and not config.load_in_8bit:
            # might need to use accelerate instead
            # self.model = config.accelerator.prepare(self.model)
            hlog(f"Using Data Parallelism, putting model on device {self._device}")
            self.model = self.model.to(self._device)

        self.model_name = _simplify_name(config.pretrained)
        self.model_sha = config.get_model_sha()

        self.precision = _get_precision(config, model_auto_config=self._config)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        return self._max_length

    def init_model_parallel(self, model_parallel: bool = None) -> Tuple[bool, Optional[dict], Optional[str]]:
        """Compute all the parameters related to model_parallel"""
        if not is_accelerate_available():
            return False, None, None

        self.num_local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.num_machines = int(os.environ.get("WORLD_SIZE", 0)) // self.num_local_processes
        if self.num_machines == 0:
            hlog("We are not in a distributed setting. Setting model_parallel to False.")
            model_parallel = False

        if model_parallel is None:
            max_memory_all_gpus = get_max_memory()  # A dict of the max memory for all the gpus
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            model_parallel = bool(self.num_local_processes < len(max_memory_all_gpus))
            hlog(
                f"Setting model parallel to {model_parallel} since "
                f"the number of local processes is {self.num_local_processes} "
                f"and the number of GPUs is {len(max_memory_all_gpus)}"
            )
        if model_parallel:
            max_memory_all_gpus = get_max_memory()  # A dict of the max memory for all the gpus
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            max_mem_this_process = {
                k: v
                for k, v in max_memory_all_gpus.items()
                if k % self.num_local_processes == (self.accelerator.process_index % self.num_local_processes)
            }
            device_map = "auto"
            hlog(
                f"Model parallel was set to True, setting max memory per GPU to {max_mem_this_process} and device map to {device_map}"
            )
        else:
            max_mem_this_process = None
            device_map = None
            hlog(
                f"Model parallel was set to False, max memory set to {max_mem_this_process} and device map to {device_map}"
            )
        return model_parallel, max_mem_this_process, device_map

    def _create_auto_model(self, config: BaseModelConfig, env_config: EnvConfig) -> transformers.PreTrainedModel:
        """
        Creates an instance of the pretrained HF model.

        Args:
            pretrained (str): The name or path of the pretrained model.
            revision (str): The revision of the model.
            subfolder (Optional[str], optional): The subfolder within the model. Defaults to None.
            max_memory (Optional[dict], optional): The maximum memory to allocate for the model per GPU. Defaults to None.
            device_map (Optional[dict], optional): The device mapping for the model. Defaults to None.
            torch_dtype (Optional[Union[str, torch.dtype]], optional): The torch data type for the model. Defaults to None.
            quantization_config (Optional[Union[BitsAndBytesConfig, GPTQConfig]], optional): The quantization configuration for the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            cache_dir (str, optional): The cache directory for the model. Defaults to "/scratch".

        Returns:
            transformers.PreTrainedModel: The created auto model instance.
        """
        config.model_parallel, max_memory, device_map = self.init_model_parallel(config.model_parallel)
        torch_dtype = _get_dtype(config.dtype, self._config)

        model = AutoModelForCausalLM.from_pretrained(
            config.pretrained,
            revision=config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
            max_memory=max_memory,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=config.trust_remote_code,
            cache_dir=env_config.cache_dir,
            offload_folder=env_config.cache_dir,
            token=env_config.token,
            quantization_config=config.quantization_config,
        )

        return model

    def _create_auto_tokenizer(
        self, config: BaseModelConfig, env_config: EnvConfig
    ) -> transformers.PreTrainedTokenizer:
        return self._create_auto_tokenizer_with_name(config.pretrained, config=config, env_config=env_config)

    def _create_auto_tokenizer_with_name(
        self, model_name: str, config: BaseModelConfig, env_config: EnvConfig
    ) -> transformers.PreTrainedTokenizer:
        """
        Create a Hugging Face AutoTokenizer for language model.

        Args:
            pretrained (str): The identifier of the pretrained model to load.
            revision (str): The specific model version to load.
            subfolder (str): The subfolder within the model repository.
            tokenizer (str, optional): The identifier of the tokenizer to load. If not provided, the default tokenizer for the pretrained model will be used.
            cache_dir (str, optional): The directory to cache the downloaded models and tokens. Defaults to "/scratch".
            trust_remote_code (bool, optional): Whether to trust remote code execution during tokenization. Defaults to False.

        Returns:
            transformers.PreTrainedTokenizer: The created tokenizer.

        Raises:
            RecursionError: If an error occurs during tokenization, a fallback tokenizer with "<unk>" token will be created.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name if config.tokenizer is None else config.tokenizer,
                revision=config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
                cache_dir=env_config.cache_dir,
                token=env_config.token,
                trust_remote_code=config.trust_remote_code,
                padding_side="left",
                truncation_side="left",
            )
        except RecursionError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name if config.tokenizer is None else config.tokenizer,
                revision=config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
                cache_dir=env_config.cache_dir,
                token=env_config.token,
                trust_remote_code=config.trust_remote_code,
                unk_token="<unk>",
                padding_side="left",
                truncation_side="left",
            )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = self.max_length
        hlog("Tokenizer truncation and padding size set to the left side.")

        return tokenizer

    def _init_max_length(self, max_length) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.

        Args:
            max_length (Optional[int]): The maximum length of the input sequence. If not provided, it will be determined
                based on the model's configuration or tokenizer's model_max_length attribute.

        Returns:
            int: Max length to use depending on the available args and config
        """
        if max_length is not None:
            return max_length
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")

        for attr in seqlen_config_attrs:
            if hasattr(self._config, attr):
                return getattr(self._config, attr)

        if hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        # Default max sequence length setting for when no `max_length` is provided
        # or no max length config setting is found in the model or tokenizer.
        return 2048

    @property
    def batch_size(self) -> int:
        if self._batch_size >= 0:
            self._batch_size = self._get_batch_size(max_input_length=self.max_length)
        return self._batch_size  # * gpus

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    @property
    def disable_tqdm(self) -> bool:
        disable_tqdm = False
        if self.accelerator:
            disable_tqdm = bool(not self.accelerator.is_main_process)
        return disable_tqdm

    def _check_continuations_start_space(self, continuation: str) -> str:
        """Some models tokenizer want a space at the beginning and other not. We update this if needed here.
        multichoice_continuations_start_space can be:
        - True (add a space if these isn't one)
        - False (remove a space if there is one)
        - None (Don't touch - default)
        Todo: find a way to add this back WITHOUT breaking compatibility with the harness
        """
        if self.multichoice_continuations_start_space is True and continuation[0] != " ":
            continuation = " " + continuation
        if self.multichoice_continuations_start_space is False and continuation[0] == " ":
            continuation = continuation.lstrip()
        return continuation

    def _model_call(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs).logits

    def _get_batch_size(self, max_input_length: int, override_bs: int = 0, starting_batch_size: int = 512) -> int:
        if override_bs > 0:
            return override_bs
        hlog(f"Detecting largest batch size with max_input_length={max_input_length}")

        @find_executable_batch_size(
            starting_batch_size=starting_batch_size
        )  # if OOM, then halves batch_size and tries again
        def forward_batch(batch_size):
            test_batch = torch.ones(
                (batch_size + int(0.1 * batch_size), max_input_length), device=self.device
            ).long()  # We add 10% for marging :)
            F.log_softmax(self._model_call(test_batch).float(), dim=-1).cpu()
            return batch_size

        batch_size = forward_batch()
        hlog(f"Determined largest batch size: {batch_size}")
        return batch_size

    def greedy_until_with_logits(
        self,
        requests: list[GreedyUntilWithLogitsRequest],
        override_bs: Optional[int] = None,
    ) -> list[GenerateReturn]:
        """
        Generates sequences greedily until a stopping condition is met,
        returning both the generated sequences and the logits.

        Args:
            requests (list[tuple[str, dict]]): A list of input requests,
                where each request is a tuple containing a prompt string and a dictionary of additional parameters.
            override_bs (Optional[int], optional): Overrides the batch size for generation. Defaults to None.

        Returns:
            list[GenerateReturn]: A list of GenerateReturn objects,
                where each object contains the generated sequence and the corresponding logits.
        """

        return self.greedy_until(
            requests,
            returns_logits=True,
            disable_tqdm=self.disable_tqdm,
            override_bs=override_bs,
        )

    def greedy_until(
        self,
        requests: list[GreedyUntilRequest],
        returns_logits: bool = False,
        override_bs: Optional[int] = None,
    ) -> list[GenerateReturn]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            returns_logits (bool, optional): Whether to return the logits of the generated responses. Defaults to False.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerateReturn]: list of generated responses.
        """
        for request in requests:
            request.stop_sequence = as_list(request.stop_sequence) + [self.tokenizer.eos_token]
            request.tokenized_context = self.tok_encode(request.context)

        dataset = GenerativeTaskDataset(requests=requests, dataset_splits=self.DATASET_SPLITS)
        starting_batch_size = STARTING_BATCH_SIZE
        results = []

        for split_start, split_end in tqdm(
            dataset.splits_start_end_iterator(),
            total=self.DATASET_SPLITS,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            # Longest context in the current split is the first item (since we sort reversed)
            longest_context_continuation_size_in_split = len(dataset[0].tokenized_context) + dataset[0].generation_size
            max_context_continuation_size_allowed = min(longest_context_continuation_size_in_split, self.max_length)
            batch_size = self._get_batch_size(
                override_bs=override_bs,
                max_input_length=max_context_continuation_size_allowed,
                starting_batch_size=starting_batch_size,
            )
            # For next iteration, since the batch will be smaller, we'll test a bigger batch size
            starting_batch_size = batch_size * 2

            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)
            if self.accelerator:
                dataloader = self.accelerator.prepare(dataloader)

            for batch in tqdm(
                dataloader, desc="Greedy generation", position=1, leave=False, disable=self.disable_tqdm
            ):
                # NOTE: we are assuming all items in a batch behave similarly (same
                # stop_tokens and max_tokens genrated) which is not necessarily
                # the case! Because of that we only use batch size of 1
                stop_tokens = batch[0].stop_sequence
                max_generated_tokens = batch[0].generation_size
                context = [c.context for c in batch]
                max_context_size_allowed = self.max_length - max_generated_tokens

                tokenized = self.tokenizer(
                    context,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_context_size_allowed,
                    add_special_tokens=self.add_special_tokens,
                ).to(self.device)

                prepared_batch = Batch(
                    input_ids=tokenized["input_ids"],
                    input_lengths=[len(item == 1) for item in tokenized["attention_mask"]],
                    input_mask=tokenized["attention_mask"],
                    truncated=[0] * len(tokenized["input_ids"]),
                    padded=[0] * len(tokenized["input_ids"]),
                )

                cur_reponses = self._generate(
                    batch=prepared_batch,
                    max_tokens=max_generated_tokens,
                    stop_tokens=stop_tokens,
                    returns_logits=returns_logits,
                )
                results.extend(cur_reponses)

        return dataset.get_original_order(results)

    def _generate(
        self,
        batch: Batch,
        max_tokens: int,
        stop_tokens: list[str],
        returns_logits: Optional[bool] = False,
    ) -> list[GenerateReturn]:
        """Contains the actual logic of the generation.
        First computes the stop sequences, then generates the predictions, then converts the outputs to GenerateReturn.
        """
        stopping_criteria = stop_sequences_criteria(self.tokenizer, stop_sequences=stop_tokens, batch=batch)

        # Compute model generation
        outputs = self.model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.input_mask,
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if returns_logits:
            logits = self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        generations = outputs.sequences[:, batch.input_ids.size(1) :]
        generations, len_gens = self.pad_and_gather(generations)
        batch.input_ids, len_ids = self.pad_and_gather(batch.input_ids)

        logits, len_logits = None, None
        if returns_logits:
            logits, len_logits = self.pad_and_gather(logits)
            logits = logits.cpu().numpy()

        # We gather remaining info
        batch.truncated = torch.tensor(batch.truncated, device=self.device)
        if self.accelerator:
            batch.truncated = self.accelerator.gather_for_metrics(batch.truncated)
        batch.padded = torch.tensor(batch.padded, device=self.device)
        if self.accelerator:
            batch.padded = self.accelerator.gather_for_metrics(batch.padded)

        # We convert to GenerateReturn outputs
        all_responses = []
        for ix, (generation, batched_input, trunc, padded) in enumerate(
            zip(generations, batch.input_ids, batch.truncated, batch.padded)
        ):
            # Ensure the generated responses do not contain the stop sequences.
            generation = generation[: len_gens[ix]]
            decoded_generation = self.tok_decode([generation])[0]

            for term in stop_tokens:
                decoded_generation = decoded_generation.split(term)[0]

            cur_response = GenerateReturn(
                result=decoded_generation,
                logits=logits[ix][: len_logits[ix]] if returns_logits else None,
                generated_tokens=generation,
                input_tokens=batched_input[: len_ids[ix]],
                truncated_tokens_count=trunc.cpu().item(),
                padded_tokens_count=padded.cpu().item(),
            )
            all_responses.append(cur_response)

        return all_responses

    def loglikelihood(
        self,
        requests: list[LoglikelihoodRequest],
        override_bs: Optional[int] = None,
    ) -> list[LoglikelihoodReturn]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.

        Args:
            requests (list[Tuple[str, dict]]): _description_

        Returns:
            list[Tuple[float, bool]]: _description_
        """
        for request in requests:
            if request.context == "":
                request.tokenized_context = [self.tokenizer.eos_token_id]
                request.tokenized_continuation = self.tok_encode(request.choice)
            else:
                # The following line is mandatory for compatibility with the harness
                request.tokenized_context, request.tokenized_continuation = self.tok_encode_pair(
                    request.context, request.choice
                )

        return self._loglikelihood_tokens(requests, override_bs=override_bs)

    def loglikelihood_rolling(
        self,
        requests: list[LoglikelihoodRollingRequest],
        override_bs=None,
    ) -> list[LoglikelihoodReturn]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""

        for request in requests:  # tuple of one elem
            request.tokenized_context = [self.tokenizer.eos_token_id]  # Fake context
            request.tokenized_continuation = self.tok_encode(request.context)

        results = self._loglikelihood_tokens(
            requests,
            override_bs=override_bs,
            return_bool_score=False,
        )
        return results

    def _loglikelihood_tokens(
        self,
        requests: list[LoglikelihoodRequest],
        override_bs: int = -1,
        return_bool_score: bool = True,
    ) -> list[LoglikelihoodReturn]:
        dataset = LoglikelihoodDataset(requests=requests, dataset_splits=self.DATASET_SPLITS)
        starting_batch_size = STARTING_BATCH_SIZE
        res = []

        for split_start, split_end in tqdm(dataset.splits_start_end_iterator()):
            context_enc = dataset[0].tokenized_context
            continuation_enc = dataset[0].tokenized_continuation
            max_context_continuation_size_allowed = len(
                (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
            )

            batch_size = self._get_batch_size(
                override_bs=override_bs,
                max_input_length=max_context_continuation_size_allowed,
                starting_batch_size=starting_batch_size,
            )
            starting_batch_size = batch_size * 2

            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)
            if self.accelerator:
                dataloader = self.accelerator.prepare(dataloader)

            for batch in tqdm(dataloader, disable=self.disable_tqdm):
                prepared_batch = self.prepare_batch(
                    batch,
                    padding_length=max_context_continuation_size_allowed,
                    max_context=max_context_continuation_size_allowed,
                )

                model_output = self._model_call(prepared_batch.input_ids)
                logits = F.log_softmax(model_output, dim=-1)  # [batch, padding_length, vocab]

                logits_sum = []
                max_equals = []
                batch_cont_tokens = []
                for cur_request, cur_logits, inplen in zip(batch, logits, prepared_batch.input_lengths):
                    cont_toks = torch.tensor(cur_request.tokenized_continuation, dtype=torch.long, device=self.device)
                    contlen = cont_toks.shape[0]
                    # We only look at the continuation tokens
                    if contlen > inplen:
                        # Continuation is longer than the input size, we are in rolling mode (only continuation)
                        cur_logits = cur_logits.unsqueeze(0).to(self.device)  # [1, seq, vocab]
                        cont_toks = cont_toks[:inplen].unsqueeze(0).to(self.device)  # [1, seq]
                    else:
                        cur_logits = (
                            cur_logits[inplen - contlen : inplen].unsqueeze(0).to(self.device)
                        )  # [1, seq, voc]
                        cont_toks = cont_toks.unsqueeze(0).to(self.device)  # [1, seq]

                    # Check if per-token argmax is exactly equal to continuation
                    greedy_tokens = cur_logits.argmax(dim=-1).to(self.device)
                    # Sometimes the continuation is longer than allowed by the model, we only look at the first tokens
                    max_equal = (greedy_tokens == cont_toks).all().squeeze(0).to(self.device)

                    # Obtain log-probs at the corresponding continuation token indices
                    cur_logits = torch.gather(cur_logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    logits_sum.append(cur_logits.sum())
                    max_equals.append(max_equal)
                    batch_cont_tokens.append(cont_toks)

                # Sync all
                ## Need reshaping before gather
                batched_inputs, len_inputs = self.pad_and_gather(prepared_batch.input_ids)
                max_cont_tokens_length = max(len(c[0]) for c in batch_cont_tokens)
                batch_cont_tokens = torch.cat(
                    [
                        F.pad(c, (0, max_cont_tokens_length - c.shape[1], 0, 0), value=self.tokenizer.pad_token_id)
                        for c in batch_cont_tokens
                    ],
                    dim=0,
                )
                batch_cont_tokens, len_tokens = self.pad_and_gather(batch_cont_tokens)
                ## Can be gathered as such
                logits = torch.tensor(logits_sum, device=self.device)
                max_equal = torch.tensor(max_equals, device=self.device)
                batch_truncated = torch.tensor(prepared_batch.truncated, device=self.device)
                batch_padded = torch.tensor(prepared_batch.padded, device=self.device)
                if self.accelerator:
                    logits = self.accelerator.gather_for_metrics(logits)
                    max_equal = self.accelerator.gather_for_metrics(max_equal)
                    batch_truncated = self.accelerator.gather_for_metrics(batch_truncated)
                    batch_padded = self.accelerator.gather_for_metrics(batch_padded)

                for ix, (logit, cont_tokens, maxe, batched_input, trunc, padded) in enumerate(
                    zip(logits, batch_cont_tokens, max_equal, batched_inputs, batch_truncated, batch_padded)
                ):
                    answer = LoglikelihoodReturn(
                        # todo: we might want to store the logits unsummed
                        result=(float(logit.sum()), bool(maxe)) if return_bool_score else float(logit.sum()),
                        input_tokens=batched_input[: len_inputs[ix]].cpu().tolist(),
                        generated_tokens=cont_tokens[: len_tokens[ix]].cpu().tolist(),
                        truncated_tokens_count=trunc.cpu().item(),
                        padded_tokens_count=padded.cpu().item(),
                    )
                    res.append(answer)

                # Clean up GPUS
                del model_output
                del logits
                del batched_inputs
                del batch_truncated
                del batch_padded

        return dataset.get_original_order(res)

    def prepare_batch(
        self, batch: list[Request], padding_length: int, max_context: Optional[int] = None, single_token: bool = False
    ):
        """Tokenize a batch of inputs and return also the length, truncations and padding"""
        if single_token:
            inputs = [request.tokenized_context for request in batch]
        else:
            inputs = [
                request.tokenized_context + request.tokenized_continuation[:-1] for request in batch
            ]  # The last token (an eos) doesn't need to be given to the model

        input_tokens = []
        attention_masks = []
        input_lengths = []
        truncated = []
        padded = []

        if max_context is None:
            hlog_warn("max_context is None, using max_length")
            max_context = self.max_length

        # Each sample is concatenated and cut to lenght or padded to max_length
        for orig_tokens in inputs:
            truncated.append(max(len(orig_tokens) - max_context, 0))

            # Truncate from the left if needed to fit in the model's context
            tokens = torch.tensor((orig_tokens)[-max_context:], dtype=torch.long).to(self.device)
            sequence_len = tokens.shape[0]

            # We add padding, if needed
            padding_length = padding_length if padding_length is not None else sequence_len

            if padding_length - sequence_len < 0:
                hlog_err(f"Padding length {padding_length} is smaller than input length {sequence_len}")
                raise ValueError("Negative padding")

            padded.append(padding_length - sequence_len)
            # Right padding - it likely would be better to do left padding
            tokens = F.pad(tokens, (0, padding_length - sequence_len), value=self.tokenizer.pad_token_id)

            # We create the attention mask to ignore padding
            mask = tokens == self.tokenizer.pad_token_id
            attention_masks.append(mask)

            input_tokens.append(tokens.unsqueeze(0))  # [1, padding_length]
            input_lengths.append(sequence_len)

        batched_inputs = torch.cat(input_tokens, dim=0)  # [batch, padding_length]
        attention_masks = torch.cat(attention_masks, dim=0)

        return Batch(
            input_ids=batched_inputs,
            input_mask=attention_masks,
            input_lengths=input_lengths,
            truncated=truncated,
            padded=padded,
        )

    def pad_and_gather(self, output_tensor: torch.Tensor) -> torch.Tensor:
        """Gather together tensors of (possibly) various size spread on separate GPUs (first exchange the lengths and then pad and gather)"""
        # Create a tensor of size batch_size, [output_length] * batch_size, for each each process
        length_tensor = torch.tensor([output_tensor.shape[1]] * output_tensor.shape[0], device=self.device)
        if self.accelerator is not None:
            # Gather all the lengths, we end up with a tensor of size num_processes [output_length_1, output_length_2, ...]
            length_tensor = self.accelerator.gather(length_tensor)
        # We pad the output_tensor to the max length
        max_length = length_tensor.max().item()
        output_tensor = F.pad(
            output_tensor, (0, max_length - output_tensor.shape[1], 0, 0), value=self.tokenizer.pad_token_id
        )
        if self.accelerator:
            output_tensor = self.accelerator.gather_for_metrics(output_tensor)
        return output_tensor, length_tensor

    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenReturn]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.

        Args:
            requests (list[Tuple[str, dict]]): _description_

        Returns:
            list[Tuple[float, bool]]: _description_
        """
        for request in requests:
            if request.context == "":
                request.tokenized_context = [self.tokenizer.eos_token_id]
            else:
                request.tokenized_context = self.tok_encode(request.context)

            # Some models tokenizer want a space at the beginning and other not
            continuations = [self._check_continuations_start_space(c) for c in request.choices]

            # We must not accidentally prepend a continuation with a start of sentence token.
            continuations_enc = [self.tok_encode(c, add_special_tokens=False) for c in continuations]
            if any(len(c) > 1 for c in continuations_enc):
                raise ValueError(
                    f"Trying to do single token multiple choice but one choice has several tokens: {continuations_enc}. "
                    "If the additional pre-token is a space, try to set --no_multichoice_continuations_start_space "
                )
            request.tokenized_continuation = continuations_enc

        return self._loglikelihood_single_token(requests, override_bs=override_bs)

    def _loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: int = -1
    ) -> list[LoglikelihoodSingleTokenReturn]:
        dataset = LoglikelihoodSingleTokenDataset(requests=requests, dataset_splits=self.DATASET_SPLITS)
        starting_batch_size = STARTING_BATCH_SIZE
        res = []

        for split_start, split_end in tqdm(dataset.splits_start_end_iterator()):
            context_enc = dataset[0].tokenized_context
            max_context = len(context_enc[-self.max_length :])
            batch_size = self._get_batch_size(override_bs=override_bs, max_input_length=max_context)
            starting_batch_size = batch_size * 2

            dataloader = DataLoader(dataset, batch_size=starting_batch_size, collate_fn=lambda batch: batch)
            if self.accelerator is not None:
                dataloader = self.accelerator.prepare(dataloader)

            for batch in tqdm(dataloader, disable=self.disable_tqdm, position=1):
                prepared_batch = self.prepare_batch(
                    batch, padding_length=max_context, max_context=max_context, single_token=True
                )

                out = self._model_call(prepared_batch.input_ids)  # [batch, padding_length, vocab]
                out = F.log_softmax(out, dim=-1)  # we do a softmax over the options, no the vocab

                batch_probs = []
                batch_cont_tokens = []
                for cur_request, logits, inplen in zip(batch, out, prepared_batch.input_lengths):
                    # Get the last token
                    logits = logits[inplen - 1]  # [vocab]

                    cont_toks = torch.tensor(
                        cur_request.tokenized_continuation, dtype=torch.long, device=self.device
                    ).squeeze(-1)  # [num_choices]

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    probs = torch.gather(logits, dim=0, index=cont_toks)  # [num_choices]

                    # Answer: (log prob, is-exact-match)
                    # probs = torch.nn.functional.softmax(logits.float(), dim=0)  # [num_choices]
                    batch_probs.append(probs)
                    batch_cont_tokens.append(cont_toks)

                # Sync all
                ## Need reshape before gather
                batched_inputs, len_inputs = self.pad_and_gather(prepared_batch.input_ids)
                batch_probs = torch.stack(batch_probs)
                batch_probs, len_probs = self.pad_and_gather(batch_probs)
                batch_cont_tokens = torch.stack(batch_cont_tokens)
                batch_cont_tokens, len_cont = self.pad_and_gather(batch_cont_tokens)

                ## No reshape
                batch_truncated = torch.tensor(prepared_batch.truncated, device=self.device)
                batch_padded = torch.tensor(prepared_batch.padded, device=self.device)
                if self.accelerator:
                    batch_truncated = self.accelerator.gather_for_metrics(batch_truncated)
                    batch_padded = self.accelerator.gather_for_metrics(batch_padded)

                for ix, (probs, cont_tokens, batched_input, trunc, padded) in enumerate(
                    zip(batch_probs, batch_cont_tokens, batched_inputs, batch_truncated, batch_padded)
                ):
                    answer = LoglikelihoodSingleTokenReturn(
                        result=probs[: len_probs[ix]].detach().cpu().tolist(),
                        input_tokens=batched_input[: len_inputs[ix]].cpu().tolist(),
                        generated_tokens=cont_tokens[: len_cont[ix]].cpu().tolist(),
                        truncated_tokens_count=trunc.cpu().item(),
                        padded_tokens_count=padded.cpu().item(),
                    )
                    res.append(answer)

                # Clean up GPUS
                del out
                del batch_probs
                del batched_inputs
                del batch_truncated
                del batch_padded

        return dataset.get_original_order(res)


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        batch: Batch,
    ):
        initial_decoder_input_length = batch.input_ids.shape[1]
        batch_size = batch.input_ids.shape[0]

        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :][:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: list[str],
    batch: Batch,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[MultiTokenEOSCriteria(sequence, tokenizer, batch) for sequence in stop_sequences],
        ]
    )
