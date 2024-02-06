import os
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

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
)
from lighteval.utils import (
    is_accelerate_available,
)
from lighteval.utils_parallelism import find_executable_batch_size


if is_accelerate_available():
    from accelerate.utils import get_max_memory

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TokenSequence = Union[list[int], torch.LongTensor, torch.Tensor, BatchEncoding]

DATASET_SPLITS = 4
STARTING_BATCH_SIZE = 512


class BaseModel(LightevalModel):
    # Default max sequence length setting for when no `max_length` is provided
    # or no max length config setting is found in the model or tokenizer.
    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(
        self,
        config: BaseModelConfig,
        env_config: EnvConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self.accelerator = config.accelerator
        self._batch_size = config.batch_size
        self._max_length = self._init_max_length(config.max_length)
        self._config = config.init_configs(env_config)

        self.add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False
        self.tokenizer = self._create_auto_tokenizer(config, env_config)

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

    @property
    def max_length(self) -> int:
        return self._max_length

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
        return self._DEFAULT_MAX_LENGTH

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

    # Tokenization helpers
    def tok_encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def tok_encode(self, str_to_encode: str | list[str], add_special_tokens: Optional[bool] = None) -> TokenSequence:
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens
        if isinstance(str_to_encode, str):
            return self.tokenizer.encode(str_to_encode, add_special_tokens=add_special_tokens)
        return self.tokenizer(
            str_to_encode,
            padding=True,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )

    def tok_decode(self, tokens: torch.LongTensor) -> list[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

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
        dataset_splits: int = 4,
    ) -> list[GenerateReturn]:
        """
        Generates sequences greedily until a stopping condition is met,
        returning both the generated sequences and the logits.

        Args:
            requests (list[tuple[str, dict]]): A list of input requests,
                where each request is a tuple containing a prompt string and a dictionary of additional parameters.
            override_bs (Optional[int], optional): Overrides the batch size for generation. Defaults to None.
            dataset_splits (int, optional): Number of splits to divide the dataset into for parallel generation. Defaults to 4.

        Returns:
            list[GenerateReturn]: A list of GenerateReturn objects,
                where each object contains the generated sequence and the corresponding logits.
        """

        return self.greedy_until(
            requests,
            returns_logits=True,
            disable_tqdm=self.disable_tqdm,
            override_bs=override_bs,
            dataset_splits=dataset_splits,
        )

    def greedy_until(
        self,
        requests: list[GreedyUntilRequest],
        returns_logits: bool = False,
        override_bs: Optional[int] = None,
        dataset_splits: int = 4,
    ) -> list[GenerateReturn]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            returns_logits (bool, optional): Whether to return the logits of the generated responses. Defaults to False.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.
            dataset_splits (int, optional): Number of splits to divide the dataset into. Defaults to 4.

        Returns:
            list[GenerateReturn]: list of generated responses.
        """
        for request in requests:
            request.stop_sequence = request.stop_sequence + [self.tokenizer.eos_token]
        dataset = GenerativeTaskDataset(requests=requests, dataset_splits=dataset_splits)
        starting_batch_size = STARTING_BATCH_SIZE
        results = []

        for split_start, split_end in tqdm(
            dataset.splits_start_end_iterator(),
            total=DATASET_SPLITS,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            # longest context in the current split
            context = dataset[0].context
            max_gen = dataset[0].generation_size

            longest_context_continuation_size_in_split = len(context) + max_gen
            max_context_continuation_size_allowed = min(longest_context_continuation_size_in_split, self.max_length)
            batch_size = self._get_batch_size(
                override_bs=override_bs,
                max_input_length=max_context_continuation_size_allowed,
                starting_batch_size=starting_batch_size,
            )
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
                max_context_allowed_by_model = self.max_length - max_generated_tokens

                tokenized = self.tokenizer(
                    context,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_context_allowed_by_model,
                    add_special_tokens=self.add_special_tokens,
                ).to(self.device)

                prepared_batch = Batch(
                    input_ids=tokenized["input_ids"],
                    input_lengths=[len(item == 1) for item in tokenized["attention_mask"]],
                    input_mask=tokenized["attention_mask"],
                    truncated=[0] * len(tokenized["input_ids"]),
                    padded=[0] * len(tokenized["input_ids"]),
                )

                # responses, logits and input_ids have all been gathered accross GPUs already
                # but we also grab the original length of these vectors, which have been padded
                # while being gathered - the added info
                responses, logits, input_ids, len_resps, len_logits, len_ids = self._model_generate(
                    input_ids=prepared_batch.input_ids,
                    attention_mask=prepared_batch.input_mask,
                    max_tokens=max_generated_tokens,
                    stop=stop_tokens,
                    returns_logits=returns_logits,
                )

                if returns_logits:
                    logits = logits.cpu().numpy()

                batch_truncated = torch.tensor(prepared_batch.truncated, device=self.device)
                if self.accelerator:
                    batch_truncated = self.accelerator.gather_for_metrics(batch_truncated)
                batch_padded = torch.tensor(prepared_batch.padded, device=self.device)
                if self.accelerator:
                    batch_padded = self.accelerator.gather_for_metrics(batch_padded)

                for ix, (response, batched_input, trunc, padded) in enumerate(
                    zip(responses, input_ids, batch_truncated, batch_padded)
                ):
                    # Ensure the generated responses do not contain the stop sequences.
                    response = response[: len_resps[ix]]
                    decoded_response = self.tok_decode([response])[0]

                    for term in stop_tokens:
                        decoded_response = decoded_response.split(term)[0]

                    cur_response = GenerateReturn(
                        result=decoded_response,
                        logits=logits[ix][: len_logits[ix]] if returns_logits else None,
                        generated_tokens=response,
                        input_tokens=batched_input[: len_ids[ix]],
                        truncated_tokens_count=trunc.cpu().item(),
                        padded_tokens_count=padded.cpu().item(),
                    )
                    results.append(cur_response)

        return dataset.get_original_order(results)

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
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
                # DO NOT CHANGE THE FOLLOWING LINE!
                # It is mandatory for compatibility with the harness!!!
                request.tokenized_context, request.tokenized_continuation = self.tok_encode_pair(
                    request.context, request.choice
                )

            # tokenized_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(requests, override_bs=override_bs, dataset_splits=DATASET_SPLITS)

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs=None
    ) -> list[LoglikelihoodReturn]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""

        for request in requests:  # tuple of one elem
            request.tokenized_context = [self.tokenizer.eos_token_id]  # Fake context
            request.tokenized_continuation = self.tok_encode(request.context)
            # tokenized_reqs.append((("", context), fake_context_enc, context_enc))

        results = self._loglikelihood_tokens(
            requests,
            override_bs=override_bs,
            return_bool_score=False,
            dataset_splits=DATASET_SPLITS,
        )
        return results

    def _loglikelihood_tokens(
        self,
        requests,
        override_bs: int = -1,
        dataset_splits: int = 4,
        return_bool_score: bool = True,
    ) -> list[LoglikelihoodReturn]:
        dataset = LoglikelihoodDataset(requests=requests, dataset_splits=dataset_splits)
        starting_batch_size = STARTING_BATCH_SIZE
        res = []

        for split_start, split_end in tqdm(dataset.splits_start_end_iterator()):
            context_enc = dataset[0].tokenized_context
            continuation_enc = dataset[0].tokenized_continuation
            max_context = len((context_enc + continuation_enc)[-(self.max_length + 1) :][:-1])

            batch_size = self._get_batch_size(
                override_bs=override_bs, max_input_length=max_context, starting_batch_size=starting_batch_size
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)
            starting_batch_size = batch_size * 2

            if self.accelerator:
                dataloader = self.accelerator.prepare(dataloader)

            for batch in tqdm(dataloader, disable=self.disable_tqdm):
                inputs = [
                    request.tokenized_context + request.tokenized_continuation[:-1] for request in batch
                ]  # The last token doesn't need to be input in the model
                prepared_batch = self.prepare_batch(inputs, padding_length=max_context, max_context=max_context)

                out = self._model_call(prepared_batch.input_ids)
                multi_logits = F.log_softmax(out, dim=-1)  # [batch, padding_length, vocab]

                logits_sum = []
                max_equals = []
                batch_cont_tokens = []
                for cur_request, logits, inplen in zip(batch, multi_logits, prepared_batch.input_lengths):
                    cont_toks = cur_request.tokenized_continuation
                    # We only look at the continuation tokens
                    contlen = len(cont_toks)
                    if contlen > inplen:
                        # continuation is longer than the allowed context size, everything is a continuation
                        logits = logits.unsqueeze(0).to(self.device)  # [1, seq, vocab]
                        cont_toks = (
                            torch.tensor(cont_toks, dtype=torch.long, device=self.device)[:inplen]
                            .unsqueeze(0)
                            .to(self.device)
                        )  # [1, seq]
                    else:
                        logits = logits[inplen - contlen : inplen].unsqueeze(0).to(self.device)  # [1, seq, vocab]
                        cont_toks = (
                            torch.tensor(cont_toks, dtype=torch.long, device=self.device).unsqueeze(0).to(self.device)
                        )  # [1, seq]

                    # Check if per-token argmax is exactly equal to continuation
                    greedy_tokens = logits.argmax(dim=-1).to(self.device)
                    # Sometimes the continuation is longer than allowed by the model, we only look at the first tokens
                    max_equal = (greedy_tokens == cont_toks).all().squeeze(0).to(self.device)

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    logits_sum.append(logits.sum())
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
                        result=(float(logit.sum()), bool(maxe)) if return_bool_score else float(logit.sum()),
                        input_tokens=batched_input[: len_inputs[ix]].cpu().tolist(),
                        generated_tokens=cont_tokens[: len_tokens[ix]].cpu().tolist(),
                        truncated_tokens_count=trunc.cpu().item(),
                        padded_tokens_count=padded.cpu().item(),
                    )
                    res.append(answer)

                # Clean up GPUS
                del out
                del logits
                del batched_inputs
                del batch_truncated
                del batch_padded

        return dataset.get_original_order(res)

    def prepare_batch(self, batch: list[str], padding_length: int, max_context: Optional[int] = None):
        """Tokenize a batch of inputs and return also the length, truncations and padding"""
        inputs = []
        attention_masks = []
        input_lengths = []
        truncated = []
        padded = []

        if max_context is None:
            hlog_warn("max_context is None, using max_length")
            max_context = self.max_length

        # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
        # tensors, then we pack them together into a batch, call the model, and then pick it all apart
        # again because vectorizing is annoying

        # Each sample is concatenated and cut to lenght or padded to max_length
        for tokens in batch:
            truncated.append(max(len(tokens) - max_context, 0))

            # how this all works:
            #          CTX      CONT
            # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
            # gpt2    \               \
            # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
            # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

            # when too long to fit in context, truncate from the left
            inp = torch.tensor(
                (tokens)[-max_context:],  # [:-1],
                dtype=torch.long,
            ).to(self.device)

            (inplen,) = inp.shape

            # since in _collate we make sure length is descending, the longest is always the first one.
            padding_length = padding_length if padding_length is not None else inplen

            if padding_length - inplen < 0:
                hlog_err(f"{tokens=}")
                hlog_err(f"{max_context=}")
                hlog_err(f"{padding_length=}")
                hlog_err(f"{inp.shape=}")
                hlog_err(f"Padding length {padding_length} is smaller than input length {inplen}")
                raise ValueError("Negative padding")

            padded.append(padding_length - inplen)

            # pad length from seq to padding_length
            att = torch.tensor([1] * len(inp) + [0] * (padding_length - inplen), dtype=torch.long).to(self.device)
            inp = torch.cat(
                [
                    inp,  # [seq]
                    torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device),  # [padding_length - seq]
                ],
                dim=0,
            )
            attention_masks.append(att.unsqueeze(0))

            inputs.append(inp.unsqueeze(0))  # [1, padding_length]
            input_lengths.append(inplen)

        batched_inputs = torch.cat(inputs, dim=0)  # [batch, padding_length]
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
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: int = -1, dataset_splits: int = 4
    ) -> list[LoglikelihoodSingleTokenReturn]:
        dataset = LoglikelihoodSingleTokenDataset(requests=requests, dataset_splits=dataset_splits)
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
                inputs = [context_enc for _, context_enc, _ in batch]

                prepared_batch = self.prepare_batch(inputs, padding_length=max_context, max_context=max_context)

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

    def _model_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_tokens: int,
        stop: Optional[list[str]] = None,
        returns_logits: Optional[bool] = False,
    ) -> TokenSequence:
        # max_tokens is the maximum number of *new* tokens
        # Ensure that the context does not encroach into the `space` for the generation.`
        # input_ids = inputs["input_ids"][:, max_tokens - self.max_length :].to(self.device)
        # if inputs["input_ids"].shape != input_ids.shape:
        #     hlog_warn(
        #         f"The input was truncated from {inputs['input_ids'].shape} to {input_ids.shape}, with a model maximum context length of {self.max_length} and max {max_tokens} to generate."
        #     )
        # attention_mask = inputs["attention_mask"][:, max_tokens - self.max_length :].to(self.device)

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer,
            stop_sequences=stop,
            initial_decoder_input_length=input_ids.shape[1],
            batch_size=input_ids.shape[0],
        )

        # Depending on whether accelerate is used or not
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=returns_logits,
        )
        if returns_logits:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
        generations = outputs.sequences[:, input_ids.size(1) :]
        generations, len_gens = self.pad_and_gather(generations)
        input_ids, len_ids = self.pad_and_gather(input_ids)

        if returns_logits:
            # Used input_ids to get its max_length
            transition_scores, len_logits = self.pad_and_gather(transition_scores)
        else:
            transition_scores, len_logits = None, None

        return generations, transition_scores, input_ids, len_gens, len_logits, len_ids


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ):
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
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(sequence, tokenizer, initial_decoder_input_length, batch_size)
                for sequence in stop_sequences
            ],
        ]
    )
