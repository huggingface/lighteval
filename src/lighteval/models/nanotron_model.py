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

# ruff: noqa: C901
import os
import time
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
import transformers
from datasets.download.streaming_download_manager import xPath
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, BatchEncoding

from lighteval.config.lighteval_config import FullNanotronConfig
from lighteval.data import (
    GenDistributedSampler,
    GenerativeTaskDatasetNanotron,
    LoglikelihoodDataset,
    LoglikelihoodSingleTokenDataset,
)
from lighteval.logging.hierarchical_logger import hlog_err, hlog_warn
from lighteval.models.base_model import LightevalModel, ModelInfo
from lighteval.models.model_output import (
    Batch,
    GenerativeResponse,
    LoglikelihoodResponse,
    LoglikelihoodSingleTokenResponse,
)
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
)
from lighteval.utils.imports import is_nanotron_available
from lighteval.utils.parallelism import find_executable_batch_size
from lighteval.utils.utils import EnvConfig, as_list


os.environ["TOKENIZERS_PARALLELISM"] = "false"

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

if is_nanotron_available():
    from nanotron import distributed as dist
    from nanotron import logging
    from nanotron.generation.decode import decode_tokenized
    from nanotron.logging import human_format, log_rank
    from nanotron.models import build_model
    from nanotron.parallel.context import ParallelContext
    from nanotron.parallel.parameters import sanity_check
    from nanotron.parallel.pipeline_parallel.block import get_min_max_rank
    from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
    from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
    from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state, set_random_seed
    from nanotron.serialize import load_weights
    from nanotron.trainer import CONFIG_TO_MODEL_CLASS, mark_tied_parameters

logger = logging.get_logger(__name__)


class NanotronLightevalModel(LightevalModel):
    # Default max sequence length setting for when no `max_length` is provided
    # or no max length config setting is found in the model or tokenizer.
    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(
        self,
        checkpoint_path: str,
        nanotron_config: FullNanotronConfig,
        parallel_context: ParallelContext,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        add_special_tokens: Optional[bool] = True,
        dtype: Optional[Union[str, torch.dtype]] = None,
        trust_remote_code: bool = False,
        debug_one_layer_model: bool = False,
        model_class: Optional[Type] = None,
        env_config: EnvConfig = None,
    ):
        """Initializes a nanotron model for evaluation.
        Args:
        """
        model_args = nanotron_config.nanotron_config.model
        tokenizer = nanotron_config.nanotron_config.tokenizer
        lighteval_config = nanotron_config.lighteval_config
        parallel_config = nanotron_config.lighteval_config.parallelism

        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self.parallel_config = parallel_config
        self.parallel_context = parallel_context

        if parallel_config.pp > 1:
            # To implement PP parallelism we need to think about how we want to sync the output for the PP ranks without outputs
            raise ValueError("PP parallelism is not supported yet")

        # multichoice_continuations_start_space can be True (forcing space), False (forcing no space) or None (no forcing)
        multichoice_continuations_start_space = lighteval_config.tasks.multichoice_continuations_start_space

        self.generation_config = lighteval_config.generation
        if isinstance(self.generation_config, dict):
            raise ValueError("We don't support yet generation configs per tasks")

        if dtype is None:
            dtype = torch.bfloat16
        self.dtype = dtype

        self.model_config = model_args.model_config
        if debug_one_layer_model:
            self.model_config.num_hidden_layers = 1

        self._add_special_tokens = add_special_tokens
        self._tokenizer = self._create_auto_tokenizer(
            pretrained=tokenizer.tokenizer_name_or_path,
            env_config=env_config,
            trust_remote_code=trust_remote_code,
        )
        self._tokenizer.model_max_length = self.max_length

        model_config_cls = self.model_config.__class__.__name__
        if model_class is not None:
            CONFIG_TO_MODEL_CLASS[model_config_cls] = model_class
        if model_config_cls not in CONFIG_TO_MODEL_CLASS:
            raise ValueError(
                f"Unsupported model config {model_config_cls}. Only {CONFIG_TO_MODEL_CLASS.keys()} are supported"
            )

        log_rank(
            "Building model",
            logger=logger,
            level=logging.WARNING,
            rank=0,
        )

        # Set random states
        set_random_seed(42)

        # Get synchronized random states
        if parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
            random_states = RandomStates(
                {
                    "tp_synced": get_synced_random_state(
                        random_state=get_current_random_state(), pg=parallel_context.tp_pg
                    )
                }
            )
        else:
            # We don't need to sync across TP when using sequence parallel (REDUCE_SCATTER)
            random_states = RandomStates({})

        model = build_model(
            model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
                config=self.model_config,
                parallel_context=parallel_context,
                parallel_config=parallel_config,
                random_states=random_states,
            ),
            dtype=dtype,
            parallel_context=parallel_context,
        )

        # Mark some parameters as tied
        # TODO @nouamane: this is only needed for training, can we just mark params as NanotronParameter instead?
        mark_tied_parameters(model=model, parallel_context=parallel_context, parallel_config=parallel_config)

        log_rank(
            "Sanity checks on model",
            logger=logger,
            level=logging.WARNING,
            rank=0,
        )
        # Sanity check model
        sanity_check(root_module=model)

        # Load checkpoint
        log_rank(
            f"Loading checkpoint from {checkpoint_path}:",
            logger=logger,
            level=logging.WARNING,
            rank=0,
        )
        load_weights(model=model, parallel_context=parallel_context, root_folder=xPath(checkpoint_path))
        model.eval()

        # We don't need the loss
        self.model = model.transformer if model_config_cls == "FalconConfig" else model.model

        # Get the input and output ranks for the model (when using PP parallelism)
        self.input_pp_rank, self.output_pp_rank = get_min_max_rank(module=self.model)

        self.multichoice_continuations_start_space = multichoice_continuations_start_space

        self.model_info = ModelInfo(
            model_name=f"{nanotron_config.nanotron_config.general.run}/{nanotron_config.nanotron_config.general.step}"
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        tokenizer: Optional[str] = None,
        env_config: EnvConfig = None,
        trust_remote_code: bool = False,
    ) -> transformers.PreTrainedTokenizer:
        """Returns a pre-trained tokenizer from a pre-trained tokenizer configuration."""

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained if tokenizer is None else tokenizer,
                cache_dir=env_config.cache_dir,
                token=env_config.token,
                trust_remote_code=trust_remote_code,
            )
        except RecursionError:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained if tokenizer is None else tokenizer,
                cache_dir=env_config.cache_dir,
                token=env_config.token,
                unk_token="<unk>",
                trust_remote_code=trust_remote_code,
            )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return tokenizer

    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            return False

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        if self._max_length is not None:
            return self._max_length
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model_config, attr):
                return getattr(self.model_config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def device(self) -> Union[int, str, torch.device]:
        return "cuda"

    def _get_batch_size(self, max_input_length: int, override_bs: int = 0, starting_batch_size: int = 512) -> int:
        if override_bs:
            return override_bs
        logger.warning("Detecting largest batch size")

        @find_executable_batch_size(
            starting_batch_size=starting_batch_size
        )  # if OOM, then halves batch_size and tries again
        def forward_batch(batch_size):
            logger.warning(f"Testing batch size {batch_size}")
            test_batch = torch.ones(
                (batch_size + int(0.1 * batch_size), max_input_length), device=self.device
            ).long()  # We add 10% for marging :)
            F.log_softmax(self._model_call(test_batch).float(), dim=-1).cpu()
            return batch_size

        batch_size = forward_batch()
        logger.warning("Determined largest batch size: %d", batch_size)
        return batch_size

    def tok_encode(self, string: str, add_special_tokens: Optional[bool] = None) -> TokenSequence:
        # TODO: Merge `tok_encode_batch` here.
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def homogeneize_ending_conditions(self, ending_condition: tuple | dict | list | str) -> tuple[list, int]:
        """Ending conditions are submitted in several possible formats.
        By default in lighteval we pass them as tuples (stop sequence, max number of items).
        In the harness they sometimes are passed as dicts {"until": .., "max_length": ...} or
        as only ending conditions, either lists or strings.
        Here, we convert all these formats to a tuple containing a list of ending conditions,
        and a float for the max length allowed.
        """
        max_tokens, stop_sequences = None, None
        # Filling with input values or default
        if isinstance(ending_condition, tuple) and len(ending_condition) == 2:
            stop_sequence_arg, max_gen_tokens_arg = ending_condition
            stop_sequences = as_list(stop_sequence_arg)
            max_tokens = max_gen_tokens_arg
        elif isinstance(ending_condition, dict):  # Tasks in the harness sometimes pass a dict to rf.greedy_until
            try:
                stop_sequences = as_list(ending_condition["until"])
            except KeyError:
                stop_sequences = []
            try:
                max_tokens = ending_condition["max_length"]
            except KeyError:
                max_tokens = self._max_gen_toks
        else:  # only gave stop sequences  as an ending condition
            stop_sequences = as_list(ending_condition)

        # Managing empty cases
        if max_tokens is None:
            max_tokens = self._max_gen_toks
        if stop_sequences is None or (len(stop_sequences) == 1 and stop_sequences[0] is None):  # or num_fewshot == 0:
            stop_tokens = [self.eot_token]
        else:
            stop_tokens = list(stop_sequences) + [self.eot_token]

        assert isinstance(max_tokens, int)
        assert isinstance(stop_tokens, list)

        return stop_tokens, max_tokens

    def _check_continuations_start_space(self, continuation: str) -> str:
        """Some models tokenizer want a space at the beginning and other not. We update this if needed here.
        multichoice_continuations_start_space can be:
        - True (add a space if these isn't one)
        - False (remove a space if there is one)
        - None (Don't touch - default)
        Todo: find a way to add this back WITHOUT breaking compatibility with the harness
        """
        if self.multichoice_continuations_start_space is not None:
            if self.multichoice_continuations_start_space and continuation[0] != " ":
                continuation = " " + continuation
            if not self.multichoice_continuations_start_space and continuation[0] == " ":
                continuation = continuation.lstrip()
        return continuation

    def loglikelihood_single_token(
        self, requests: List[Tuple[str, dict]], override_bs=0
    ) -> List[LoglikelihoodSingleTokenResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.

        Args:
            requests (List[Tuple[str, dict]]): _description_

        Returns:
            List[Tuple[float, bool]]: _description_
        """
        for request in tqdm(
            requests, desc="Tokenizing", disable=bool(dist.get_rank(self.parallel_context.world_pg) != 0)
        ):
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
                    "If the additional pre-token is a space, try to set multichoice_continuations_start_space=False in the model parameters "
                )
            request.tokenized_continuation = continuations_enc

        return self._loglikelihood_single_token(
            requests,
            override_bs=override_bs,
            disable_tqdm=bool(dist.get_rank(self.parallel_context.world_pg) != 0),
        )

    def loglikelihood(self, requests: List[LoglikelihoodRequest], override_bs=None) -> List[LoglikelihoodResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        for request in tqdm(
            requests, desc="Tokenizing", disable=bool(dist.get_rank(self.parallel_context.world_pg) != 0)
        ):
            if request.context == "":
                request.tokenized_context = [self.tokenizer.eos_token_id]
                request.tokenized_continuation = self.tok_encode(request.choice)
            else:
                # The following line is mandatory for compatibility with the harness
                request.tokenized_context, request.tokenized_continuation = self.tok_encode_pair(
                    request.context, request.choice
                )

        return self._loglikelihood_tokens(
            requests,
            override_bs=override_bs,
            disable_tqdm=bool(dist.get_rank(self.parallel_context.world_pg) != 0),
        )

    def loglikelihood_rolling(
        self, requests: List[LoglikelihoodRollingRequest], override_bs: int = 0
    ) -> List[LoglikelihoodResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        for request in tqdm(
            requests, desc="Tokenizing", disable=bool(dist.get_rank(self.parallel_context.world_pg) != 0)
        ):  # tuple of one elem
            request.tokenized_context = [self.tokenizer.eos_token_id]  # Fake context
            request.tokenized_continuation = self.tok_encode(request.context)

        results = self._loglikelihood_tokens(
            requests,
            override_bs=override_bs,
            disable_tqdm=bool(dist.get_rank(self.parallel_context.world_pg) != 0),
            return_bool_score=False,
        )
        return results

    def prepare_batch(
        self,
        batch: List[str],
        padding_length: int,
        max_context: Optional[int] = None,
        full_attention_masks: bool = False,
        pad_on_left: bool = False,
    ) -> Batch:
        """Tokenize a batch of inputs and return also the length, truncations and padding

        We truncate to keep only at most `max_context` tokens
        We pad to `padding_length` tokens
        """
        # if not full_attention_masks:
        #     raise ValueError(
        #         "Only full attention masks are supported for now - fix Flash Attention 2 support for more"
        #     )
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)

        if current_pp_rank != self.input_pp_rank:
            # Nothing to do we're not on input GPUs
            return TensorPointer(self.input_pp_rank), TensorPointer(self.input_pp_rank), -1, -1, -1

        inputs = []
        attention_masks = []
        input_lengths = []
        truncated = []
        padded = []

        if max_context is None:
            max_context = self.max_length

        if max_context % self.parallel_config.tp != 0:
            # We need to round up to the next multiple of self.parallel_config.tp
            if (max_context + (self.parallel_config.tp - max_context % self.parallel_config.tp)) < self.max_length:
                # We can add some tokens
                max_context = max_context + (self.parallel_config.tp - max_context % self.parallel_config.tp)
            else:
                # We need to remove some tokens
                max_context = max_context - (max_context % self.parallel_config.tp)

        if padding_length % self.parallel_config.tp != 0:
            # We need to round up to the next multiple of self.parallel_config.tp
            if (
                padding_length + (self.parallel_config.tp - padding_length % self.parallel_config.tp)
            ) < self.max_length:
                # We can add some tokens
                padding_length = padding_length + (self.parallel_config.tp - padding_length % self.parallel_config.tp)
            else:
                # We need to remove some tokens
                padding_length = padding_length - (padding_length % self.parallel_config.tp)

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
            )

            (inplen,) = inp.shape

            # since in _collate we make sure length is descending, the longest is always the first one.
            padding_length = padding_length if padding_length is not None else inplen
            if padding_length - inplen < 0:
                raise ValueError("Negative padding")
            padded.append(padding_length - inplen)

            # pad length from seq to padding_length
            if full_attention_masks:
                att = torch.tensor([1] * padding_length, dtype=torch.bool)
            else:
                if pad_on_left:
                    att = torch.tensor([0] * (padding_length - inplen) + [1] * len(inp), dtype=torch.bool)
                else:
                    att = torch.tensor([1] * len(inp) + [0] * (padding_length - inplen), dtype=torch.bool)
            if pad_on_left:
                inp = torch.cat(
                    [
                        torch.zeros(padding_length - inplen, dtype=torch.long),  # [padding_length - seq]
                        inp,  # [seq]
                    ],
                    dim=0,
                )
            else:
                inp = torch.cat(
                    [
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long),  # [padding_length - seq]
                    ],
                    dim=0,
                )
            attention_masks.append(att.unsqueeze(0))

            inputs.append(inp.unsqueeze(0))  # [1, padding_length]
            input_lengths.append(inplen)

        input_ids = torch.cat(inputs, dim=0).to("cuda")  # [batch, padding_length]
        input_mask = torch.cat(attention_masks, dim=0).to("cuda")

        return Batch(
            input_ids=input_ids, input_mask=input_mask, input_lengths=input_lengths, truncated=truncated, padded=padded
        )

    def gather(self, output_tensor: torch.Tensor, process_group: dist.ProcessGroup = None) -> torch.Tensor:
        """Gather together tensors of (possibly) various size spread on separate GPUs (first exchange the lengths and then pad and gather)"""
        if process_group is None:
            process_group = self.parallel_context.dp_pg
        output_tensor = output_tensor.contiguous()
        gathered_outputs = [torch.zeros_like(output_tensor) for _ in range(process_group.size())]
        dist.all_gather(gathered_outputs, output_tensor, group=process_group, async_op=False)
        gathered_outputs = torch.cat(gathered_outputs)
        return gathered_outputs

    def pad_and_gather(self, output_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather together tensors of (possibly) various size spread on separate GPUs (first exchange the lengths and then pad and gather)"""
        if output_tensor.ndim == 1:
            length_tensor = torch.tensor([output_tensor.shape[0]], device=self.device)
        elif output_tensor.ndim == 2:
            length_tensor = torch.tensor([output_tensor.shape[1]] * output_tensor.shape[0], device=self.device)
        else:
            raise ValueError("Unsupported tensor shape")
        gathered_length = self.gather(length_tensor)
        max_length = gathered_length.max().item()

        if output_tensor.ndim == 1:
            output_tensor = F.pad(
                output_tensor, (0, max_length - output_tensor.shape[0]), value=self.tokenizer.pad_token_id
            )
        else:
            output_tensor = F.pad(
                output_tensor, (0, max_length - output_tensor.shape[1], 0, 0), value=self.tokenizer.pad_token_id
            )
        gathered_outputs = self.gather(output_tensor)

        return gathered_outputs, gathered_length

    def _get_subsets(self, dataset, num_dataset_splits):
        total_length = len(dataset)
        subset_length = int(float(total_length) / float(num_dataset_splits)) + 1
        if subset_length < self.parallel_context.dp_pg.size():
            # We need at least one subset sample per DP process
            subset_length = self.parallel_context.dp_pg.size()
        return total_length, subset_length

    @torch.inference_mode()
    def _loglikelihood_single_token(
        self, requests, disable_tqdm: bool = False, override_bs: int = 0, num_dataset_splits: int = 1
    ) -> List[LoglikelihoodSingleTokenResponse]:
        dataset = LoglikelihoodSingleTokenDataset(requests=requests)
        res = []

        # Dataset is sorted in descending size.
        # every 20-25% of the dataset we try to double the batch size for speed up
        printed_error = False
        starting_batch_size = 512

        total_length, subset_length = self._get_subsets(dataset, num_dataset_splits)

        for s, subset_start in enumerate(
            tqdm(
                range(0, total_length, subset_length),
                disable=disable_tqdm,
                position=0,
                desc=f"loglikelihood_single_token -- for Node {dist.get_rank(self.parallel_context.world_pg)}",
            )
        ):
            dataset.split_start = subset_start
            dataset.split_end = min(subset_start + subset_length, total_length)

            # automatic (variable) batch size detection for vectorization
            # pull longest context sample from request
            context_enc = dataset[0].tokenized_context
            max_context = len(context_enc[-self.max_length :])
            batch_size = self._get_batch_size(
                override_bs=override_bs, max_input_length=max_context, starting_batch_size=starting_batch_size
            )

            starting_batch_size = batch_size * 2  # for the next round

            # For the DP replicas
            distributed_sampler = DistributedSampler(
                dataset,
                num_replicas=self.parallel_context.dp_pg.size(),
                rank=dist.get_rank(self.parallel_context.dp_pg),
                shuffle=False,
                drop_last=False,
            )
            to_remove_at_the_end = distributed_sampler.total_size - len(dataset)

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=distributed_sampler,
                collate_fn=lambda batch: batch,
                drop_last=False,
            )

            tq = tqdm(
                dataloader,
                disable=disable_tqdm,
                position=1,
                desc=f"loglikelihood_single_token in subset {s} Node {dist.get_rank(self.parallel_context.world_pg)}",
            )

            for j, batch_data in enumerate(tq):
                if j < 3:
                    log_rank(
                        f"Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MB. Peak reserved memory: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MB",
                        logger=logger,
                        level=logging.INFO,
                        group=self.parallel_context.world_pg,
                        rank=0,
                    )
                iteration_start_time = time.time()
                inputs = [item.tokenized_context for item in batch_data]

                batch_model = self.prepare_batch(
                    inputs, padding_length=max_context, max_context=max_context, full_attention_masks=True
                )
                # batched_inputs, batch_attention, input_lengths, truncated, padded

                out = self.model(input_ids=batch_model.input_ids, input_mask=batch_model.input_mask)

                if dist.get_rank(self.parallel_context.pp_pg) == self.output_pp_rank:
                    # This process got outputs

                    # Gather all the output accross TP
                    out = out.transpose(0, 1).contiguous()  # [batch, seq_length, vocab]

                    gathered_out = [torch.zeros_like(out) for _ in range(self.parallel_context.tp_pg.size())]
                    dist.all_gather(gathered_out, out, group=self.parallel_context.tp_pg, async_op=False)
                    out = torch.cat(gathered_out, dim=-1)

                    out = F.log_softmax(out, dim=-1)  # [batch, padding_length, vocab]

                    batch_probs = []
                    batch_cont_tokens = []
                    for i, (batch, logits, inplen) in enumerate(zip(batch_data, out, batch_model.input_lengths)):
                        context = batch.context
                        cont_toks = batch.tokenized_continuation
                        # Get the last token
                        logits = logits[inplen - 1]  # [vocab]

                        cont_toks = torch.tensor(cont_toks, dtype=torch.long, device=self.device).squeeze(
                            -1
                        )  # [num_choices]

                        top_k = torch.topk(logits, 20)[1].tolist()
                        if any(bool(el not in top_k) for el in cont_toks) and not printed_error:
                            top_toks_str = "|".join(self.tokenizer.decode(tt).replace("\n", "") for tt in top_k)
                            cont_toks_str = "|".join(
                                self.tokenizer.decode(tt).replace("\n", "") for tt in cont_toks.tolist()
                            )
                            logger.error(
                                f"Not all the solutions are in the top 20 most likely tokens on rank {dist.get_rank(self.parallel_context.world_pg)} "
                                f"Batch {j} element {i}: {context[0][-150:]} "
                                f"top_tokens: {top_toks_str}\ncont_tokens: {cont_toks_str}"
                            )
                            # for i in range(inplen - 50, min(len(out[1]), inplen + 10)):
                            #     print(
                            #         i,
                            #         "|".join(
                            #             self.tokenizer.decode(tt) for tt in torch.topk(out[1][i], 10)[1].tolist()
                            #         ),
                            #     )

                            printed_error = True
                        # Obtain log-probs at the corresponding continuation token indices
                        # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                        probs = torch.gather(logits, dim=0, index=cont_toks)  # [num_choices]

                        # Answer: (log prob, is-exact-match)
                        # probs = torch.nn.functional.softmax(logits.float(), dim=0)  # [num_choices]
                        batch_probs.append(probs)
                        batch_cont_tokens.append(cont_toks)

                    # Sync all
                    # Need reshape/padding both locally (on each node) and generally accross nodes
                    batched_inputs, _ = self.pad_and_gather(batch_model.input_ids)
                    lengths = torch.tensor(batch_model.input_lengths, device=self.device)
                    batched_lengths = self.gather(lengths)

                    probs_lengths = torch.tensor([len(b) for b in batch_probs], device=self.device)
                    max_probs_lengths = probs_lengths.max().item()
                    batched_probs_lengths = self.gather(probs_lengths)
                    batch_probs = torch.stack(
                        [F.pad(c, (0, max_probs_lengths - c.shape[0]), value=0) for c in batch_probs],
                        dim=0,
                    )
                    batch_probs, _ = self.pad_and_gather(batch_probs)

                    cont_tokens_lengths = torch.tensor([len(b) for b in batch_cont_tokens], device=self.device)
                    max_cont_tokens_lengths = cont_tokens_lengths.max().item()
                    batched_cont_tokens_lengths = self.gather(cont_tokens_lengths)
                    batch_cont_tokens = torch.stack(
                        [F.pad(c, (0, max_cont_tokens_lengths - c.shape[0]), value=0) for c in batch_probs],
                        dim=0,
                    )
                    batch_cont_tokens, _ = self.pad_and_gather(batch_cont_tokens)

                    # No reshape
                    batch_truncated = torch.tensor(batch_model.truncated, device=self.device)
                    batch_truncated = self.gather(batch_truncated)
                    batch_padded = torch.tensor(batch_model.padded, device=self.device)
                    batch_padded = self.gather(batch_padded)

                    batch_res = []
                    for ix, (probs, cont_tokens, batched_input, trunc, padded, batched_length) in enumerate(
                        zip(
                            batch_probs,
                            batch_cont_tokens,
                            batched_inputs,
                            batch_truncated,
                            batch_padded,
                            batched_lengths,
                        )
                    ):
                        answer = LoglikelihoodSingleTokenResponse(
                            result=probs[: batched_probs_lengths[ix]].numpy(force=True),
                            input_tokens=batched_input[: batched_length.item()].numpy(force=True),
                            generated_tokens=cont_tokens[: batched_cont_tokens_lengths[ix]].numpy(force=True),
                            truncated_tokens_count=trunc.cpu().item(),
                            padded_tokens_count=padded.cpu().item(),
                        )
                        batch_res.append(answer)

                    # Sort batches back when we add then in res - because of the DistributedSampler, they are interleaved in the results:
                    # Ex with DP=3 and a batch of size 3 we end up with 0 2 4 6 1 3 5 7 instead of 0 1 2 3 4 5 6 7
                    assert len(batch_res) % self.parallel_context.dp_pg.size() == 0
                    for i in range(len(batch_res) // self.parallel_context.dp_pg.size()):
                        for j in range(self.parallel_context.dp_pg.size()):
                            res.append(batch_res[i + j * (len(batch_res) // self.parallel_context.dp_pg.size())])

                    # A bit of logging
                    elapsed_time_per_iteration_ms = (time.time() - iteration_start_time) * 1000
                    tokens_per_sec = batched_inputs.numel() / (elapsed_time_per_iteration_ms / 1000)

                    tq.desc = f"loglikelihood_single_token Subset {s} Node {dist.get_rank(self.parallel_context.world_pg)} - {human_format(tokens_per_sec)} tokens/s"

                    # Clean up GPUs
                    del out
                    del batch_probs
                    del batched_inputs
                    del batch_cont_tokens
                    del batch_truncated
                    del batch_padded

            # At the end of the subset, remove the additional samples we may have added to make the dataset divisible by the number of processes
            assert to_remove_at_the_end >= 0
            res = res[: len(res) - to_remove_at_the_end]

        # if dist.get_rank(self.parallel_context.tp_pg) == 0:
        #     for i, r in enumerate(res):
        #         print(f"i {i} results: {r.result[-30:]}")
        #         print(f"i {i} input_tokens: {r.input_tokens[-(r.padded + 10):-r.padded]}")
        #         print(f"i {i} cont_tokens: {r.cont_tokens[-30:]}")
        #         print(f"i {i} truncated: {r.truncated}")
        #         print(f"i {i} padded: {r.padded}")

        if dist.get_rank(self.parallel_context.pp_pg) == self.output_pp_rank:
            assert (
                len(res) == total_length
            ), f"we didn't cover all the data: len(res) == total_length ({len(res)} == {total_length})"

        if len(res) == 0:
            # We are in a process which return no output (beginning/middle of the PP group)
            return []

        return dataset.get_original_order(res)

    @torch.inference_mode()
    def _loglikelihood_tokens(
        self,
        requests,
        disable_tqdm: bool = False,
        override_bs: int = -1,
        num_dataset_splits: int = 1,
        return_bool_score: bool = True,
    ) -> List[LoglikelihoodResponse]:
        dataset = LoglikelihoodDataset(requests=requests, num_dataset_splits=num_dataset_splits)
        res = []

        # Dataset is sorted in descending size.
        # every 20-25% of the dataset we try to double the batch size for speed up
        starting_batch_size = 512

        total_length, subset_length = self._get_subsets(dataset, num_dataset_splits)

        for s, subset_start in enumerate(
            tqdm(
                range(0, total_length, subset_length),
                disable=disable_tqdm,
                position=0,
                desc=f"loglikelihood -- Node {dist.get_rank(self.parallel_context.world_pg)}",
            )
        ):
            dataset.split_start = subset_start
            dataset.split_end = min(subset_start + subset_length, total_length)

            # automatic (variable) batch size detection for vectorization
            # pull longest context sample from request
            context_enc = dataset[0].tokenized_context
            continuation_enc = dataset[0].tokenized_continuation

            max_context = len((context_enc + continuation_enc)[-(self.max_length + 1) :][:-1])

            batch_size = self._get_batch_size(
                override_bs=override_bs, max_input_length=max_context, starting_batch_size=starting_batch_size
            )
            starting_batch_size = batch_size * 2  # for the next round

            # For the DP replicas
            distributed_sampler = DistributedSampler(
                dataset,
                num_replicas=self.parallel_context.dp_pg.size(),
                rank=dist.get_rank(self.parallel_context.dp_pg),
                shuffle=False,
                drop_last=False,
            )
            to_remove_at_the_end = distributed_sampler.total_size - len(dataset)

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=distributed_sampler,
                drop_last=False,
                collate_fn=lambda batch: batch,
            )

            tq = tqdm(
                dataloader,
                disable=disable_tqdm,
                desc=f"loglikelihood in subset {s} Node {dist.get_rank(self.parallel_context.world_pg)}",
            )

            for j, batch_data in enumerate(tq):
                if j < 3:
                    log_rank(
                        f"Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MB. Peak reserved memory: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MB",
                        logger=logger,
                        level=logging.INFO,
                        group=self.parallel_context.world_pg,
                        rank=0,
                    )
                iteration_start_time = time.time()
                inputs = [
                    item.tokenized_context + item.tokenized_continuation[:-1] for item in batch_data
                ]  # The last token doesn't need to be input in the model
                batch_model = self.prepare_batch(
                    inputs, padding_length=max_context, max_context=max_context, full_attention_masks=True
                )
                # batched_inputs, batch_attention, input_lengths, truncated, padded
                with torch.no_grad():
                    out = self.model(input_ids=batch_model.input_ids, input_mask=batch_model.input_mask)

                if dist.get_rank(self.parallel_context.pp_pg) == self.output_pp_rank:
                    # This process got outputs

                    # Gather all the output accross TP
                    gathered_out = [torch.zeros_like(out) for _ in range(self.parallel_context.tp_pg.size())]
                    dist.all_gather(gathered_out, out, group=self.parallel_context.tp_pg, async_op=False)
                    out = torch.cat(gathered_out, dim=-1)

                    out = out.transpose(0, 1)  # [batch, seq_length, vocab]
                    multi_logits = F.log_softmax(out, dim=-1)  # [batch, padding_length, vocab]

                    logits_sum = []
                    max_equals = []
                    batch_cont_tokens = []
                    for cur_request, cur_logits, inplen in zip(batch_data, multi_logits, batch_model.input_lengths):
                        cont_toks = torch.tensor(
                            cur_request.tokenized_continuation, dtype=torch.long, device=self.device
                        )
                        contlen = cont_toks.shape[0]
                        # We only look at the continuation tokens
                        if contlen > inplen:
                            # Continuation is longer than the input size, we are in rolling mode (only continuation)
                            cur_logits = cur_logits.unsqueeze(0).to(self.device)  # [1, seq, vocab]
                            cont_toks = cont_toks[:inplen].unsqueeze(0).to(self.device)  # [1, seq]
                        else:
                            # if contlen == 1:
                            #     top_k = torch.topk(logits, 20)[1].tolist()
                            #     if any(bool(el not in top_k) for el in cont_toks) and not printed_error:
                            #         top_toks_str = '|'.join(self.tokenizer.decode(tt).replace('\n', '') for tt in top_k)
                            #         cont_toks_str = '|'.join(self.tokenizer.decode(tt).replace('\n', '') for tt in cont_toks.tolist())
                            #         logger.error(
                            #             f"Not all the solutions are in the top 20 most likely tokens on rank {dist.get_rank(self.parallel_context.world_pg)} "
                            #             f"top_tokens: {top_toks_str}\ncont_tokens: {cont_toks_str}")

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
                    # Need reshaping before gather
                    batched_inputs, _ = self.pad_and_gather(batch_model.input_ids)
                    lengths = torch.tensor(batch_model.input_lengths, device=self.device)
                    batched_lengths = self.gather(lengths)
                    cont_lengths = torch.tensor([len(c[0]) for c in batch_cont_tokens], device=self.device)
                    batch_cont_length = self.gather(cont_lengths)
                    max_cont_tokens_length = max(len(c[0]) for c in batch_cont_tokens)
                    batch_cont_tokens = torch.cat(
                        [
                            F.pad(c, (0, max_cont_tokens_length - c.shape[1], 0, 0), value=self.tokenizer.pad_token_id)
                            for c in batch_cont_tokens
                        ],
                        dim=0,
                    )
                    batch_cont_tokens, _ = self.pad_and_gather(batch_cont_tokens)
                    # Can be gathered as such
                    logits = torch.tensor(logits_sum, device=self.device)
                    logits = self.gather(logits)
                    max_equal = torch.tensor(max_equals, device=self.device)
                    max_equal = self.gather(max_equal)
                    batch_truncated = torch.tensor(batch_model.truncated, device=self.device)
                    batch_truncated = self.gather(batch_truncated)
                    batch_padded = torch.tensor(batch_model.padded, device=self.device)
                    batch_padded = self.gather(batch_padded)

                    batch_res = []
                    for ix, (
                        logit,
                        cont_tokens,
                        cont_length,
                        maxe,
                        batched_input,
                        batched_length,
                        trunc,
                        padded,
                    ) in enumerate(
                        zip(
                            logits,
                            batch_cont_tokens,
                            batch_cont_length,
                            max_equal,
                            batched_inputs,
                            batched_lengths,
                            batch_truncated,
                            batch_padded,
                        )
                    ):
                        answer = LoglikelihoodResponse(
                            result=(float(logit), bool(maxe)) if return_bool_score else float(logit.sum()),
                            input_tokens=batched_input[: batched_length.item()].numpy(force=True),
                            generated_tokens=cont_tokens[: cont_length.item()].numpy(force=True),
                            truncated_tokens_count=trunc.cpu().item(),
                            padded_tokens_count=padded.cpu().item(),
                        )
                        batch_res.append(answer)

                    # Sort batches back when we add then in res - because of the DistributedSampler, they are interleaved in the results:
                    # Ex with DP=3 and a batch of size 3 we end up with A D G B E H C F I instead of A B C D E F G H I
                    assert len(batch_res) % self.parallel_context.dp_pg.size() == 0
                    for i in range(len(batch_res) // self.parallel_context.dp_pg.size()):
                        for j in range(self.parallel_context.dp_pg.size()):
                            res.append(batch_res[i + j * (len(batch_res) // self.parallel_context.dp_pg.size())])

                    # A bit of logging
                    elapsed_time_per_iteration_ms = (time.time() - iteration_start_time) * 1000
                    tokens_per_sec = batched_inputs.numel() / (elapsed_time_per_iteration_ms / 1000)
                    tq.desc = f"loglikelihood Subset {s} Node {dist.get_rank(self.parallel_context.world_pg)} - {human_format(tokens_per_sec)} tokens/s"

                    # Clean up GPUs
                    del out
                    del logits
                    del batched_inputs
                    del batch_truncated
                    del batch_padded

            # At the end of the subset, remove the additional samples we may have added to make the dataset divisible by the number of processes
            assert to_remove_at_the_end >= 0
            res = res[: len(res) - to_remove_at_the_end]

        # if dist.get_rank(self.parallel_context.tp_pg) == 0:
        #     for i, r in enumerate(res):
        #         print(f"i {i} results: {r.result[-30:]}")
        #         print(f"i {i} input_tokens: {r.input_tokens[-(r.padded + 10):-r.padded]}")
        #         print(f"i {i} cont_tokens: {r.cont_tokens[-30:]}")
        #         print(f"i {i} truncated: {r.truncated}")
        #         print(f"i {i} padded: {r.padded}")

        if dist.get_rank(self.parallel_context.pp_pg) == self.output_pp_rank:
            assert len(res) == total_length, "we didn't cover all the data"

        if len(res) == 0:
            # We are in a process which return no output (beginning/middle of the PP group)
            return []

        return dataset.get_original_order(res)

    @torch.inference_mode()
    def greedy_until(
        self,
        requests: List[GreedyUntilRequest],
        disable_tqdm: bool = False,
        override_bs: int = -1,
        num_dataset_splits: int = 1,
    ) -> List[GenerativeResponse]:
        """Greedy generation until a stop token is generated."""
        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        for request in requests:
            request.stop_sequence = as_list(request.stop_sequence) + [self.tokenizer.eos_token]
            request.tokenized_context = self.tok_encode(request.context)

        dataset = GenerativeTaskDatasetNanotron(requests=requests, num_dataset_splits=num_dataset_splits)
        res = []

        # Dataset is sorted in descending size.
        # every 20-25% of the dataset we try to double the batch size for speed up
        starting_batch_size = 512

        total_length, subset_length = self._get_subsets(dataset, num_dataset_splits)

        for s, subset_start in enumerate(
            tqdm(
                range(0, total_length, subset_length),
                disable=disable_tqdm,
                position=0,
                desc=f"greedy -- Node {dist.get_rank(self.parallel_context.world_pg)}",
            )
        ):
            dataset.split_start = subset_start
            dataset.split_end = min(subset_start + subset_length, total_length)

            if dataset[0][1].generation_size is None:
                # No constraints on the generation size: max length allowed is the max model context
                max_input_length = self.max_length
            else:
                # Longest context in the current split is the first item (since we sort reversed)
                context_enc = dataset[0][1].tokenized_context
                max_gen = max(item[1].generation_size for item in dataset)
                max_input_length = min(len(context_enc) + max_gen, self.max_length)

            batch_size = self._get_batch_size(
                override_bs=override_bs,
                max_input_length=max_input_length,
                starting_batch_size=starting_batch_size,
            )
            # For next iteration, since the batch will be smaller, we'll test a bigger batch size
            starting_batch_size = batch_size * 2

            # For the DP replicas
            distributed_sampler = GenDistributedSampler(
                dataset,
                num_replicas=self.parallel_context.dp_pg.size(),
                rank=dist.get_rank(self.parallel_context.dp_pg),
                shuffle=False,
                drop_last=False,
            )
            to_remove_at_the_end = distributed_sampler.total_size - len(dataset)

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=distributed_sampler,
                drop_last=False,
                collate_fn=lambda batch: batch,
            )

            tq = tqdm(dataloader, desc=f"greedy in subset {s} Node {dist.get_rank(self.parallel_context.world_pg)}")
            for j, indexed_batch in enumerate(tq):
                if j < 3:
                    log_rank(
                        f"Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MB. Peak reserved memory: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MB",
                        logger=logger,
                        level=logging.INFO,
                        group=self.parallel_context.world_pg,
                        rank=0,
                    )
                iteration_start_time = time.time()
                sample_index, batch = zip(*indexed_batch)

                # NOTE: we are assuming all items in a batch behave similarly (same
                # stop_tokens and max_tokens genrated) which is not necessarily
                # the case! Because of that we should only use batch size of 1

                # Since items are sorted by inverse length, the first one always has
                # the maximum allowed generation size for the batch, unless we want to force truncation
                # need to pass them somewhere ! stop_tokens = batch[0].stop_sequence
                max_new_tokens = batch[0].generation_size
                returns_logits = batch[0].use_logits
                num_samples = batch[0].num_samples
                if num_samples > 1:
                    hlog_err(
                        "Nonotron models does not allow sampling evaluations - this is likely to fail or provide problematic results"
                    )

                context = [c.context for c in batch]

                # See doc https://huggingface.co/docs/transformers/v4.38.2/en/pad_truncation#padding-and-truncation
                # Will do left truncation and padding, as defined when creating the tokenizer
                tokenized = self.tokenizer(
                    context,
                    truncation="longest_first",  # we truncate to the model max length if needed
                    padding="longest",  # we pad to the longest sequence
                    return_tensors="pt",
                    max_length=self.max_length - 1,  # we always allow minimum one token of generation
                    add_special_tokens=self.add_special_tokens,
                ).to(self.device)

                # The main question for this step is the following:
                # Would we rather truncate the prompt to allow generation to go to max_new_tokens, at the risk
                # of losing some meaning, or have some generations that are exceedingly short?
                # The choice we go for here is to avoid truncating the prompt if we can, since it
                # should have been managed by the prompt creator/few shot manager if requested by the user.
                context_size = tokenized["input_ids"].shape[1]
                if context_size > self.max_length:
                    hlog_warn(
                        f"The context size of your batch ({context_size}) is bigger than the maximum context size allowed by the model ({self.max_length}) for a task in"
                        + str({i.task_name for i in batch})
                        + ". This is likely to lead to some errors."  # noqa C401
                    )
                    # There will be truncation of at least one sample, maximum generation size will be one
                    max_new_tokens = 1
                else:  # We can't allow generation of more than max_length
                    max_new_tokens = min(self.max_length - context_size, max_new_tokens)

                batch_model = Batch(
                    input_ids=tokenized["input_ids"],
                    input_lengths=[len(item == 1) for item in tokenized["attention_mask"]],
                    input_mask=tokenized["attention_mask"],
                    truncated=[
                        len(c) - tokenized["input_ids"].shape[1] if len(c) > tokenized["input_ids"].shape[1] else 0
                        for c in context
                    ],
                    padded=[sum(mask == 0) for mask in tokenized["attention_mask"]],
                )

                # responses, logits and input_ids have all been gathered accross GPUs already
                # but we also grab the original length of these vectors, which have been padded
                # while being gathered - the added info
                outputs = decode_tokenized(
                    input_ids=batch_model.input_ids,
                    input_mask=batch_model.input_mask,
                    model=self.model,
                    parallel_context=self.parallel_context,
                    max_new_tokens=max_new_tokens,
                    max_micro_batch_size=batch_size,  # ok for PP=1 for PP>1 we'll need to split the batch
                    returns_logits=returns_logits,
                    generation_config=self.generation_config,
                )
                dist.barrier()  # Got everyone to send their stuff
                outputs = list(outputs)

                generations = torch.stack([o.generation_ids[o.input_ids.shape[0] :] for o in outputs])
                batch_input_ids, len_ids = self.pad_and_gather(batch_model.input_ids)
                batch_generations, _ = self.pad_and_gather(generations)

                if returns_logits:
                    logits = torch.stack([o.logits for o in outputs])
                    logits, len_logits = self.pad_and_gather(logits)

                if dist.get_rank(self.parallel_context.pp_pg) == self.output_pp_rank:
                    generations = batch_generations.numpy(force=True)
                    input_ids = batch_input_ids.numpy(force=True)

                    batch_sample_index = torch.tensor(sample_index, device=self.device)
                    batch_sample_index = self.gather(batch_sample_index)
                    batch_truncated = torch.tensor(batch_model.truncated, device=self.device)
                    batch_truncated = self.gather(batch_truncated)
                    batch_padded = torch.tensor(batch_model.padded, device=self.device)
                    batch_padded = self.gather(batch_padded)

                    batch_res = []
                    for ix, (generation, batched_input, trunc, padded, sample_index) in enumerate(
                        zip(generations, input_ids, batch_truncated, batch_padded, batch_sample_index)
                    ):
                        # Ensure the generated responses do not contain the stop sequences.
                        decoded_response = self.tokenizer.decode(generation, skip_special_tokens=False)
                        stop_terms = dataset[sample_index][1].stop_sequence
                        for stop_term in stop_terms:
                            decoded_response = decoded_response.split(stop_term)[0]
                        # partial caching
                        cur_response = GenerativeResponse(
                            result=decoded_response,
                            logits=logits[ix][: len_logits[ix]] if returns_logits else None,
                            generated_tokens=generation,
                            input_tokens=batched_input[: len_ids[ix]],
                            truncated_tokens_count=trunc.cpu().item(),
                            padded_tokens_count=padded.cpu().item(),
                        )
                        # self.cache_hook.add_partial("greedy_until", (context, stop_tokens), cur_response)
                        batch_res.append(cur_response)

                    # Sort batches back when we add then in res - because of the DistributedSampler, they are interleaved in the results:
                    # Ex with DP=3 and a batch of size 3 we end up with A D G B E H C F I instead of A B C D E F G H I
                    assert len(batch_res) % self.parallel_context.dp_pg.size() == 0
                    for i in range(len(batch_res) // self.parallel_context.dp_pg.size()):
                        for j in range(self.parallel_context.dp_pg.size()):
                            res.append(batch_res[i + j * (len(batch_res) // self.parallel_context.dp_pg.size())])

                    # A bit of logging
                    elapsed_time_per_iteration_ms = (time.time() - iteration_start_time) * 1000
                    tokens_per_sec = (batch_input_ids.numel() + batch_generations.numel()) / (
                        elapsed_time_per_iteration_ms / 1000
                    )

                    tq.desc = f"greedy_until Subset {s} Node {dist.get_rank(self.parallel_context.world_pg)} - {human_format(tokens_per_sec)} tokens/s"

            # At the end of the subset, remove the additional samples we may have added to make the dataset divisible by the number of processes
            assert to_remove_at_the_end >= 0
            res = res[: len(res) - to_remove_at_the_end]

        if dist.get_rank(self.parallel_context.pp_pg) == self.output_pp_rank:
            assert (
                len(res) == total_length
            ), f"we didn't cover all the data: len(res) == total_length ({len(res)} == {total_length})"

        if len(res) == 0:
            # We are in a process which return no output (beginning/middle of the PP group)
            return []

        return dataset.get_original_order(res)


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
