import functools
import gc
import inspect

import torch

from lighteval.logging.hierarchical_logger import hlog, logger
from lighteval.utils import (
    NO_ACCELERATE_ERROR_MSG,
    NO_NANOTRON_ERROR_MSG,
    is_accelerate_available,
    is_nanotron_available,
)


def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory

    Example:

    ```python
    >>> from lighteval.utils_parallelism import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                else:
                    raise

    return decorator


def test_all_gather(accelerator=None, parallel_context=None):
    """
    Test the gather operation in a parallel setup.

    Args:
        accelerator (Optional): The accelerator object used for parallelism.
        parallel_context (Optional): The parallel context object used for parallelism.

    Raises:
        ImportError: If the required accelerator or parallel context is not available.
    """
    if accelerator:
        if not is_accelerate_available():
            raise ImportError(NO_ACCELERATE_ERROR_MSG)
        hlog("Test gather tensor")
        test_tensor: torch.Tensor = torch.tensor([accelerator.process_index], device=accelerator.device)
        gathered_tensor: torch.Tensor = accelerator.gather(test_tensor)
        hlog(f"gathered_tensor {gathered_tensor}, should be {list(range(accelerator.num_processes))}")
        accelerator.wait_for_everyone()
    elif parallel_context:
        if not is_nanotron_available():
            raise ImportError(NO_NANOTRON_ERROR_MSG)
        from nanotron import distributed as dist
        from nanotron import logging

        hlog("Test gather tensor")
        # Do a first NCCL sync to warmup and try to avoid Timeout after model/data loading
        logging.log_rank(
            f"[TEST] Running NCCL sync for ranks {list(range(parallel_context.world_pg.size()))}",
            logger=logger,
            level=logging.WARNING,
            group=parallel_context.dp_pg,
            rank=0,
        )
        test_tensor = torch.tensor([dist.get_rank(parallel_context.world_pg)], device=torch.device("cuda"))
        test_tensor_list = [torch.zeros_like(test_tensor) for _ in range(parallel_context.world_pg.size())]
        dist.all_gather(test_tensor_list, test_tensor, group=parallel_context.world_pg, async_op=False)
        dist.barrier()
        logging.log_rank(
            f"[TEST] NCCL sync for ranks {[t.item() for t in test_tensor_list]}",
            logger=logger,
            level=logging.WARNING,
            group=parallel_context.dp_pg,
            rank=0,
        )

        del test_tensor_list
        del test_tensor
    else:
        hlog("Not running in a parallel setup, nothing to test")
