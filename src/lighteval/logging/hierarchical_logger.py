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

import sys
import time
from datetime import timedelta
from logging import Logger
from typing import Any, Callable

from lighteval.utils.imports import is_accelerate_available, is_nanotron_available


if is_nanotron_available():
    from nanotron.logging import get_logger

    logger = get_logger(__name__, log_level="INFO")
elif is_accelerate_available():
    from accelerate.logging import get_logger

    logger = get_logger(__name__, log_level="INFO")
else:
    logger = Logger(__name__, level="INFO")

from colorama import Fore, Style


class HierarchicalLogger:
    """
    Tracks the execution flow of the code as blocks, along with how long we're spending in each block.
    Should not be called on its own, use [`hlog`], [`hlog_warn`] and [`hlog_err`] to log things, and
    [`htrack_block`] to start a block level log step, or [`htrack`] to start a function level log step.
    """

    def __init__(self) -> None:
        self.start_times: list[float] = []

    def indent(self) -> str:
        """Manages the block level text indentation for nested blocks"""
        return "  " * len(self.start_times)

    def track_begin(self, x: Any) -> None:
        """Starts a block level tracker, stores the step begin time"""
        logger.warning(f"{self.indent()}{str(x)} \u007b")  # \u007b is {
        sys.stdout.flush()
        self.start_times.append(time.time())

    def track_end(self) -> None:
        """Ends a block level tracker, prints the elapsed time for the associated step"""
        duration = time.time() - self.start_times.pop()
        logger.warning(f"{self.indent()}\u007d [{str(timedelta(seconds=duration))}]")  # \u007d is }
        sys.stdout.flush()

    def log(self, x: Any) -> None:
        logger.warning(self.indent() + str(x))
        sys.stdout.flush()


HIERARCHICAL_LOGGER = HierarchicalLogger()
BACKUP_LOGGER = Logger(__name__, level="INFO")


# Exposed public methods
def hlog(x: Any) -> None:
    """Info logger.

    Logs a string version of x through the singleton [`HierarchicalLogger`].
    """
    try:
        HIERARCHICAL_LOGGER.log(x)
    except RuntimeError:
        BACKUP_LOGGER.warning(x)


def hlog_warn(x: Any) -> None:
    """Warning logger.

    Logs a string version of x, which will appear in a yellow color, through the singleton [`HierarchicalLogger`].
    """
    try:
        HIERARCHICAL_LOGGER.log(Fore.YELLOW + str(x) + Style.RESET_ALL)
    except RuntimeError:
        BACKUP_LOGGER.warning(Fore.YELLOW + str(x) + Style.RESET_ALL)


def hlog_err(x: Any) -> None:
    """Error logger.

    Logs a string version of x, which will appear in a red color, through the singleton [`HierarchicalLogger`].
    """
    try:
        HIERARCHICAL_LOGGER.log(Fore.RED + str(x) + Style.RESET_ALL)
    except RuntimeError:
        BACKUP_LOGGER.warning(Fore.RED + str(x) + Style.RESET_ALL)


class htrack_block:
    """
    Block annotator: hierarchical logging block, which encapsulate the current step's logs and duration.

    Usage:
        with htrack_block('Step'):
            hlog('current logs')

    Output:
        Step {
            current logs
        } [0s]
    """

    def __init__(self, x: Any) -> None:
        self.x = x

    def __enter__(self) -> None:
        HIERARCHICAL_LOGGER.track_begin(self.x)

    def __exit__(self, tpe: Any, value: Any, callback: Any) -> None:
        HIERARCHICAL_LOGGER.track_end()


class htrack:
    """
    Function annotator: prints called function parameters, then opens an hierarchical [`htrack_block`]
    which encapsulate the current step's logs and duration.

    Usage:
        @htrack()
        def function(args):
            with htrack_block('Step'):
                hlog('current logs')

    Output:
        function: args, {
            Step {
                current logs
            } [0s]
        }
    """

    def __call__(self, fn: Callable) -> Any:
        def wrapper(*args, **kwargs):  # type:ignore
            # Parent name to prepend
            if len(args) > 0 and hasattr(args[0], fn.__name__):
                parent = type(args[0]).__name__ + "."
            else:
                parent = ""

            args_list = ""
            if len(args) > 0 or len(kwargs) > 0:
                args_list = ": "
                for v in enumerate(args):
                    args_list += f"{str(v)}, "
                for k, v in kwargs.items():
                    args_list += f"{str(k)}: {str(v)}, "

            with htrack_block(parent + fn.__name__ + args_list):
                return fn(*args, **kwargs)

        return wrapper
