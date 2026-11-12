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

import os


def timeout(timeout_seconds: int = 10):  # noqa: C901
    """A decorator that applies a timeout to the decorated function.

    Args:
        timeout_seconds (int): Number of seconds before timing out the decorated function.
            Defaults to 10 seconds.

    Notes:
        On Unix systems, uses a signal-based alarm approach which is more efficient as it doesn't require spawning a new process.
        On Windows systems, signal.alarm is not available, so the function runs in a worker thread instead.
        If it times out, the caller gets a TimeoutError, but the thread cannot be stopped and finishes in the background.

    Returns:
        Callable: A decorator function that wraps the original function with timeout functionality
    """
    if os.name == "posix":
        # Unix-like approach: signal.alarm
        import signal

        def decorator(func):
            def handler(signum, frame):
                raise TimeoutError("Operation timed out!")

            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    # Cancel the alarm and restore previous handler
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            return wrapper

        return decorator

    else:
        # Windows approach: run the function in a worker thread.
        # A child process cannot be used here: Windows starts processes with "spawn", which has to pickle
        # the target, and the decorated functions are often closures or local functions that cannot be
        # pickled. That made every call fail, and the callers turned the error into a wrong answer.
        import threading

        def decorator(func):
            def wrapper(*args, **kwargs):
                outcome = {}

                def run_func():
                    try:
                        outcome["result"] = func(*args, **kwargs)
                    except BaseException as e:
                        outcome["error"] = e

                worker = threading.Thread(target=run_func, daemon=True)
                worker.start()
                worker.join(timeout_seconds)

                if worker.is_alive():
                    # A thread cannot be killed, so it is left to finish in the background. It is a daemon
                    # thread, so it will not keep the interpreter alive at exit.
                    raise TimeoutError("Operation timed out!")

                if "error" in outcome:
                    raise outcome["error"]
                return outcome["result"]

            return wrapper

        return decorator
