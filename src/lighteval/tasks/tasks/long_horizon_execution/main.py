"""
name:
Long Horizon Execution

dataset:
arvindh75/Long-Horizon-Execution

abstract:
Evaluation benchmark for long-context execution capabilities of language models.
Tests a model's ability to maintain state and perform cumulative operations over
long sequences of inputs. Supports both single-turn (all inputs at once) and
multi-turn (inputs provided incrementally) evaluation modes.

The task requires models to:
1. Maintain a dictionary mapping keys to values
2. Process a sequence of keys
3. Calculate cumulative sums after each key or group of keys
4. Handle varying context sizes and turn complexities

Single-turn evaluation (Section 3.3): Model outputs only the final cumulative sum
after processing all keys, allowing any aggregation strategy.

Multi-turn evaluation: Model processes keys in batches of K per turn, maintaining
conversation history and outputting cumulative sums incrementally. Evaluates
fractional accuracy (correct turns / total turns).

languages:
english

tags:
long-context, state-tracking, arithmetic, execution

paper:
https://arxiv.org/abs/2509.09677

starred:
true
"""

from lighteval.tasks.tasks.long_horizon_execution.multi_turn import create_multi_turn_tasks
from lighteval.tasks.tasks.long_horizon_execution.single_turn import create_single_turn_tasks


single_turn_tasks = create_single_turn_tasks()
multi_turn_tasks = create_multi_turn_tasks()

TASKS_TABLE = single_turn_tasks + multi_turn_tasks
