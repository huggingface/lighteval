#!/usr/bin/env python

# MIT License

# Copyright (c) 2024 Taratra D. RAHARISON and The HuggingFace Team

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

import argparse
import importlib
import json
import os

import pkg_resources


def load_tasks_table_extended(module_name: any) -> list:
    """
    load the module module_name

    Args:
    - module_name the name of the module we want to load
    Returns:
    - TASKS_TABLE: a list of the task in the module
    """
    module_path = f"lighteval.tasks.extended.{module_name}.main"
    module_loaded = importlib.import_module(module_path)
    tasks_list = None
    try:
        tasks_list = module_loaded.TASKS_TABLE
    except Exception as e:
        print(e)
    return tasks_list if tasks_list is not None else []


def get_tasks_table_json() -> list:
    """
    Fetch tasks/tasks_table.jsonl
    Returns
    - a list of all the tasks in tasks/tasks_table.jsonl
    """
    tasks = []
    # Handling tasks_table.jsonl
    # Get the path to the resource file
    tasks_table_path = pkg_resources.resource_filename("lighteval", "tasks/tasks_table.jsonl")
    with open(tasks_table_path) as jsonl_tasks_table:
        jsonl_tasks_table_content = jsonl_tasks_table.read()
        for jline in jsonl_tasks_table_content.splitlines():
            tasks.append(json.loads(jline))
    return tasks


def get_extended_tasks() -> list:
    """
    Fetch all the tasks in the extended suite
    Returns
    - a list of all the extended tasks
    """
    tasks_extended = []
    extended_tasks_dir = pkg_resources.resource_filename("lighteval", "tasks/extended")
    for root, dirs, files in os.walk(extended_tasks_dir):
        for file in files:
            if file == "main.py":
                module_name = os.path.basename(root)
                tasks_table = load_tasks_table_extended(module_name)
                tasks_extended += tasks_table
    return tasks_extended


def group_by_suite(tasks: list, tasks_extended: list) -> dict:
    """
    Group tasks by suite and sort them alphabetically
    Args:
    - tasks: list of tasks in tasks/tasks_table.jsonl
    - tasks_extended: list of extended tasks
    Returns:
    - a dict of tasks grouped by suite
    """
    grouped_by_suite = {}
    for task in tasks:
        for suite in task["suite"]:
            if suite not in grouped_by_suite.keys():
                grouped_by_suite[suite] = [task["name"]]
            else:
                grouped_by_suite[suite].append(task["name"])
                grouped_by_suite[suite].sort()

    grouped_by_suite["extended"] = []
    # Adding extended suite
    for task in tasks_extended:
        grouped_by_suite["extended"].append(task["name"])
    grouped_by_suite["extended"].sort()
    return grouped_by_suite


def list_tasks_command():
    """
    List all the available tasks in tasks_table.jsonl and the extended directory
    Assumes the existence of TASKS_TABLE in the main.py file for each extended
    tasks in tasks/extended
    """
    try:
        # Handling tasks_table.jsonl
        tasks = get_tasks_table_json()

        # Handling extended tasks
        tasks_extended = get_extended_tasks()

        # Grouping by suite the tasks
        grouped_by_suite = group_by_suite(tasks, tasks_extended)

        # Print tasks
        print("Available tasks: (Grouped by suite)\n")
        for suite, task_list in grouped_by_suite.items():
            print("- " + suite)
            for task in task_list:
                print("\t - " + task)
    except Exception as e:
        print("Error: ", e)


def main():
    parser = argparse.ArgumentParser(description="CLI tool for lighteval, a lightweight framework for LLM evaluation")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    args = parser.parse_args()

    if args.list_tasks:
        list_tasks_command()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
