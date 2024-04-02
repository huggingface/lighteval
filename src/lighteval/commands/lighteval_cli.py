#!/usr/bin/env python

# MIT License

# Copyright (c) 2024 Taratra D. RAHARISON

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
    #module_spec = importlib.util.find_spec(module_path)
    print(module_path)
    module_spec = importlib.import_module("TASKS_TABLE", module_path)
    print(module_spec)
    #if module_spec:
        #module = importlib.util.module_from_spec(module_spec)
        #module_spec.loader.exec_module(module)
        #print(module)
        #if hasattr(module, "TASKS_TABLE"):
            #return module.TASKS_TABLE
    return []

def list_tasks_command():
    """
    List all the avalaible tasks in tasks_table.jsonl and the extended directory
    Assumes the existence of TASKS_TABLE in the main.py file for each extended 
    tasks in tasks/extended
    """
    try:
        tasks = []
        # Handling tasks_table.jsonl
        # Get the path to the resource file
        tasks_table_path = pkg_resources.resource_filename('lighteval', 'tasks/tasks_table.jsonl')
        with open(tasks_table_path) as jsonl_tasks_table:
            jsonl_tasks_table_content = jsonl_tasks_table.read()
            for jline in jsonl_tasks_table_content.splitlines():
                tasks.append(json.loads(jline))
        
        # Handling extended tasks
        tasks_extended = []
        extended_tasks_dir = pkg_resources.resource_filename("lighteval", "tasks/extended")
        print("tasks_dir_extended ",extended_tasks_dir)
        for module_name in pkg_resources.resource_listdir("lighteval", "tasks/extended"):
            tasks_table = load_tasks_table_extended(module_name)
            tasks_extended += tasks_table
        tasks += tasks_extended
        if len(tasks) > 0:
            print("Available tasks: ")
            for task in tasks:
                print("- " + task["name"])

    #except FileNotFoundError:
    except Exception as e:
        print('Error: ', e)


def main():
    parser = argparse.ArgumentParser(description='CLI tool for lighteval, a lightweight framework for LLM evaluation')
    parser.add_argument('--list-tasks', action='store_true', help='List available tasks')
    args = parser.parse_args()

    if args.list_tasks:
        list_tasks_command()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
