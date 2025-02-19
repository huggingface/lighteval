# MIT License

# Copyright (c) 2025 The HuggingFace Team

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
import re
import time
from typing import Any

import requests
from scipy import optimize


URL_CODEFORCES_STANDINGS = "https://codeforces.com/api/contest.standings?contestId={contest_id}&showUnofficial=false"
URL_RATING_CHANGES = "https://codeforces.com/api/contest.ratingChanges?contestId={contest_id}"


def make_html_problem(line: dict[str, Any]) -> str:
    title = line["title"]
    html = f"<html><body><h1>{title}</h1>"
    input_, output = line["input"], line["output"]
    interaction = line["interaction"]
    note = line["note"]
    examples = line["examples"]
    # Use interaction if provided, otherwise use input and output
    html += f"<h2>Description</h2><div>{line['description']}</div>"
    if interaction:
        html += f"<h2>Interaction</h2><div>{interaction}</div>"
    else:
        html += f"<h2>Input</h2><div>{input_}</div>"
        html += f"<h2>Output</h2><div>{output}</div>"

    # The example is always present
    example_text = ""
    for example in examples:
        example_text += f"<div>Input:\n{example[0]}</div><div>Output:\n{example[1]}</div>"
    html += f"<h2>{'Example' if len(examples) == 1 else 'Examples'}</h2>{example_text}"

    if note:
        html += f"<h2>Note</h2><div>{note}</div>"

    html += "</body></html>"
    return html


# Let the whole mapping defined, but the output should be cpp
LANG_MAP = {
    "kotlin": 88,  # Kotlin 1.9.21
    "cpp": 91,  # GNU G++23 14.2 (64 bit, msys2)
    "ruby": 67,  # Ruby 3.2.2
    "d": 28,  # D DMD32 v2.105.0
    "python": 70,  # PyPy 3.10 (7.3.15, 64bit)
    "pascal": 51,  # PascalABC.NET 3.8.3
    "rust": 75,  # Rust 1.75.0 (2021)
    "go": 32,  # Go 1.22.2
    "node.js": 55,  # Node.js 15.8.0 (64bit)
    "haskell": 12,  # Haskell GHC 8.10.1
    "javascript": 34,  # JavaScript V8 4.8.0
    "csharp": 79,  # C# 10, .NET SDK 6.0
    "perl": 13,  # Perl 5.20.1
    "java": 87,  # Java 21 64bit
    "ocaml": 19,  # OCaml 4.02.1
    "delphi": 3,  # Delphi 7
    "php": 6,  # PHP 8.1.7
    "scala": 20,  # Scala 2.12.8
    "c": 43,  # GNU GCC C11 5.1.0
}


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Extracts code blocks from a text, returning a list of tuples with the language and code (if found)."""
    pattern = r"```(\w*)\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def submit_code(prob: str, lang_id: int, code: str, tag: str = "", retry: int = 3, delay: int = 10) -> int | str:
    """Submits code for a specific problem to the API endpoint.

    Args:
        prob (str): The problem identifier to submit code for.
        lang (int): The programming language id of the submitted code.
        code (str): The actual code to be submitted.
        tag (str, optional): Additional tag for the submission. Defaults to empty string.
        retry (int, optional): Number of retry attempts if the request fails.
            Defaults to RETRY constant.

    Returns:
        dict/str: If successful, returns a JSON response containing submission details.
            If all retries fail, returns an error message string.

    Example:
        >>> result = submit_code("2000A", 70, "print('Hello')")
        >>> result = submit_code("2000A", 91, "int main() {}", "test", retry=3)
    """
    token = os.getenv("CODEELO_TOKEN")  # Replace with your own token
    base_url = os.getenv("CODEELO_BASE_URL")

    if not token or not base_url:
        raise ValueError("Please set the CODEELO_TOKEN and CODEELO_BASE_URL environment variables.")

    try:
        url = f"{base_url}/submit_code"
        headers = {"Content-Type": "application/json", "Authorization": token}
        payload = {"prob": prob, "lang": lang_id, "code": code, "tag": tag}
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        assert response.status_code == 200, "Failed to submit code"
        return response.json()["submission_id"]

    except Exception as e:
        if retry > 0:
            print(f"Failed to submit code, retrying in {delay} seconds")
            time.sleep(delay)
            return submit_code(prob, lang_id, code, tag, retry - 1)
        else:
            return f"Failed to submit code: {str(e)}"


def calc_elo_rating(contest_id: int, problem_status: dict[str, Any]) -> int:  # noqa: C901
    """Compute the ELO rating for the given contest id.

    Args:
        contest_id (int): _description_
        problem_status (dict[str, Any]): _description_

    Returns:
        int: _description_
    """
    standings = requests.get(URL_CODEFORCES_STANDINGS.format(contest_id=contest_id)).json()
    rating_changes = requests.get(URL_RATING_CHANGES.format(contest_id=contest_id)).json()

    try:
        handle_set = {
            standings["result"]["rows"][i]["party"]["members"][0]["handle"]
            for i in range(len(standings["result"]["rows"]))
        } and {rating_changes["result"][i]["handle"] for i in range(len(rating_changes["result"]))}
        standings["result"]["rows"] = [
            standings["result"]["rows"][i]
            for i in range(len(standings["result"]["rows"]))
            if standings["result"]["rows"][i]["party"]["members"][0]["handle"] in handle_set
        ]
        rating_changes["result"] = [
            rating_changes["result"][i]
            for i in range(len(rating_changes["result"]))
            if rating_changes["result"][i]["handle"] in handle_set
        ]

        assert (len(standings["result"]["rows"]) == len(rating_changes["result"])) and len(
            standings["result"]["rows"]
        ) > 200, "No result"

    except Exception as e:
        print(e)

    if (
        ("result" not in standings)
        or ("result" not in rating_changes)
        or (len(standings["result"]["rows"]) != len(rating_changes["result"]))
        or (len(standings["result"]["rows"]) <= 200)
    ):
        print("No result, return 0")
        return 0

    max_rating = 0
    for i in range(len(rating_changes["result"])):
        max_rating = max(max_rating, rating_changes["result"][i]["oldRating"])

    # Obtain score and penalty
    score = 0
    penalty = 0

    for problem in standings["result"]["problems"]:
        prob = f"{problem['contestId']}{problem['index']}"
        if prob in problem_status.keys():
            for ith, status in enumerate(problem_status[prob]):
                if status == "AC":
                    if "points" in problem:
                        score += max(0, problem["points"] - 50 * ith)
                    else:
                        score += 1
                        penalty += ith * 10
                    break

    # Obtain number of participants and target rank
    n = len(standings["result"]["rows"])

    rank = n
    for i in range(n):
        if (standings["result"]["rows"][i]["points"] < score) or (
            (standings["result"]["rows"][i]["points"] == score)
            and (standings["result"]["rows"][i]["penalty"] > penalty)
        ):
            rank = i
            break

    return find_rating(rating_changes, rank, max_rating=max_rating)


def calculate_elo_expectation(candidate_rating: float, player_ratings: list[float]) -> float:
    """Calculate the expected score based on Elo rating formula"""
    return 1 + sum(1 / (1 + 10 ** ((candidate_rating - rating) / 400)) for rating in player_ratings)


def find_rating(rating_changes: dict, target_rank: float, max_rating: int = 4000) -> int:
    """Find the rating using scipy's root finding methods"""
    old_ratings = [change["oldRating"] for change in rating_changes["result"]]

    def rating_difference(x: float) -> float:
        return calculate_elo_expectation(x, old_ratings) - target_rank

    # Use binary search method from scipy
    result = optimize.root_scalar(rating_difference, bracket=[0, max_rating + 100], method="brentq")

    return int(result.root)


def check_status(submission_id, retry: int = 3, delay: int = 10) -> dict[str, Any] | str:
    """Checks the status of a specific submission using the API endpoint.

    Args:
        submission_id (str): The ID of the submission to check.
        retry (int, optional): Number of retry attempts if the request fails.

    Returns:
        dict/str: If successful, returns a JSON response containing submission status.
            If all retries fail, returns an error message.

    Example:
        >>> status = check_status("12345")
        >>> status = check_status("67890", retry=3)
    """
    token = os.getenv("CODEELO_TOKEN")
    base_url = os.getenv("CODEELO_BASE_URL")

    try:
        url = f"{base_url}/check_status"
        headers = {"Content-Type": "application/json", "Authorization": token}
        params = {"submission_id": submission_id}
        response = requests.get(url, headers=headers, params=params, timeout=20)
        assert response.status_code == 200
        return response.json()["status_canonical"]
    except Exception as e:
        if retry > 0:
            print(f"Failed to get problem, retrying in {delay} seconds")
            time.sleep(delay)
            return check_status(submission_id, retry - 1)
        else:
            return f"Failed to get problem: {str(e)}"
