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

import pytest
from huggingface_hub import HfApi
from huggingface_hub.hf_api import DatasetInfo


TESTING_EMPTY_HF_ORG_ID = "lighteval-tests"


@pytest.fixture
def testing_empty_hf_org_id(org_id: str = TESTING_EMPTY_HF_ORG_ID):
    old_token = os.getenv("HF_TOKEN")
    os.environ["HF_TOKEN"] = os.getenv("HF_TEST_TOKEN") or ""

    def list_repos(org_id: str):
        return list(hf_api.list_models(author=org_id)) + list(hf_api.list_datasets(author=org_id))

    def clean_repos(org_id: str):
        repos = list_repos(org_id)
        for repo in repos:
            hf_api.delete_repo(repo.id, repo_type="dataset" if isinstance(repo, DatasetInfo) else "model")

    hf_api = HfApi()
    # Remove all repositories in the HF org
    clean_repos(org_id)

    # Verify that all repositories have been removed
    remaining_repos = list_repos(org_id)
    assert len(remaining_repos) == 0, f"Expected 0 repositories, but found {len(remaining_repos)}"

    yield org_id

    # Clean up: recreate any necessary default repositories after the test
    # This step is optional and depends on your specific needs
    clean_repos(org_id)
    os.environ["HF_TOKEN"] = old_token if old_token is not None else ""
