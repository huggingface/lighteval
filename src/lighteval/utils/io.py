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
from dataclasses import dataclass

from fsspec import AbstractFileSystem, url_to_fs
from huggingface_hub import HfFileSystem


@dataclass(frozen=True)
class FsspecDataResource:
    fs: AbstractFileSystem
    path: str

    @classmethod
    def from_uri(cls, uri: str) -> "FsspecDataResource":
        fs, path = url_to_fs(uri)
        return cls(fs=fs, path=path)

    def __truediv__(self, other: str) -> "FsspecDataResource":
        return FsspecDataResource(fs=self.fs, path=os.path.join(self.path, other))

    def __str__(self) -> str:
        return self.path


def get_hf_repo_id(resource: FsspecDataResource) -> str:
    if isinstance(resource.fs, HfFileSystem):
        return "/".join(resource.path.split("/")[:2])
    raise ValueError("Resource is not a Hugging Face Hub repository")
