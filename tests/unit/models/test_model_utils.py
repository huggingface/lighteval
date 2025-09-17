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

import unittest
from unittest.mock import Mock

from lighteval.models.utils import uses_chat_template


class TestUseChatTemplate(unittest.TestCase):
    def test_uses_chat_template_with_chat_template_present(self):
        """Test that uses_chat_template returns True when tokenizer has a chat template."""
        mock_tokenizer = Mock()
        mock_tokenizer.chat_template = "{% for message in messages %}..."

        result = uses_chat_template(tokenizer=mock_tokenizer)
        self.assertTrue(result)

    def test_uses_chat_template_with_no_chat_template(self):
        """Test that uses_chat_template returns False when tokenizer has no chat template."""
        mock_tokenizer = Mock()
        mock_tokenizer.chat_template = None

        result = uses_chat_template(tokenizer=mock_tokenizer)
        self.assertFalse(result)

    def test_uses_chat_template_with_chat_template_present_override(self):
        """Test that uses_chat_template returns True when tokenizer has a chat template."""
        mock_tokenizer = Mock()
        mock_tokenizer.chat_template = "{% for message in messages %}..."

        result = uses_chat_template(tokenizer=mock_tokenizer, override_chat_template=False)
        self.assertFalse(result)

    def test_uses_chat_template_with_no_chat_template_override(self):
        """Test that uses_chat_template returns False when tokenizer has no chat template."""
        mock_tokenizer = Mock()
        mock_tokenizer.chat_template = None

        result = uses_chat_template(tokenizer=mock_tokenizer, override_chat_template=True)
        self.assertTrue(result)
