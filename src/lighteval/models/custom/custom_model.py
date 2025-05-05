# MIT License
#
# Copyright (c) 2024 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from lighteval.models.utils import ModelConfig


class CustomModelConfig(ModelConfig):
    """
    Configuration class for loading custom model implementations in Lighteval.

    This config allows users to define and load their own model implementations by specifying
    a Python file containing a custom model class that inherits from LightevalModel.

    The custom model file should contain exactly one class that inherits from LightevalModel.
    This class will be automatically detected and instantiated when loading the model.

    Args:
        model (str):
            An identifier for the model. This can be used to track which model was evaluated
            in the results and logs.

        model_definition_file_path (str):
            Path to a Python file containing the custom model implementation. This file must
            define exactly one class that inherits from LightevalModel. The class should
            implement all required methods from the LightevalModel interface.

    Example usage:
        ```python
        # Define config
        config = CustomModelConfig(
            model="my-custom-model",
            model_definition_file_path="path/to/my_model.py"
        )

        # Example custom model file (my_model.py):
        from lighteval.models.abstract_model import LightevalModel

        class MyCustomModel(LightevalModel):
            def __init__(self, config, env_config):
                super().__init__(config, env_config)
                # Custom initialization...

            def greedy_until(self, *args, **kwargs):
                # Custom generation logic...
                pass
        ```

    An example of a custom model can be found in `examples/custom_models/google_translate_model.py`.

    Notes:
        - The custom model class must inherit from LightevalModel and implement all required methods
        - Only one class inheriting from LightevalModel should be defined in the file
        - The model file is dynamically loaded at runtime, so ensure all dependencies are available
        - Exercise caution when loading custom model files as they can execute arbitrary code
    """

    model_name: str
    model_definition_file_path: str
