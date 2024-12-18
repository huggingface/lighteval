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
import logging
import logging.config

import colorlog
import typer

import lighteval.main_accelerate
import lighteval.main_baseline
import lighteval.main_endpoint
import lighteval.main_nanotron
import lighteval.main_tasks
import lighteval.main_vllm


app = typer.Typer()

logging_config = dict(  # noqa C408
    version=1,
    formatters={
        "c": {
            "()": colorlog.ColoredFormatter,
            "format": "[%(asctime)s] [%(log_color)s%(levelname)8s%(reset)s]: %(message)s (%(filename)s:%(lineno)s)",
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        },
    },
    handlers={"h": {"class": "logging.StreamHandler", "formatter": "c", "level": logging.INFO}},
    root={
        "handlers": ["h"],
        "level": logging.INFO,
    },
)

logging.config.dictConfig(logging_config)
logging.captureWarnings(capture=True)

app.command(rich_help_panel="Evaluation Backends")(lighteval.main_accelerate.accelerate)
app.command(rich_help_panel="Evaluation Utils")(lighteval.main_baseline.baseline)
app.command(rich_help_panel="Evaluation Backends")(lighteval.main_nanotron.nanotron)
app.command(rich_help_panel="Evaluation Backends")(lighteval.main_vllm.vllm)
app.add_typer(
    lighteval.main_endpoint.app,
    name="endpoint",
    rich_help_panel="Evaluation Backends",
    help="Evaluate models using some endpoint (tgi, inference endpoint, openai) as backend.",
)
app.add_typer(
    lighteval.main_tasks.app,
    name="tasks",
    rich_help_panel="Utils",
    help="List or inspect tasks.",
)


if __name__ == "__main__":
    app()
