"""
name:
InstructTTSEval

dataset:
kosw/instructtts-qwen-eval

abstract:
InstructTTSEval is a benchmark for evaluating Text-to-Speech systems' ability
to follow complex natural-language style instructions, across three
progressively challenging instruction types: APS (acoustic-parameter
specification), DSD (descriptive-style directive) and RP (role-play scenario).
It runs in two phases: first synthesize one wav per dataset row with the TTS
system under test, named {id}_{APS|DSD|RP}.wav; then this task scores each
clip with an audio-understanding judge model (default
Qwen/Qwen3-Omni-30B-A3B-Thinking) reached through any OpenAI-compatible
endpoint that accepts audio input (e.g. vLLM). The evaluated lighteval model
is a placeholder (use model=dummy) — the judge inside the metric produces the
scores. Configuration is via environment variables: INSTRUCTTTS_WAV_DIR
(folder with the generated wavs; when unset, the reference audio embedded in
the dataset is judged instead), INSTRUCTTTS_BASE_URL (judge endpoint, e.g.
http://localhost:8901/v1), INSTRUCTTTS_MODEL, INSTRUCTTTS_API_KEY,
INSTRUCTTTS_MAX_TOKENS, INSTRUCTTTS_TIMEOUT, and INSTRUCTTTS_DRYRUN=1 to
smoke-test the pipeline without a judge.

languages:
english

tags:
audio, tts, instruction-following, llm-as-judge

paper:
https://arxiv.org/abs/2506.16381
"""

import base64
import json
import logging
import os
import re
import tempfile

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


logger = logging.getLogger(__name__)

DEFAULT_REPO = "kosw/instructtts-qwen-eval"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
INSTRUCTION_TYPES = ["APS", "DSD", "RP"]
# The judge prompt template ships with the dataset; this placeholder is where
# the per-sample style instruction is inserted.
PROMPT_PLACEHOLDER = "<此处插入待评测的语音风格描述>"
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_TEMPLATE_CACHE: dict = {}
_AUDIO_TMPDIR: list = []


def _repo_id() -> str:
    return os.environ.get("INSTRUCTTTS_REPO", DEFAULT_REPO)


def _judge_prompt_template() -> str:
    repo = _repo_id()
    if repo not in _TEMPLATE_CACHE:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, "eval_prompt.txt", repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            _TEMPLATE_CACHE[repo] = f.read()
    return _TEMPLATE_CACHE[repo]


def _materialize_audio(audio, name: str) -> str:
    """Return a wav path for one dataset audio cell (dict or torchcodec decoder)."""
    if not _AUDIO_TMPDIR:
        _AUDIO_TMPDIR.append(tempfile.mkdtemp(prefix="instructtts_audio_"))
    path = os.path.join(_AUDIO_TMPDIR[0], f"{name}.wav")
    if os.path.isfile(path):
        return path
    if isinstance(audio, dict):
        if audio.get("bytes"):
            with open(path, "wb") as f:
                f.write(audio["bytes"])
            return path
        if audio.get("path") and os.path.isfile(audio["path"]):
            return audio["path"]
        if audio.get("array") is not None:
            import soundfile as sf

            sf.write(path, audio["array"], audio["sampling_rate"])
            return path
    if hasattr(audio, "get_all_samples"):  # datasets>=4 torchcodec AudioDecoder
        import soundfile as sf

        samples = audio.get_all_samples()
        sf.write(path, samples.data.numpy().T.squeeze(), samples.sample_rate)
        return path
    raise TypeError(f"Unsupported audio cell type: {type(audio)!r}")


def instructtts_prompt(line, task_name: str = ""):
    sample_id, instruction_type = line["id"], line["instruction_type"]
    specific = {
        "id": sample_id,
        "instruction_type": instruction_type,
        "instruction": line["instruction"],
    }
    wav_dir = os.environ.get("INSTRUCTTTS_WAV_DIR")
    if wav_dir:
        specific["audio_path"] = os.path.join(wav_dir, f"{sample_id}_{instruction_type}.wav")
    else:
        specific["audio_path"] = _materialize_audio(line["audio"], f"{sample_id}_{instruction_type}")
    return Doc(
        task_name=task_name,
        query=line["instruction"],
        choices=[],
        gold_index=[],
        specific=specific,
    )


def _normalize_verdict(value):
    """Coerce the judge's 一致性 field to a bool (it is often a string)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().strip('"“”').lower()
        if v in ("true", "yes", "符合", "一致", "是"):
            return True
        if v in ("false", "no", "不符合", "不一致", "否"):
            return False
    return None


def _extract_verdict(text: str):
    """Pull a normalized True/False verdict out of one judge response, else None."""
    for match in re.findall(r"\{(?:[^{}]|\"(?:\\.|[^\"\\])*\")*\}", text):
        try:
            parsed = json.loads(match)
        except json.JSONDecodeError:
            continue
        if "一致性" in parsed:
            return _normalize_verdict(parsed["一致性"])
    return None


def _judge_clip(audio_path: str, instruction: str):
    """Ask the audio judge whether one clip matches its style instruction."""
    from openai import OpenAI

    base_url = os.environ.get("INSTRUCTTTS_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "INSTRUCTTTS_BASE_URL is not set. The InstructTTSEval judge runs on an "
            "OpenAI-compatible endpoint that accepts audio input, e.g.\n"
            "  vllm serve tturing/Qwen3-Omni-30B-A3B-Thinking-FP8 "
            "--served-model-name Qwen/Qwen3-Omni-30B-A3B-Thinking "
            "--port 8901 --max-model-len 32768 --reasoning-parser qwen3\n"
            "then set INSTRUCTTTS_BASE_URL=http://localhost:8901/v1 "
            "(or set INSTRUCTTTS_DRYRUN=1 to smoke-test without a judge)."
        )
    client = OpenAI(
        base_url=base_url,
        api_key=os.environ.get("INSTRUCTTTS_API_KEY", "EMPTY"),
        timeout=float(os.environ.get("INSTRUCTTTS_TIMEOUT", "600")),
    )
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")
    audio_format = os.path.splitext(audio_path)[1].lstrip(".").lower() or "wav"
    prompt_text = _judge_prompt_template().replace(PROMPT_PLACEHOLDER, instruction)

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=os.environ.get("INSTRUCTTTS_MODEL", DEFAULT_JUDGE_MODEL),
                messages=[
                    {"role": "system", "content": "You are a speech analysis expert."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_b64, "format": audio_format},
                            },
                        ],
                    },
                ],
                # greedy first; sample on retry to escape a bad greedy trajectory
                temperature=0 if attempt == 0 else 0.7,
                max_tokens=int(os.environ.get("INSTRUCTTTS_MAX_TOKENS", "4096")),
            )
        except Exception as e:
            logger.warning(f"[{audio_path}] judge request failed on attempt {attempt + 1}: {e}")
            continue
        message = response.choices[0].message
        # Thinking judges answer after a reasoning trace: with a reasoning parser
        # the trace arrives in reasoning_content, otherwise inline in <think> tags.
        candidates = []
        content = (message.content or "").strip()
        if content:
            candidates.append(THINK_RE.sub("", content).strip() or content)
        reasoning = (getattr(message, "reasoning_content", None) or "").strip()
        if reasoning:
            candidates.append(reasoning)
        for text in candidates:
            verdict = _extract_verdict(text)
            if verdict is not None:
                return verdict
        logger.warning(f"[{audio_path}] no usable verdict in attempt {attempt + 1}")
    return None


class InstructTTSJudge(SampleLevelComputation):
    """Judge-as-metric: the audio judge produces the score; the model response is ignored."""

    def compute(self, doc: Doc, model_response=None, **kwargs) -> float:
        if os.environ.get("INSTRUCTTTS_DRYRUN"):
            return 1.0
        audio_path = doc.specific["audio_path"]
        if not os.path.isfile(audio_path):
            logger.warning(f"Missing generated audio, scoring 0: {audio_path}")
            return 0.0
        verdict = _judge_clip(audio_path, doc.specific["instruction"])
        return 1.0 if verdict else 0.0


qwen_judge_acc = SampleLevelMetric(
    metric_name="qwen_judge_acc",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=InstructTTSJudge(),
    corpus_level_fn=np.mean,
)


def _make_task(instruction_type: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"instructtts:{instruction_type.lower()}",
        prompt_function=instructtts_prompt,
        hf_repo=DEFAULT_REPO,
        hf_subset="default",
        # NOTE: row filtering must use hf_filter — the `filter` field is not applied to rows
        hf_filter=lambda line, t=instruction_type: line["instruction_type"] == t,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="",
        few_shots_select="random",
        metrics=[qwen_judge_acc],
        generation_size=16,
        stop_sequence=[],
    )


TASKS_TABLE = [_make_task(t) for t in INSTRUCTION_TYPES]
