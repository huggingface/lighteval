# Use the Python API

Lighteval can be used from a custom python script. To evaluate a model you will
need to setup an `evaluation_tracker`, `pipeline_parameters`, `model_config`
and a `pipeline`.

After that, simply run the pipeline and save the results.


```python
import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_config import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


def main():
    evaluation_tracker = EvaluationTracker(
        output_dir="./results",
        save_details=True,
        push_to_hub=True,
        hub_results_org="SaylorTwift",
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
    )

    model_config = VLLMModelConfig(
            pretrained="HuggingFaceH4/zephyr-7b-beta",
            dtype="float16",
            use_chat_template=True,
    )

    task = "helm|mmlu|5|1"

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
        custom_task_directory=None, # if using a custom task
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

if __name__ == "__main__":
    main()
```
