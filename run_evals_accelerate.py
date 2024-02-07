import argparse

from lighteval.main_accelerate import CACHE_DIR, main


def get_parser():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    weight_type_group = parser.add_mutually_exclusive_group()

    weight_type_group.add_argument(
        "--delta_weights",
        action="store_true",
        default=False,
        help="set to True of your model should be merged with a base model, also need to provide the base model name",
    )
    weight_type_group.add_argument(
        "--adapter_weights",
        action="store_true",
        default=False,
        help="set to True of your model has been trained with peft, also need to provide the base model name",
    )
    parser.add_argument(
        "--base_model", type=str, default=None, help="name of the base model to be used for delta or adapter weights"
    )

    parser.add_argument("--model_args", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_dtype", type=str, default=None)
    parser.add_argument(
        "--multichoice_continuations_start_space",
        action="store_true",
        help="Whether to force multiple choice continuations starts with a space",
    )
    parser.add_argument(
        "--no_multichoice_continuations_start_space",
        action="store_true",
        help="Whether to force multiple choice continuations do not starts with a space",
    )
    parser.add_argument("--push_results_to_hub", default=False, action="store_true")
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument("--push_details_to_hub", default=False, action="store_true")
    parser.add_argument(
        "--public_run", default=False, action="store_true", help="Push results and details to a public repo"
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--override_batch_size", type=int, default=-1)
    parser.add_argument("--dataset_loading_processes", type=int, default=1)
    parser.add_argument("--inference_server_address", type=str, default=None)
    parser.add_argument("--inference_server_auth", type=str, default=None)
    parser.add_argument("--num_fewshot_seeds", type=int, default=1, help="Number of trials the few shots")
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument(
        "--results_org",
        type=str,
        help="Hub organisation where you want to store the results. Your current token must have write access to it",
    )
    parser.add_argument("--job_id", type=str, help="Optional Job ID for future reference", default="")
    parser.add_argument("--use_chat_template", default=False, action="store_true")
    parser.add_argument(
        "--custom_tasks_file",
        type=str,
        default=None,
        help="Path to a file with custom tasks (a TASK list of dict and potentially prompt formating functions)",
    )
    group.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Id of a task, e.g. 'original|mmlu:abstract_algebra|5' or path to a texte file with a list of tasks",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args, unknowns = parser.parse_known_args()
    main(args)
