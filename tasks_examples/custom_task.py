TASKS_TABLE = [
    {
        "name": "mmlu:anatomy",
        "suite": ["custom"],
        "prompt_function": "mmlu_anatomy",
        "hf_repo": "lighteval/mmlu",
        "hf_subset": "anatomy",
        "hf_avail_splits": ["auxiliary_train", "test", "validation", "dev"],
        "evaluation_splits": ["test"],
        "few_shots_split": "dev",
        "few_shots_select": "sequential",
        "generation_size": 5,
        "metric": ["loglikelihood_acc_single_token"],
        "stop_sequence": ["\n"],
        "output_regex": None,
        "frozen": False,
    },
    {
        "name": "mmlu:anatomy_signs",
        "suite": ["custom"],
        "prompt_function": "mmlu_anatomy_signs",
        "hf_repo": "lighteval/mmlu",
        "hf_subset": "anatomy",
        "hf_avail_splits": ["auxiliary_train", "test", "validation", "dev"],
        "evaluation_splits": ["test"],
        "few_shots_split": "dev",
        "few_shots_select": "sequential",
        "generation_size": 5,
        "metric": ["loglikelihood_acc_single_token"],
        "stop_sequence": ["\n"],
        "output_regex": None,
        "frozen": False,
    },
]


def mmlu_anatomy_signs(line):
    return mmlu_signs(line, "anatomy")


def mmlu_anatomy(line):
    return mmlu_numbers(line, "anatomy")


def mmlu_numbers(line, topic):
    prompt = f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}.\n\n"
    prompt += line["question"] + "\n"
    prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(["1", "2", "3", "4"], line["choices"])])
    prompt += "Answer:"

    gold_ix = ["1", "2", "3", "4"].index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]
    is_few_shots = line.get("__few_shots", False)  # We are adding few shots

    return {
        "query": prompt,
        "choices": [" 1", " 2", " 3", " 4"] if is_few_shots else ["1", "2", "3", "4"],
        "target_for_fewshot_sorting": [" 1", " 2", " 3", " 4"][gold_ix],
        "gold_index": gold_ix,
        "instruction": f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}.\n\n",
    }


def mmlu_signs(line, topic):
    prompt = f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}.\n\n"
    prompt += line["question"] + "\n"
    prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(["+", "*", "=", "#"], line["choices"])])
    prompt += "Answer:"

    gold_ix = ["+", "*", "=", "#"].index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]
    is_few_shots = line.get("__few_shots", False)  # We are adding few shots

    return {
        "query": prompt,
        "choices": [" +", " *", " =", " #"] if is_few_shots else ["+", "*", "=", "#"],
        "target_for_fewshot_sorting": [" +", " *", " =", " #"][gold_ix],
        "gold_index": gold_ix,
        "instruction": f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}.\n\n",
    }
