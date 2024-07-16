def tasks_to_string(tasks: list, n_fewshot: int = 0) -> str:
    return ",".join([f"custom|{t.name}|{n_fewshot}|1" for t in tasks])