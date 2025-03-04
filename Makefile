.PHONY: style format

check_dirs := tests src examples community_tasks


style:
	ruff format $(check_dirs)
	ruff check --fix $(check_dirs)

quality:
	ruff format --check $(check_dirs)
	ruff check $(check_dirs)
