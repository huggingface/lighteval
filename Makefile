.PHONY: style format


style:
	ruff format .
	ruff check --fix .


quality:
	ruff format --check .
	ruff check .
