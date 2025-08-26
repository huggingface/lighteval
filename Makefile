.PHONY: style format


style:
	uvx ruff format .
	uvx ruff check --fix .


quality:
	uvx ruff format --check .
	uvx ruff check .
