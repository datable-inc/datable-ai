CODE_DIR = .

ruff-format:
	poetry run ruff format .

ruff-check:
	poetry run ruff check --output-format=github --fix .

pre-commit-run:
	poetry run pre-commit run --all-files

code-check: ruff-format ruff-check pre-commit-run

test:
	poetry run pytest tests/ --doctest-modules --junitxml=junit/test-results.xml

.PHONY: code-check test
