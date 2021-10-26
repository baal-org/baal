.PHONY: lint
lint: check-mypy-error-count
	poetry run flake8 baal experiments

.PHONY: test
test: lint
	poetry run pytest tests --cov=baal

.PHONY: format
format:
	poetry run black baal experiments

.PHONY: requirements.txt
requirements.txt: poetry.lock
	poetry export --without-hashes -f requirements.txt > requirements.txt

.PHONY: mypy
mypy:
	poetry run mypy --show-error-codes baal


.PHONY: check-mypy-error-count
check-mypy-error-count: MYPY_INFO = $(shell expr `poetry run mypy baal | grep ": error" | wc -l`)
check-mypy-error-count: MYPY_ERROR_COUNT = 16

check-mypy-error-count:
	@if [ ${MYPY_INFO} -gt ${MYPY_ERROR_COUNT} ]; then \
		echo mypy error count $(MYPY_INFO) is more than $(MYPY_ERROR_COUNT); \
		false; \
	fi