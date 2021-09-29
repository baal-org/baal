lint:
	poetry run flake8 baal

test: lint
	poetry run pytest tests --cov=baal

format:
	poetry run black baal

requirements.txt: poetry.lock
	poetry export --without-hashes -f requirements.txt > requirements.txt