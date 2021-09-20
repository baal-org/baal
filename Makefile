lint:
	poetry run flake8 baal

test: lint
	poetry run pytest tests --cov=baal

format:
	poetry run black baal