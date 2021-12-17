default:
	black --check .
	mypy -p simplifier
	pydocstyle simplifier
	isort -c .
	pytest

coverage:
	coverage run --source . -m pytest
	coverage report
	rm -f .coverage

format:
	black .
	isort .
