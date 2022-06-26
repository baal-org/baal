# How can you contribute to Baal.

Thank you for your interest in contributing to Baal!

For support, don't hesitate to look at our [Support page](https://baal.readthedocs.io/en/latest/#support).
We are more than happy to do pair-coding sessions and quick meetings as needed.

There are several ways one can contribute to the project. 

### Bug fixes

Please submit a Pull Request (PR) that includes the fix as well as a test that would reproduce the issue.

### Documentation

Please submit a PR with the new content. 

### New features

Please submit an issue describing the feature. Once it is approved by a team member, you, or another contributor can submit a PR.

### New methods

To submit a new method, please submit an issue that would describe the new method with full references to *published* papers.
You must provide some early results that indicates that the new method is at least better than random.
Once the method is approved, you can submit a PR which would include:

1. The method.
2. The ModelWrapper class/patch method associated if needed.
3. An experiment script with results.
4. If you would like, you can submit a blog post in `/docs` describing your method.

## Getting Started

To contribute to Baal codebase, first clone our repo:

`git clone git@github.com:baal-org/baal.git`

Optionally, you can install `pyenv` to quickly manage Python versions:
```bash
curl https://pyenv.run | bash
pyenv install 3.9.13
pyenv global 3.9.13
```


Now install Poetry, our package manager:
`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`

Finally install Baal:

`cd baal && poetry install`

You can now start coding!

### Makefile

The Makefile allows us to make testing/linting quite easy.

```bash
make format # Format the code
make test # Run Pytest, Flake8 and Mypy
```

Thank you!