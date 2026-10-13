# Contributing

Thanks for helping out! Bug reports, ideas and pull requests are all welcome.

## Setup

```sh
uv sync
uv run pre-commit install   # lint, format and type-check on every commit
```

## Tests

```sh
uv run pytest tests/unit   # fast, no Docker needed
uv run pytest              # adds end-to-end tests; needs Docker
uv run pytest -m gpu       # serves real models; needs a free GPU
```

CI runs the pre-commit checks and the unit tests on every pull request. Please run the end-to-end tests locally if your change touches Docker, the model store or serving.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) without a scope: `type: description`, e.g. `feat: add model search` or `fix: handle missing GPU`. The pre-commit hook checks this for you.
