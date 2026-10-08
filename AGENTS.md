# Agent guidelines

## Running code and tests

Use pixi to run Python code and tests (the project requires Python 3.12).
Prefer the `mace-cpu` environment by default:

```sh
pixi run -e mace-cpu pytest tests/
pixi run -e mace-cpu python script.py
```

The `mace` environment requires CUDA >= 12 and fails on machines without a GPU.

## Linting and formatting

Run the git hooks (ruff, ty, etc.) with pre-commit:

```sh
pixi run -e mace-cpu pre-commit run --files <changed files>
```
