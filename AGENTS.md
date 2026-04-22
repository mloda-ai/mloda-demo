# AGENTS.md

Must read [README.md](README.md) first.

This project uses the mloda framework. Assume any given task is related to mloda.

## Environment

```bash
source .venv/bin/activate
```

## Dependencies

Use `uv` to install dependencies:
```bash
uv sync --all-extras
```

## Running checks

Use `tox` to run all checks:
```bash
tox
```

Run checks from the project virtualenv:
```bash
source .venv/bin/activate && tox
```

`tox` is the required final verification step after code or dependency changes.
Running only `pytest` is not sufficient for completion.

### Run individual checks

```bash
pytest
ruff format --check --line-length 120 .
ruff check .
mypy --strict --ignore-missing-imports .
bandit -c pyproject.toml -r -q .
```

## Commit messages

Use Conventional Commit format for all commits so semantic versioning/release tooling can parse intent.
Do not include `Co-Authored-By` lines or any other mention of AI agents in commit messages.

Examples:
- `fix: handle empty feature set`
- `chore(deps): bump mloda to 0.4.6`

## mloda-registry: Plugin Development Resources

The [mloda-registry](https://github.com/mloda-ai/mloda-registry) provides essential documentation and patterns for mloda development:

**Documentation & Guides:**
- Plugin development guides: `docs/guides/` (40+ patterns and decision trees)
- Best practices for FeatureGroups, ComputeFrameworks, Extenders, and Linkers
- Testing utilities and conventions

**Claude Code Skills:**
- `.claude/skills/` — skills that assist with plugin development
- Leverage these for pattern guidance when implementing FeatureGroups, ComputeFrameworks, or Extenders

Consider generating project-specific skills for your own plugin repository to provide tailored AI assistance for your implementation patterns and conventions.
