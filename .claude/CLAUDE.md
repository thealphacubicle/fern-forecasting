# CLAUDE.md

## Quick Reference

| Item | Value |
| --- | --- |
| Python version | 3.11+ |
| Package manager | uv |
| Linter/formatter | ruff |
| Test framework | pytest |
| Build backend | hatchling |
| Layout | src/my_package + tests |

## Essential Commands

### First-time setup
```bash
uv sync --extra dev
uv run pre-commit install
```

### Run tests
```bash
uv run pytest -v
```

### Lint and format
```bash
uv run ruff check .
uv run ruff format .
```

### Pre-commit
```bash
uv run pre-commit run --all-files
```

### Build
```bash
uv build
```

## Code Conventions
- Use type hints for all function signatures and public attributes.
- Use Google-style docstrings for public modules, classes, and functions.
- Use double quotes for string literals.
- Use modern union syntax (`A | B`, `T | None`) instead of `typing.Union`/`typing.Optional`.
- Use built-in collection generics (`list[str]`, `dict[str, int]`) instead of `typing.List`/`typing.Dict`.
- Prefer `logging` over `print` in application code.

## Architecture

```text
.
|-- pyproject.toml
|-- src/
|   `-- my_package/
|       `-- __init__.py
`-- tests/
    `-- test_*.py
```

## Adding Dependencies with uv

### Runtime dependency
```bash
uv add <package>
```

### Development dependency
```bash
uv add --dev <package>
```

### Sync lock/environment after edits
```bash
uv sync --extra dev
```

## Gotchas
- Editable install behavior: run `uv sync --extra dev` after dependency or build config changes so local imports resolve consistently.
- Coverage HTML output is written to `htmlcov/index.html`; open it after running a coverage command.
- Pre-commit may auto-fix files; rerun hooks until clean, then rerun tests.

## Slash Commands

| Slash command | Purpose |
| --- | --- |
| /add-module | Scaffold a new module and matching tests under src/my_package and tests. |
| /fix-lint | Auto-fix Ruff issues, format code, and report remaining lint findings. |
| /test-coverage | Run coverage for src/my_package and report weak spots with test suggestions. |
