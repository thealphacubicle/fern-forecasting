# fern-forecasting

MKTG4604 forecasting project. Data science stack; work lives in notebooks and scripts.

| Item | Value |
|---|---|
| Package | `fern_forecasting` |
| Python | 3.11+ (venv runs 3.13) |
| Package manager | `uv` |
| Linter/formatter | `ruff` |
| Build backend | `hatchling` |

## Commands

```bash
uv sync --extra dev                    # install / sync deps
uv run pre-commit install              # install git hooks (first-time)
uv run ruff check . && uv run ruff format .   # lint + format
uv run pre-commit run --all-files      # run all hooks
uv build                               # build dist
```

## Slash Commands

| Command | Purpose |
|---|---|
| `/add-module` | Scaffold a module under `src/fern_forecasting` |
| `/fix-lint` | Auto-fix Ruff issues, format, report remaining findings |

@.claude/rules/python-style.md
@.claude/rules/project-structure.md
@.claude/rules/dependencies.md
