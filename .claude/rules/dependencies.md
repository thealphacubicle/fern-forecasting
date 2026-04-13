# Dependency Management Rules

## Adding Packages

```bash
uv add <package>            # runtime dependency
uv add --dev <package>      # development-only dependency
uv sync --extra dev         # sync lock file + environment after any change
```

Note: all data science deps (`numpy`, `pandas`, etc.) are currently in the `dev` group — move to runtime if the package is published.

## Gotchas
- Run `uv sync --extra dev` after editing `pyproject.toml` or the lock file; local imports won't resolve otherwise.
- Pre-commit may auto-fix staged files on commit; re-stage and retry until hooks pass clean.
- Do not install packages directly with `pip`; always go through `uv` to keep the lock file consistent.
