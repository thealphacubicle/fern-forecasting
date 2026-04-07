---
description: Scaffold a new module and matching tests.
---

# /add-module

## Inputs
1. Ask for module name (snake_case).
2. Ask for a one-line description of the module responsibility.

## Actions
1. Create `src/my_package/<name>.py` with:
   - Module docstring using the provided description.
   - One placeholder public function with full type hints.
   - A Google-style docstring for that function.
   - A minimal implementation body that is safe and deterministic.
2. Create `tests/test_<name>.py` with:
   - One happy-path test.
   - One edge-case test.
   - One `pytest.mark.parametrize` test (with explicit `ids=`).
3. If the module should export public symbols:
   - Update `src/my_package/__init__.py` to import and expose them in `__all__`.

## Output
- Summarize files created/updated.
- Remind the user to run:

```bash
uv run pytest
```
