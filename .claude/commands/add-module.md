---
description: Scaffold a new module under src/fern_forecasting.
---

# /add-module

## Inputs
1. Ask for module name (snake_case).
2. Ask for a one-line description of the module responsibility.

## Actions
1. Create `src/fern_forecasting/<name>.py` with:
   - Module docstring using the provided description.
   - One placeholder public function with full type hints.
   - A Google-style docstring for that function.
   - A minimal implementation body that is safe and deterministic.
2. If the module should export public symbols:
   - Update `src/fern_forecasting/__init__.py` to import and expose them in `__all__`.

## Output
- Summarize files created/updated.
