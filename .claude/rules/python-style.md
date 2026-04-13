# Python Style Rules

## Type Hints
- Required on all function signatures and public attributes.
- Use modern union syntax: `A | B`, `T | None` — not `Union[A, B]` or `Optional[T]`.
- Use built-in collection generics: `list[str]`, `dict[str, int]` — not `List`, `Dict` from `typing`.

## Docstrings
- Google-style for all public modules, classes, and functions.
- Private helpers (`_name`) do not require docstrings unless logic is non-obvious.

## Formatting
- Double quotes for all string literals.
- Enforced by `ruff format` — do not override manually.

## Logging
- Use `logging` in all application/script code.
- `print` is acceptable inside notebooks only.

## Imports
- Standard library → third-party → local; one blank line between groups.
- No wildcard imports (`from x import *`).
