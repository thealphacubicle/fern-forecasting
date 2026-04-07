---
description: Auto-fix and validate Ruff lint status.
---

# /fix-lint

## Actions
1. Run:

```bash
uv run ruff check --fix .
```

2. Then run:

```bash
uv run ruff format .
```

3. Run final validation:

```bash
uv run ruff check .
```

## Reporting Requirements
- Report how many issues were auto-fixed in step 1.
- If final validation fails, list each remaining issue with `file:line`.
- For each unfixable issue type, explain the likely manual fix in one concise sentence.
- If validation passes, explicitly state lint is clean.
