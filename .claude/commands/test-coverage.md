---
description: Run coverage and suggest high-impact tests.
---

# /test-coverage

## Actions
1. Run:

```bash
uv run pytest --cov=src/my_package --cov-report=term-missing
```

## Reporting Requirements
- Report pass/fail status and total tests run.
- Provide a table of modules below 80% coverage including:
  - Module path
  - Coverage percent
  - Uncovered line ranges from term-missing output
- Suggest 2-3 specific tests that would most improve coverage, focused on:
  - Uncovered branches
  - Error handling paths
  - Public API edge cases
