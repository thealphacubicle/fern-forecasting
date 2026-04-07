#!/usr/bin/env bash
set -euo pipefail

echo "[devcontainer] Installing Claude Code..."
npm install --global @anthropic-ai/claude-code

echo "[devcontainer] Creating Python virtual environment..."
uv venv .venv

echo "[devcontainer] Installing project dependencies with uv..."
if [[ -f "pyproject.toml" ]]; then
  uv sync --extra dev
elif [[ -f "requirements.txt" ]]; then
  uv pip install --python .venv/bin/python -r requirements.txt
else
  echo "[devcontainer] No pyproject.toml or requirements.txt found; skipping dependency install."
fi
