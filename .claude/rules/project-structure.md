# Project Structure

```
fern-forecasting/
├── pyproject.toml
└── src/
    ├── notebooks/          # Jupyter EDA and analysis notebooks
    │   └── 01_eda.ipynb
    ├── scripts/            # Standalone runnable scripts
    └── fern_forecasting/   # Importable package (add modules here)
        └── __init__.py
```

## Layout Rules
- Notebooks live in `src/notebooks/`; name them `NN_description.ipynb` (zero-padded number prefix).
- Runnable scripts live in `src/scripts/`; they import from `fern_forecasting` — do not inline business logic.
- Reusable logic belongs in `src/fern_forecasting/` as importable modules, not in notebooks or scripts.
- Do not create files outside this structure without a clear reason.

## Data Stack
Runtime deps (all `dev` group in `pyproject.toml`): `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `nltk`, `onnxruntime`.
