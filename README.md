# Fern Forecasting
This project explores demand forecasting and customer sentiment analysis for Fern (Boston Rose Florist) a family-owned floral shop at 225 Massachusetts Ave, Boston, MA.

Built for MKTG 4604: *Creating Business Value with Data and AI Technologies*  
Northeastern University, Spring 2026

**Team:** Blythe Berlinger, Molly Varrenti, Dhruv Laungani, Srihari Raman  
**Instructor:** Professor Daniel Katz

---

## What This Project Does

Fern is a small florist that orders inventory based on intuition. This leads 
to two costly problems: over-ordering (flowers go to waste) and under-ordering 
(stockouts during holiday spikes). Over 2023–2024, Fern wasted **$41,394** in 
unsold inventory.

We built two AI tools to fix this:

1. **Demand Forecasting** : a Gradient Boosting model that predicts weekly 
   order volume per product category using holiday flags, weather, lag sales, 
   and university event calendars.

2. **Customer Sentiment Analysis** : a VADER + LDA pipeline that scores 310 
   customer reviews by occasion and surfaces which products and occasions drive 
   satisfaction (or dissatisfaction).

Both tools feed into a **Streamlit management dashboard** that gives Fern's 
owner a weekly order recommendation, a retrospective waste report, and a live 
customer sentiment tracker.

---

## Repository Structure

There are two distinct parts to this project:

**1. Analysis Notebooks** — proof-of-concept analysis submitted to Canvas  
**2. Dashboard Code** — interactive Streamlit tool built on top of the analysis

```
fern-forecasting/
├── app/                                        # DASHBOARD CODE
│   ├── pages/
│   │   ├── 1_Demand_Outlook.py
│   │   ├── 2_Order_Sheet.py
│   │   ├── 3_How_We_Did.py
│   │   └── 4_Customer_Sentiment_Analysis.py
│   └── Home.py                                 # home page with weekly alerts
├── data/
│   ├── raw/
│   │   ├── fern_orders.csv 
│   │   ├── fern_calendar.csv
│   │   ├── fern_inventory.csv
│   │   └── fern_reviews.csv
│   └── processed/
│       ├── orders_clean.parquet
│       ├── calendar_clean.parquet
│       ├── inventory_clean.parquet
│       ├── reviews_clean.parquet
│       └── weekly_panel.parquet
├── figures/
├── models/                                     # ANALYSIS NOTEBOOKS
│   ├── demand_forecasting.ipynb                # full EDA, identifies waste problem
│   └── fern_main_analysis.ipynb                # GB model training and evaluation
├── src/
│   ├── fern_forecasting/                       # DASHBOARD ENGINE
│   │   ├── dashboard.py                        # functions: GB model, simulation, alerts
│   │   ├── features.py                         # feature engineering
│   │   └── preprocessing.py                    # data cleaning
│   ├── notebooks/                              # ANALYSIS NOTEBOOKS 
│   │   ├── 01_eda.ipynb                        # extra EDA (can be ignored)
│   │   ├── 02_sentiment_analysis.ipynb         # VADER scoring and LDA topic modeling
│   │   └── 03_value_argument.ipynb             # Waste savings simulation
│   └── scripts/
│       ├── clean_raw.py
│       └── build_panel.py
├── pyproject.toml
├── requirements.txt
└── README.md
```
---

## Data

All data was simulated by Professor Katz based on our project proposal 
data wish list. Date coverage: **January 2023 – December 2024**.

| Dataset | Rows | Description |
|---|---|---|
| `fern_orders.csv` | 5,769 | One row per transaction: product, occasion, quantity, revenue |
| `fern_calendar.csv` | 731 | One row per day: holidays, weather, university event flags |
| `fern_inventory.csv` | 942 | One row per product per week: units ordered, sold, wasted |
| `fern_reviews.csv` | 310 | One row per review: platform, star rating, review text, occasion |

---

## Analysis Notebooks and Models

These four notebooks are the analytical foundation of the project. 
Run them in this order:

### 1. `models/fern_main_analysis.ipynb` — EDA
Loads all four raw datasets and surfaces the core business problem. 
Key finding: Fern wasted **$41,394** over 2023–2024 (10.8% of revenue), 
motivating the need for data-driven inventory management.

### 2. `models/demand_forecasting.ipynb` — Demand Forecasting Model
Builds and evaluates the Gradient Boosting demand forecasting model.
- Tested Linear Regression as baseline — failed (R² = -0.124), proving 
  demand is non-linear due to holiday spikes
- Gradient Boosting succeeded (R² = 0.664, MAE = 2.31 units)
- Contains feature engineering, leakage detection, and model interpretation
- This is the analytical foundation for `dashboard.py`

### 3. `src/notebooks/02_sentiment_analysis.ipynb` — Customer Sentiment Analysis
Analyzes 310 customer reviews using two NLP methods:
- **VADER sentiment scoring** — 77% positive overall, avg score +0.47
- **LDA topic modeling** — 5 topics: Valentine's/romance, delivery/sympathy, 
  special occasions, holiday planning, walk-in/houseplants
- Key finding: walk-in customers are highest volume but lowest satisfaction
- Valentine's Day and Mother's Day have both high sentiment AND high demand

### 4. `src/notebooks/03_value_argument.ipynb` — Waste Savings Simulation
Translates model predictions into business value:
- Simulates model-based ordering (5% buffer) vs Fern's current approach
- Result: **32% reduction in waste costs**, **$851 projected annual savings**
- Connected to the "How We Did" dashboard page which runs this simulation live


---

## Model Performance

Results from `models/demand_forecasting.ipynb`:

| Model | MAE | R² | Notes |
| Model | MAE | R² | Notes |
|---|---|---|---|
| Linear Regression | 4.49 | -0.124 | Baseline  (fails due to non-linear holiday spikes) |
| Gradient Boosting | 2.31 | 0.664 | Final model used in simulation and dashboard |

The Linear Regression failure is intentional. It demonstrates that 
floral demand is non-linear and justifies the Gradient Boosting approach.

---

## Streamlit Dashboard

The dashboard in `app/` turns the notebook analysis into a tool 
Fern's owner could use every week. Run it with:

```bash
streamlit run app/Home.py
```

| Page | What It Does |
|---|---|
| **Home** | Alert-driven summary (flags waste, stockouts, sentiment drops) |
| **Demand Outlook** | 8-week forward forecast by product category |
| **Order Sheet** | Recommended weekly order with adjustable safety buffer |
| **How We Did** | Retrospective waste simulation vs model-based ordering |
| **Customer Sentiment** | VADER scores, occasion leaderboard, sentiment trend |

---

## Key Findings

- Fern wasted **$41,394** over 2023–2024 - 10.8% of sold-unit revenue
- Our GB model predicts weekly demand within **~2.3 units** on average
- Model-based ordering reduces waste costs by **32%** (~$851/year)
- **Birthday and wedding** arrangements have the highest customer sentiment
- **Walk-in customers** are the highest-volume but lowest-satisfaction 
  segment, suggesting understocking for impulse buyers
- **Valentine's Day and Mother's Day** have both high demand AND high 
  sentiment, confirming these are the right occasions to stock heavily

---

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended)

### Installation
```bash
git clone https://github.com/thealphacubicle/fern-forecasting.git
cd fern-forecasting
uv sync --group dev
```

Or with pip:
```bash
pip install -r requirements.txt
```

### Running the Notebooks
Open `src/notebooks/` in VS Code with the Jupyter extension. 
Run in order: `01_eda` → `02_sentiment_analysis` → `03_value_argument`

### Running the Dashboard
```bash
streamlit run app/Home.py
```

---

## Extra Note

For academic use only. MKTG 4604, Northeastern University, Spring 2026.
