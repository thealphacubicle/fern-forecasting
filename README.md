# Fern Forecasting

AI-powered demand forecasting and customer sentiment analysis for **Fern (Boston Rose Florist) a family-owned floral shop at 225 Massachusetts Ave, Boston, MA.

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

```
fern-forecasting/
├── app/
│   ├── pages/
│   │   ├── 1_Demand_Outlook.py
│   │   ├── 2_Order_Sheet.py
│   │   ├── 3_How_We_Did.py
│   │   └── 4_Customer_Sentiment_Analysis.py
│   └── Home.py
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
├── models/
│   ├── demand_forecasting.ipynb
│   └── fern_main_analysis.ipynb
├── src/
│   ├── fern_forecasting/
│   │   ├── dashboard.py
│   │   ├── features.py
│   │   └── preprocessing.py
│   ├── notebooks/
│   │   ├── 01_eda.ipynb
│   │   ├── 02_sentiment_analysis.ipynb
│   │   └── 03_value_argument.ipynb
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

## Analysis Notebooks

The three notebooks in `src/notebooks/` are the core deliverables. 
Run them in order:

### `src/notebooks/01_eda.ipynb` — Exploratory Data Analysis
- Loads all four raw datasets and profiles each one (shape, dtypes, date ranges)
- **Calendar:** explores holiday distribution, weather patterns, university 
  event weeks across 731 days (Jan 2023–Dec 2024)
- **Inventory:** profiles units ordered vs sold vs wasted by product category; 
  surfaces waste patterns that motivate the forecasting model
- **Orders:** explores transaction volume by occasion, product category, 
  channel (in-store vs online), and day of week
- **Reviews:** profiles star rating distribution across 310 reviews by 
  platform (Google vs Yelp) and occasion
- Key output: establishes the data quality and structure that feeds into 
  `demand_forecasting.ipynb` and `02_sentiment_analysis.ipynb`

### `src/notebooks/02_sentiment_analysis.ipynb`: Customer Sentiment Analysis
- Scores all 310 reviews using **VADER** sentiment analysis
- Applies **LDA topic modeling** to find 5 recurring themes in review text
- Key finding: walk-in customers are Fern's highest-volume but 
  lowest-satisfaction segment; Valentine's Day and Mother's Day have 
  both high volume and high sentiment
- Overall: 77% positive rate, avg sentiment score of +0.47

### `src/notebooks/03_value_argument.ipynb` : Waste Savings Simulation
- Trains a **Gradient Boosting** model on the weekly panel dataset
- Simulates model-based ordering (5% buffer) vs Fern's current 
  intuition-based ordering
- Result: **32% reduction in waste costs**, projecting **$851 in 
  annual savings**
- Connected to the dashboard's "How We Did" page which implements 
  the same simulation interactively

### `models/demand_forecasting.ipynb`: Feature Engineering & Model Development
- Built and tested the full feature engineering pipeline on the weekly panel dataset
- Identified and removed data leakage (dropped same-week derivatives like 
  `total_revenue`, `units_wasted`, `sell_through_rate`)
- Compared Linear Regression vs Gradient Boosting — LR failed (R² = -0.124) 
  proving demand is non-linear; GB succeeded (R² = 0.664, MAE = 2.31)
- Contains the feature importance analysis and model interpretation
- **This notebook is the analytical foundation for the dashboard's 
  `fit_all_forecasts()` function in `dashboard.py`**

### `models/fern_main_analysis.ipynb`: Combined EDA and Full Analysis
- Loads all four raw datasets and merges into a unified daily-level DataFrame
- Full EDA: holiday demand spikes, waste by product category, revenue trends
- Key finding: **$41,394 in total waste costs** over 2023–2024 (10.8% of revenue)
- Contains sentiment analysis integration showing which occasions feed into 
  the demand forecasting features
- Serves as the complete end-to-end analysis narrative for the presentation

---

## Model Performance

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
