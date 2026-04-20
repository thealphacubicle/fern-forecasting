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
├── src/
│   ├── fern_forecasting/                       # DASHBOARD ENGINE
│   │   ├── dashboard.py                        # functions: GB model, simulation, alerts
│   │   ├── features.py                         # feature engineering
│   │   └── preprocessing.py                    # data cleaning
│   ├── notebooks/                              # ANALYSIS NOTEBOOKS 
│   │   ├── 01_eda.ipynb                        # initial EDA, identifies waste problem
│   │   ├── 02_sentiment_analysis.ipynb         # VADER scoring and LDA topic modeling
│   │   └── 03_value_argument.ipynb             # Waste savings simulation
│   └── scripts/
│       ├── clean_raw.py
│       └── build_panel.py
├── archive/
│   └── fern_main_analysis.ipynb                # combined analysis (archived)
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

## Analysis Notebooks and Workflow

### 1. src/notebooks/01_eda.ipynb — Exploratory Data Analysis

- Introduces the four datasets and surfaces the core business problem.
- Merges orders, calendar, inventory, and review data
- Identifies strong demand spikes driven by holidays and occasions
- Quantifies inventory inefficiency

Key finding: Fern wasted $41,394 over 2023–2024 (6.9% of revenue), hence the need for data-driven ordering.

### 2. models/demand_forecasting.ipynb — Demand Forecasting Model

- Builds and evaluates the Gradient Boosting model used to predict demand.
- Constructs time-based and calendar features (holiday flags, lags, seasonality)
- Compares Linear Regression vs Gradient Boosting
- Evaluates model performance and interpretability

Results:
- Linear Regression fails (R² = -0.124) due to non-linearity. 
- RF was also tested, but had slightly lower R^2
- Gradient Boosting performs well (R² = 0.692, MAE = 2.31 units)

This notebook provides the modeling foundation for the Streamlit dashboard.

### 3. src/notebooks/02_sentiment_analysis.ipynb — Customer Sentiment Analysis

- Analyzes 310 customer reviews to uncover demand drivers.
- VADER sentiment scoring → overall sentiment = +0.47
- LDA topic modeling → identifies 5 key customer themes

Results:
- Walk-in customers = high volume, low satisfaction
- Valentine's Day & Mother’s Day = high demand AND high sentiment

### 4. src/notebooks/03_value_argument.ipynb — Business Impact Simulation

- Translates model predictions into financial value.
- Simulates model-driven ordering vs current approach
- Applies a 5% safety buffer
- Estimates cost savings

Result (initial simulation):
- 32% reduction in waste costs
- $851 projected annual savings

Updated analysis (EDA refinement):
- ~30–45% reduction under realistic forecast assumptions
- ~$6K–$9K annual savings
---

## Model Performance

Results from `models/demand_forecasting.ipynb`:

| Model | MAE | R² | Notes |
| Model | MAE | R² | Notes |
|---|---|---|---|
| Linear Regression | 4.49 | -0.124 | Baseline  (fails due to non-linear holiday spikes) |
| Gradient Boosting | 2.31 | 0.692 | Final model used in simulation and dashboard |

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
