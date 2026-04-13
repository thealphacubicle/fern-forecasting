# 🌿 Fern Forecasting

> AI-powered demand forecasting and customer review analysis for **Fern (Boston Rose Florist)** — a family-owned floral shop at 225 Massachusetts Ave, Boston, MA.

Built for MKTG 4604: *Creating Business Value with Data and AI Technologies* at Northeastern University.

**Team:** Blythe Berlinger, Molly Varrenti, Dhruv Laungani, Srihari Raman  
**Instructor:** Professor Daniel Katz

---

## 📌 Project Overview

Fern operates in a perishable inventory environment where demand spikes sharply around holidays (Valentine's Day, Mother's Day), local university events (Northeastern, Berklee graduations), and hospital/sympathy occasions. Without a data-driven system, the shop has historically relied on intuition for wholesale ordering.

This project introduces two AI-powered tools to address that:

1. **Occasion-Based Demand Forecasting** — a regression model that predicts weekly order volume by product category using calendar features, weather, and Google Search Trends as leading indicators.
2. **Review Sentiment & Topic Analysis** — an NLP pipeline that scores Yelp and Google reviews by sentiment and applies LDA topic modeling to surface which occasions and products drive customer satisfaction.

The two models are connected: sentiment analysis informs which occasions should be weighted more aggressively in inventory planning.

---

## 📂 Repository Structure

```
fern-forecasting/
│
├── data/
│   ├── raw/                     # Source CSVs (not committed if large / sensitive)
│   │   ├── fern_orders.csv      # 5,769 transaction records (Jan 2023 – Dec 2024)
│   │   ├── fern_calendar.csv    # Daily calendar: holidays, weather, university events
│   │   ├── fern_inventory.csv   # Weekly inventory: units ordered, sold, wasted
│   │   └── fern_reviews.csv     # 310 customer reviews (Google and Yelp)
│   └── processed/               # Cleaned or merged datasets produced in the pipeline
│
├── figures/                     # Plots saved from notebooks (e.g. EDA exports)
│
├── models/                      # Serialized trained models (.onnx, etc.); artifacts not committed
│
├── src/
│   ├── notebooks/
│   │   └── 01_eda.ipynb         # Exploratory data analysis across all four datasets
│   └── scripts/                 # Standalone Python scripts (as you add them)
│
│
├── pyproject.toml               # Project metadata and dependency groups (source of truth)
├── uv.lock                      # Locked versions for uv
├── requirements.txt             # Pinned export of the dev lockfile (for pip workflows)
└── README.md
```

Additional notebooks and modules for feature engineering, forecasting, and NLP are part of the project plan; only `01_eda.ipynb` lives in the repo so far.

---

## 📊 Data Overview

Raw tables live under `data/raw/` (see the repository tree above).

| Dataset | Rows | Granularity | Key Variables |
|---|---|---|---|
| `fern_orders.csv` | 5,769 | One row per transaction | `order_date`, `product_category`, `occasion_tag`, `quantity_sold`, `revenue` |
| `fern_calendar.csv` | 731 | One row per day | `holiday_name`, `days_until_next_major_holiday`, `is_university_event_week`, `avg_temp_f` |
| `fern_inventory.csv` | 942 | One row per product per week | `units_ordered`, `units_sold`, `units_wasted`, `unit_cost`, `unit_price` |
| `fern_reviews.csv` | 310 | One row per review | `platform`, `star_rating`, `review_text`, `occasion_mentioned` |

**Date coverage:** January 2023 – December 2024 across all datasets.

---

## 🔬 Methods

### Method 1: Demand Forecasting (Regression)

Predicts weekly order volume per product category. Features include:

- Day of week, season
- `days_until_next_major_holiday` (capped at 90; 999 sentinel recoded)
- `is_university_event_week`, `is_weekend`
- Temperature and precipitation
- Occasion flags engineered from order history (graduation week, hospital demand, etc.)
- Google Search Trends as leading demand indicators

**Target variable:** `quantity_sold` aggregated weekly by product category.

**Validation approach:** Time-based train/test split (train on 2023, test on 2024). The proposal also includes a 3-month live trial where Fern tracks actual inventory against model recommendations to demonstrate real-world value.

### Method 2: Sentiment Analysis + Topic Modeling (NLP)

Runs on `fern_reviews.csv`:

- **Sentiment scoring:** Scores each review as positive or negative using a pre-trained model.
- **Topic modeling:** Applies LDA to surface recurring themes (e.g., sympathy flowers, Valentine's roses, last-minute gifts, quality complaints).
- **Integration:** Sentiment scores are matched to identified topics and fed back into the forecasting model to weight high-satisfaction occasions more aggressively.

---

## ⚙️ Setup

### Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) (recommended) for installing from `pyproject.toml` / `uv.lock`

### Installation

```bash
git clone https://github.com/thealphacubicle/fern-forecasting.git
cd fern-forecasting
uv sync --group dev
```

If you prefer pip, install the pinned tree from the export file:

```bash
pip install -r requirements.txt
```

### Running the notebooks

Open `src/notebooks/` in Cursor or VS Code with the Jupyter extension, and select the project’s interpreter (`.venv` after `uv sync`). Alternatively, add Jupyter to the dev environment (`uv add --dev jupyter`) and run:

```bash
uv run jupyter notebook src/notebooks/
```

When additional notebooks are added, run them in pipeline order (EDA → feature work → modeling → NLP).

---

## 💡 Business Value

The expected output is a **recommended weekly inventory order** by product category, generated 3+ days in advance (matching the 3-day restock lead time in the inventory data). This gives Fern a data-backed alternative to intuition-based ordering and enables:

- Reduction in perishable waste
- Prevention of stockouts during high-demand holiday windows
- Targeted stocking based on which occasions and products drive customer satisfaction

---

## 📄 License

For academic use only. MKTG 4604, Northeastern University, Spring 2026.