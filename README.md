# Large-Scale Airbnb Price Prediction & Spatial Analytics Platform

> **LightGBM price prediction model trained on 95,144 London Airbnb listings.**
> R² = 0.67 · MAE = £25 · RMSE = £49 on held-out test set.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

This project builds a machine learning pipeline to predict nightly Airbnb prices across London using the [Inside Airbnb](http://insideairbnb.com/) 2024 dataset. It covers the full data science workflow — from exploratory analysis and feature engineering through model benchmarking, SHAP-based interpretability, and an interactive Streamlit dashboard.

The project also includes a PySpark vs Pandas scalability benchmark, demonstrating big-data processing on a 380,000-row synthetic dataset.

---

## Results

| Model              | MAE (£) | RMSE (£) | R²    | CV R² (5-fold)      |
|--------------------|---------|----------|-------|---------------------|
| Linear Regression  | £54.7   | £85.5    | 0.005 | 0.003 ± 0.002       |
| Random Forest      | £27.8   | £53.2    | 0.615 | 0.601 ± 0.012       |
| XGBoost            | £26.5   | £50.8    | 0.648 | 0.635 ± 0.011       |
| **LightGBM**       | **£25.3** | **£49.1** | **0.671** | **0.658 ± 0.009** |

> Metrics computed on 80/20 train-test split (random_state=42). Price values are approximate £ conversions from StandardScaler-normalised predictions.

---

## Key Findings

**What drives Airbnb prices in London?**

Based on SHAP analysis of the LightGBM model:

1. **Accommodates** — strongest predictor. Each additional guest capacity adds significant price premium.
2. **Bedrooms** — number of bedrooms is the second most impactful feature.
3. **Location** — latitude/longitude encode neighbourhood desirability. Westminster and Kensington & Chelsea command consistent premiums.
4. **Room type** — private rooms are priced at approximately 55% of equivalent entire-home listings.
5. **Availability** — listings with low 30-day availability (high demand) command higher prices.
6. **Review recency** — `days_since_review` negatively correlates with price: active, recently reviewed listings price higher.

---

## Dataset

- **Source:** [Inside Airbnb — London](http://insideairbnb.com/get-the-data/) (December 2024 scrape)
- **Size:** 95,144 listings · 75 raw features
- **Target:** Nightly price (£) — continuous regression
- **Data not included** in this repository due to size. Download from Inside Airbnb and place at `data/Airbnb_london.csv`.

---

## Project Structure

```
Large-Scale-Airbnb-Price-Prediction-Spatial-Analytics-Platform/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_EDA.ipynb                  ← Exploratory data analysis
│   ├── 02_preprocessing.ipynb        ← Cleaning & feature engineering
│   └── 03_modelling.ipynb            ← Model benchmarking & SHAP
│
├── src/
│   ├── preprocess.py                 ← Leakage-safe preprocessing pipeline
│   ├── train.py                      ← Training, CV, model serialisation
│   ├── evaluate.py                   ← SHAP analysis & diagnostic plots
│   └── visualise.py                  ← 10-figure visual analytics suite
│
├── app/
│   └── AirbnbDashboard.py            ← Streamlit interactive dashboard
│
├── models/
│   └── lgbm_pipeline.pkl             ← Trained LightGBM pipeline (generated)
│
├── figures/
│   ├── shap_beeswarm.png
│   ├── rmse_comparison.png
│   └── ...                           ← All figures (generated)
│
└── data/
    └── .gitkeep                      ← Data not committed (download separately)
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/Shaunfarhan16/Large-Scale-Airbnb-Price-Prediction-Spatial-Analytics-Platform.git
cd Large-Scale-Airbnb-Price-Prediction-Spatial-Analytics-Platform
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the data

Download `listings.csv.gz` from [Inside Airbnb — London](http://insideairbnb.com/get-the-data/) and place it at `data/Airbnb_london.csv`.

### 4. Run the preprocessing pipeline

```bash
python src/preprocess.py --data data/Airbnb_london.csv --out data/Airbnb_clean.csv
```

### 5. Train all models

```bash
python src/train.py --data data/Airbnb_clean.csv --out models/ --cv 5
```

### 6. Run SHAP evaluation

```bash
python src/evaluate.py --model models/lgbm_pipeline.pkl --data data/Airbnb_clean.csv
```

### 7. Generate all visualisations

```bash
python src/visualise.py --data data/Airbnb_clean.csv --out figures/
```

### 8. Launch the dashboard

```bash
streamlit run app/AirbnbDashboard.py
```

---

## Technical Details

### Preprocessing pipeline

- **Leakage prevention:** All price-derived columns removed before any modelling step.
- **Scaler leakage fix:** `StandardScaler` fitted on training set only, applied to test set via `transform()` only — never `fit_transform()` on the full dataset.
- **Missing values:** Review scores imputed with training-set median. `no_review_flag` binary indicator added.
- **Feature engineering:** `price_per_person`, `days_since_review`, `property_major` (top-10 bucketing), one-hot encoding of categoricals.
- **Outlier clipping:** 1st/99th percentile clip applied to numeric features.

### Model architecture

- All models wrapped in `sklearn.Pipeline` with `ColumnTransformer` for reproducible preprocessing.
- 5-fold cross-validation reported alongside held-out test metrics.
- Best model (`LightGBM`) saved via `joblib` for inference in the dashboard.

### SHAP interpretability

- `shap.TreeExplainer` applied to the trained LightGBM model.
- Beeswarm plot shows both magnitude and direction of feature impact.
- Bar plot shows global feature importance by mean absolute SHAP value.

### Big data benchmark

- Dataset replicated 4× to create a 380,000-row synthetic corpus.
- Aggregation benchmark: PySpark (`local[*]`) vs Pandas (single-thread).
- Results: PySpark delivers meaningful speed advantage on group-by aggregations at scale.

---

## Figures

| Figure | Description |
|--------|-------------|
| Fig 01 | Price distribution (log scale) |
| Fig 02 | Price by room type (boxplot) |
| Fig 03 | Median availability by month |
| Fig 04 | Hex-bin map of average price across London |
| Fig 05 | Spatial KDE of listing density |
| Fig 06 | Capacity vs price by room type |
| Fig 07 | Top-20 host market share (Pareto) |
| Fig 08 | Correlation matrix |
| Fig 09 | Review activity heatmap |
| Fig 10 | Amenities word cloud |
| Fig 11 | LightGBM SHAP beeswarm |
| Fig 12 | RMSE model comparison |

---

## Dashboard

The Streamlit dashboard provides interactive filtering by borough, month, and room type, a live price predictor powered by the trained LightGBM model, and a one-click PDF export of all charts.

```bash
streamlit run app/AirbnbDashboard.py
```

---

## Author

**Farhan Hashmi**
MSc Data Science & Artificial Intelligence — Keele University (Distinction, 2025)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-farhanhashmi-blue?logo=linkedin)](https://www.linkedin.com/in/shaunfarhan16/)
[![GitHub](https://img.shields.io/badge/GitHub-Shaunfarhan16-black?logo=github)](https://github.com/Shaunfarhan16)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
