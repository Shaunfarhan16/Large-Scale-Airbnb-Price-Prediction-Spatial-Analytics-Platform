"""
train.py
========
Airbnb London 2024 — model training, cross-validation, and serialisation.

Models benchmarked:
    - Linear Regression  (baseline)
    - Random Forest
    - XGBoost
    - LightGBM           (best performer: R²=0.67, MAE=£25, RMSE=£49)

Results
-------
| Model              | MAE (£) | RMSE (£) |  R²  |
|--------------------|---------|----------|------|
| Linear Regression  |  54.7   |   85.5   | 0.005|
| Random Forest      |  27.8   |   53.2   | 0.615|
| XGBoost            |  26.5   |   50.8   | 0.648|
| LightGBM           |  25.3   |   49.1   | 0.671|

Usage:
    python src/train.py --data data/Airbnb_clean.csv --out models/
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.dpi": 300, "figure.figsize": (7, 4),
                     "font.family": "DejaVu Sans"})
sns.set_style("whitegrid")

# Price scale constants (derived from dataset)
PRICE_STD  = 85.0   # £ — approximate std of London Airbnb nightly price
PRICE_MEAN = 120.0  # £ — approximate mean after 1/99 pct clipping


def load_data(data_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load preprocessed CSV and return features and target.

    Parameters
    ----------
    data_path : str | Path
        Path to the clean Airbnb CSV (output of preprocess.py).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) where X is the feature matrix and y is the price target.
    """
    df = pd.read_csv(data_path)
    print(f"[data] Loaded: {df.shape}")

    y = df["price"]
    X = df.drop(columns="price")

    # Remove any remaining price-derived leakage columns
    leak_cols = [c for c in X.columns if "price" in c.lower()]
    if leak_cols:
        print(f"[data] Removing leak columns: {leak_cols}")
        X = X.drop(columns=leak_cols)

    # Numeric/boolean only
    X = X.select_dtypes(include=["number", "bool"]).dropna(axis=1, how="all")

    # Drop rows with missing target
    mask = ~y.isna()
    X, y = X.loc[mask], y[mask]

    print(f"[data] Clean shape: {X.shape} | Target range: {y.min():.3f} to {y.max():.3f}")
    return X, y


def build_pipeline(estimator) -> Pipeline:
    """
    Wrap an estimator in a preprocessing + model pipeline.

    Uses median imputation for residual NaNs. Scaler is NOT applied
    here — assumed to have been applied in preprocess.py on train only.

    Parameters
    ----------
    estimator : sklearn-compatible estimator

    Returns
    -------
    Pipeline
    """
    prep = ColumnTransformer(
        [("imputer", SimpleImputer(strategy="median"), slice(0, None))],
        remainder="passthrough",
    )
    return Pipeline([("prep", prep), ("model", estimator)])


def evaluate(
    name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    price_std: float = PRICE_STD,
) -> dict:
    """
    Compute MAE, RMSE, and R² — both in scaled units and estimated £.

    Parameters
    ----------
    name : str
        Model name for display.
    y_true : pd.Series
        Ground truth target values.
    y_pred : np.ndarray
        Model predictions.
    price_std : float
        Standard deviation of original price in £ for inverse scaling.

    Returns
    -------
    dict
        Dictionary of metric values.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # Convert scaled metrics to approximate £
    mae_gbp  = mae  * price_std
    rmse_gbp = rmse * price_std

    print(f"  {name:<22} MAE: £{mae_gbp:.1f} | RMSE: £{rmse_gbp:.1f} | R²: {r2:.4f}")
    return {
        "mae_scaled": round(mae, 4),
        "rmse_scaled": round(rmse, 4),
        "r2": round(r2, 4),
        "mae_gbp": round(mae_gbp, 1),
        "rmse_gbp": round(rmse_gbp, 1),
    }


def plot_rmse_comparison(results: dict, out_dir: Path) -> None:
    """
    Bar chart comparing RMSE across all models.

    Parameters
    ----------
    results : dict
        Dictionary of model name → metrics dict.
    out_dir : Path
        Directory to save the figure.
    """
    names = list(results.keys())
    rmse_vals = [results[n]["rmse_gbp"] for n in names]

    fig, ax = plt.subplots()
    bars = sns.barplot(x=names, y=rmse_vals, palette="Blues_d", ax=ax)
    ax.set_ylabel("RMSE (£)")
    ax.set_xlabel("Model")
    ax.set_title("Model Comparison — RMSE on Held-Out Test Set (£)")

    for bar, val in zip(ax.patches, rmse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"£{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = out_dir / "rmse_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[plot] Saved: {path}")
    plt.close()


def plot_parity(
    name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Predicted vs Actual parity scatter plot.

    Parameters
    ----------
    name : str
        Model name (used in title and filename).
    y_true : pd.Series
        Ground truth values.
    y_pred : np.ndarray
        Model predictions.
    out_dir : Path
        Directory to save the figure.
    """
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual (scaled)"); ax.set_ylabel("Predicted (scaled)")
    ax.set_title(f"{name} — Predicted vs Actual")
    fig.tight_layout()
    path = out_dir / f"parity_{name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main(data_path: str, out_dir: str, cv_folds: int = 5) -> None:
    """
    Full training pipeline: load → split → train → evaluate → save.

    Parameters
    ----------
    data_path : str
        Path to clean Airbnb CSV.
    out_dir : str
        Directory to save trained models and metrics.
    cv_folds : int
        Number of cross-validation folds (default 5).
    """
    out_path    = Path(out_dir)
    fig_path    = Path("figures")
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    # Load
    X, y = load_data(data_path)

    # Split — BEFORE any fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[split] Train: {X_train.shape} | Test: {X_test.shape}")

    # Models to benchmark
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbosity=0
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            random_state=42, verbose=-1
        ),
    }

    all_results = {}
    best_r2     = -np.inf
    best_pipe   = None
    best_name   = ""

    print("\n── Training & evaluation ──────────────────────────────────────")
    for name, estimator in models.items():
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Test-set metrics
        metrics = evaluate(name, y_test, y_pred)

        # 5-fold cross-validation R² on training set
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=cv_folds, scoring="r2", n_jobs=-1
        )
        metrics["cv_r2_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_r2_std"]  = round(cv_scores.std(),  4)
        print(f"    CV R² ({cv_folds}-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        all_results[name] = metrics
        plot_parity(name, y_test, y_pred, fig_path)

        # Track best model
        if metrics["r2"] > best_r2:
            best_r2   = metrics["r2"]
            best_pipe = pipe
            best_name = name

    # Save best model
    model_file = out_path / "lgbm_pipeline.pkl"
    joblib.dump(best_pipe, model_file)
    print(f"\n[save] Best model ({best_name}) saved to: {model_file}")

    # Save metrics JSON
    metrics_file = out_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[save] Metrics saved to: {metrics_file}")

    # Plot RMSE comparison
    plot_rmse_comparison(all_results, fig_path)

    # Print final summary table
    print("\n── Final results ──────────────────────────────────────────────")
    summary = pd.DataFrame(all_results).T[
        ["mae_gbp", "rmse_gbp", "r2", "cv_r2_mean", "cv_r2_std"]
    ].sort_values("rmse_gbp")
    summary.columns = ["MAE (£)", "RMSE (£)", "R² (test)", "CV R² mean", "CV R² std"]
    print(summary.to_string())
    print(f"\nBest model: {best_name} | R²={best_r2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Airbnb price prediction models")
    parser.add_argument("--data", type=str, default="data/Airbnb_clean.csv",
                        help="Path to clean Airbnb CSV")
    parser.add_argument("--out",  type=str, default="models/",
                        help="Directory to save model artifacts")
    parser.add_argument("--cv",   type=int, default=5,
                        help="Number of cross-validation folds")
    args = parser.parse_args()

    main(args.data, args.out, args.cv)
