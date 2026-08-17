"""
evaluate.py
===========
Airbnb London 2024 — model evaluation, SHAP explainability, and diagnostics.

Loads the saved LightGBM pipeline and produces:
    - SHAP beeswarm summary plot (Fig 1)
    - SHAP bar plot — mean absolute SHAP values (Fig 2)
    - Residual KDE (Fig 3)
    - Predicted vs Actual parity (Fig 4)
    - Feature importance (gain) bar chart (Fig 5)

Usage:
    python src/evaluate.py --model models/lgbm_pipeline.pkl --data data/Airbnb_clean.csv
"""

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.dpi": 300, "figure.figsize": (9, 5),
                     "font.family": "DejaVu Sans"})
sns.set_style("whitegrid")

PRICE_STD  = 85.0
SHAP_SAMPLE = 3000  # rows to use for SHAP (full dataset is slow)


def load_model_and_data(
    model_path: str | Path,
    data_path: str | Path,
) -> tuple:
    """
    Load saved pipeline and test data.

    Parameters
    ----------
    model_path : str | Path
        Path to saved joblib pipeline.
    data_path : str | Path
        Path to clean Airbnb CSV.

    Returns
    -------
    tuple
        (pipeline, X_test, y_test, feature_names)
    """
    pipe = joblib.load(model_path)
    print(f"[load] Model loaded from: {model_path}")

    df = pd.read_csv(data_path)
    y  = df["price"]
    X  = df.drop(columns="price")

    leak_cols = [c for c in X.columns if "price" in c.lower()]
    X = X.drop(columns=leak_cols, errors="ignore")
    X = X.select_dtypes(include=["number", "bool"]).dropna(axis=1, how="all")

    mask = ~y.isna()
    X, y = X.loc[mask], y[mask]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"[load] Test set: {X_test.shape}")
    return pipe, X_test, y_test, X.columns.tolist()


def print_metrics(y_true: pd.Series, y_pred: np.ndarray) -> None:
    """Print MAE, RMSE, R² in scaled and £ units."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n── LightGBM Test Metrics ──")
    print(f"  MAE  : {mae:.4f} scaled  ≈  £{mae  * PRICE_STD:.1f}")
    print(f"  RMSE : {rmse:.4f} scaled  ≈  £{rmse * PRICE_STD:.1f}")
    print(f"  R²   : {r2:.4f}")


def plot_shap_beeswarm(
    pipe,
    X_test: pd.DataFrame,
    feature_names: list,
    out_dir: Path,
) -> None:
    """
    SHAP beeswarm summary plot — shows feature impact direction and magnitude.

    Parameters
    ----------
    pipe : Pipeline
        Trained sklearn pipeline.
    X_test : pd.DataFrame
        Test features.
    feature_names : list
        Original feature names.
    out_dir : Path
        Directory to save figure.
    """
    print("[shap] Computing SHAP values (this may take ~30s)...")

    # Prepare transformed features for SHAP
    prep      = pipe.named_steps["prep"]
    model     = pipe.named_steps["model"]
    X_t       = prep.transform(X_test)

    # Sample for speed
    idx       = np.random.choice(len(X_t), min(SHAP_SAMPLE, len(X_t)), replace=False)
    X_sample  = X_t[idx]

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_sample)

    # Beeswarm — shap.summary_plot manages its own figure/axes (no `ax` kwarg)
    shap.summary_plot(
        shap_vals, X_sample,
        feature_names=feature_names,
        show=False, plot_size=(9, 6)
    )
    fig = plt.gcf()
    plt.gca().set_title("LightGBM SHAP Summary — Feature Impact on Price Prediction")
    fig.tight_layout()
    path = out_dir / "shap_beeswarm.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[shap] Saved: {path}")

    # Bar plot — mean |SHAP|
    shap.summary_plot(
        shap_vals, X_sample,
        feature_names=feature_names,
        plot_type="bar", show=False, plot_size=(8, 5)
    )
    fig2 = plt.gcf()
    plt.gca().set_title("LightGBM — Mean Absolute SHAP Value (Feature Importance)")
    fig2.tight_layout()
    path2 = out_dir / "shap_bar.png"
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[shap] Saved: {path2}")


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Residual KDE and parity scatter plots.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth values.
    y_pred : np.ndarray
        Model predictions.
    out_dir : Path
        Directory to save figures.
    """
    residuals = y_true.values - y_pred

    # KDE
    fig, ax = plt.subplots()
    sns.kdeplot(residuals, fill=True, ax=ax, color="steelblue")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Residual (scaled price)")
    ax.set_title("LightGBM — Residual Distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "residual_kde.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Parity
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.3, s=8)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title("LightGBM — Predicted vs Actual")
    fig.tight_layout()
    fig.savefig(out_dir / "parity_lightgbm.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Residual and parity plots saved.")


def plot_feature_importance(pipe, feature_names: list, out_dir: Path) -> None:
    """
    LightGBM native feature importance (gain).

    Parameters
    ----------
    pipe : Pipeline
        Trained sklearn pipeline containing LightGBM model.
    feature_names : list
        Original feature names.
    out_dir : Path
        Directory to save figure.
    """
    model = pipe.named_steps["model"]
    imp   = pd.Series(model.feature_importances_, index=feature_names)
    top15 = imp.nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(7, 5))
    top15.plot(kind="barh", ax=ax, color="seagreen")
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("LightGBM — Top 15 Features by Gain")
    fig.tight_layout()
    path = out_dir / "feature_importance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {path}")


def main(model_path: str, data_path: str, out_dir: str) -> None:
    """
    Run full evaluation suite: metrics + SHAP + diagnostics.

    Parameters
    ----------
    model_path : str
        Path to saved joblib pipeline.
    data_path : str
        Path to clean Airbnb CSV.
    out_dir : str
        Directory to save all output figures.
    """
    fig_path = Path(out_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    pipe, X_test, y_test, feature_names = load_model_and_data(model_path, data_path)
    y_pred = pipe.predict(X_test)

    print_metrics(y_test, y_pred)
    plot_shap_beeswarm(pipe, X_test, feature_names, fig_path)
    plot_residuals(y_test, y_pred, fig_path)
    plot_feature_importance(pipe, feature_names, fig_path)

    print(f"\n[done] All figures saved to: {fig_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Airbnb price model")
    parser.add_argument("--model", type=str, default="models/lgbm_pipeline.pkl")
    parser.add_argument("--data",  type=str, default="data/Airbnb_clean.csv")
    parser.add_argument("--out",   type=str, default="figures/")
    args = parser.parse_args()

    main(args.model, args.data, args.out)
