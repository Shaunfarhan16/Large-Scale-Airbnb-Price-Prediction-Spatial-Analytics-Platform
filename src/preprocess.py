"""
preprocess.py
=============
Airbnb London 2024 — preprocessing and feature engineering pipeline.

Fixes vs original:
- No hardcoded paths (accepts data_path argument)
- StandardScaler fitted on train set only (no leakage)
- Modular functions with docstrings and type hints
- Returns scaler separately for inverse-transform at inference time

Usage:
    from src.preprocess import load_and_clean, build_features, split_and_scale
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


# ── Constants ──────────────────────────────────────────────────────────────

DROP_COLS = [
    "name", "description", "neighborhood_overview", "picture_url",
    "host_url", "scrape_id", "thumbnail_url", "listing_url",
    "host_thumbnail_url", "host_picture_url", "source",
    "calendar_updated", "neighbourhood_group_cleansed", "license",
]

CAT_COLS = [
    "room_type", "neighbourhood_cleansed", "property_major",
    "host_is_superhost",
]

NUM_COLS = ["price", "accommodates", "review_scores_rating", "bathrooms"]

TOP_N_PROPERTY_TYPES = 10


# ── Pipeline functions ─────────────────────────────────────────────────────

def load_and_clean(data_path: str | Path) -> pd.DataFrame:
    """
    Load raw Airbnb CSV and apply initial cleaning steps.

    Parameters
    ----------
    data_path : str | Path
        Path to the raw Airbnb CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering.
    """
    df = pd.read_csv(data_path)
    print(f"[load] Raw shape: {df.shape}")

    # Drop heavy free-text and web-link columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Extract numeric bathrooms from text field (e.g. "1.5 baths")
    if "bathrooms_text" in df.columns:
        df["bathrooms"] = (
            df["bathrooms_text"]
            .str.extract(r"(\d+\.?\d*)")
            .astype(float)
        )
        df.drop(columns="bathrooms_text", inplace=True, errors="ignore")

    # Price: strip currency symbols and cast to float
    if df["price"].dtype == object:
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .replace("nan", np.nan)
            .astype(float)
        )

    # Drop rows with missing target
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    print(f"[load] After price clean: {df.shape}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from cleaned dataframe.

    Steps:
    - Impute missing review scores with median
    - Create no_review_flag for listings with no reviews
    - Extract days_since_review date feature
    - Clip outliers at 1st/99th percentile
    - Create price_per_person feature
    - Bucket rare property types into 'Other'
    - One-hot encode categoricals

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe from load_and_clean().

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe.
    """
    # Impute review scores
    review_cols = df.filter(like="review_scores").columns
    df[review_cols] = df[review_cols].apply(lambda c: c.fillna(c.median()))

    # Binary flag for listings with no reviews
    df["no_review_flag"] = df["last_review"].isna().astype(int)

    # Days since last review (recency feature)
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    df["days_since_review"] = (
        pd.Timestamp.today() - df["last_review"]
    ).dt.days.fillna(9999)

    # Clip outliers at 1st/99th percentile
    existing_nums = [c for c in NUM_COLS if c in df.columns]
    q_lo = df[existing_nums].quantile(0.01)
    q_hi = df[existing_nums].quantile(0.99)
    df[existing_nums] = df[existing_nums].clip(q_lo, q_hi, axis=1)

    # Price per person (interaction feature)
    df["price_per_person"] = df["price"] / df["accommodates"].replace(0, 1)

    # Bucket rare property types
    top_n = df["property_type"].value_counts().nlargest(TOP_N_PROPERTY_TYPES).index
    df["property_major"] = np.where(
        df["property_type"].isin(top_n), df["property_type"], "Other"
    )

    # One-hot encode categoricals
    cats_present = [c for c in CAT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cats_present, drop_first=True)

    print(f"[features] Shape after engineering: {df.shape}")
    return df


def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, list]:
    """
    Train/test split and scale numeric features.

    IMPORTANT: StandardScaler is fitted on X_train ONLY and applied
    to X_test via transform() — preventing any data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataframe.
    test_size : float
        Fraction of data for test set (default 0.20).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple of (X_train, X_test, y_train, y_test, scaler, scale_cols)
    """
    # Remove any remaining price-derived leakage columns
    leak_cols = [c for c in df.columns if "price" in c.lower() and c != "price"]
    if leak_cols:
        print(f"[split] Removing leak columns: {leak_cols}")
        df = df.drop(columns=leak_cols)

    y = df["price"]
    X = df.drop(columns="price")

    # Keep numeric/boolean only
    X = X.select_dtypes(include=["number", "bool"]).dropna(axis=1, how="all")

    # Split FIRST, then scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale: fit on train only, transform both
    scale_cols = [c for c in ["accommodates", "bathrooms",
                               "days_since_review", "price_per_person"]
                  if c in X_train.columns]

    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])   # transform only — no leakage

    print(f"[split] Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, scale_cols


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Airbnb preprocessing pipeline")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to raw Airbnb CSV file")
    parser.add_argument("--out", type=str, default="data/Airbnb_clean.csv",
                        help="Output path for clean CSV")
    args = parser.parse_args()

    df = load_and_clean(args.data)
    df = build_features(df)
    df.to_csv(args.out, index=False)
    print(f"[done] Clean CSV saved to: {args.out}")
