"""
visualise.py
============
Airbnb London 2024 — 10-figure visual analytics suite.

Produces publication-quality figures (300 DPI PNG + PDF):
    Fig 01 — Price distribution (log scale)
    Fig 02 — Price by room type (boxplot)
    Fig 03 — Median availability by month
    Fig 04 — Hex-bin map of average price
    Fig 05 — Spatial KDE of listing density
    Fig 06 — Capacity vs price by room type
    Fig 07 — Top-20 host market share (Pareto)
    Fig 08 — Correlation matrix
    Fig 09 — Review activity heatmap
    Fig 10 — Amenities word cloud

Usage:
    python src/visualise.py --data data/Airbnb_clean.csv --out figures/
"""

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.dpi": 300, "figure.figsize": (7, 4),
                     "font.family": "DejaVu Sans"})
sns.set_style("whitegrid")

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


def save(fig: plt.Figure, name: str, out_dir: Path) -> None:
    """Save figure as both PNG and PDF at 300 DPI."""
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf",            bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved: {name}.png")


def load_data(data_path: str | Path) -> pd.DataFrame:
    """
    Load clean Airbnb CSV and apply leakage fix.

    Parameters
    ----------
    data_path : str | Path
        Path to Airbnb_clean.csv

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(data_path)
    print(f"[data] Loaded: {df.shape}")

    # Remove any price-derived leakage columns
    leak_cols = [c for c in df.columns if "price" in c.lower() and c != "price"]
    if leak_cols:
        print(f"[data] Removing leak cols: {leak_cols}")
        df = df.drop(columns=leak_cols)

    # Rebuild room_type from one-hots if missing
    if "room_type" not in df.columns or df["room_type"].isna().all():
        rt_cols = [c for c in df.columns if c.startswith("room_type_")]
        if rt_cols:
            df["room_type"] = (df[rt_cols]
                               .idxmax(axis=1)
                               .str.replace("room_type_", "", regex=False))

    # Keep room types with >=50 listings
    rt_valid = df["room_type"].value_counts()[lambda s: s >= 50].index
    df.loc[~df["room_type"].isin(rt_valid), "room_type"] = "Other"

    # Month from scrape date
    if "last_scraped" in df.columns:
        df["month"] = pd.to_datetime(df["last_scraped"]).dt.month_name()

    return df


def fig01_price_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Fig 01 — Price distribution on log scale."""
    fig, ax = plt.subplots()
    sns.histplot(df["price"], bins=60, kde=True, color="steelblue", ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("Nightly price (log scale)")
    ax.set_title("Fig 01 — Price Distribution (log scale) | n=95,144 listings")
    save(fig, "fig01_price_distribution", out_dir)


def fig02_price_by_room_type(df: pd.DataFrame, out_dir: Path) -> None:
    """Fig 02 — Boxplot of price by room type."""
    fig, ax = plt.subplots(figsize=(8, 4))
    order = df["room_type"].value_counts().index
    sns.boxplot(data=df, x="price", y="room_type", order=order,
                showfliers=False, ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("Price (log scale)")
    ax.set_title("Fig 02 — Price Distribution by Room Type")
    save(fig, "fig02_price_by_room_type", out_dir)


def fig03_availability_by_month(df: pd.DataFrame, out_dir: Path) -> None:
    """Fig 03 — Median availability_365 by month."""
    if "month" not in df.columns:
        print("[fig03] Skipped — no month column")
        return
    med = df.groupby("month")["availability_365"].median().reindex(MONTHS)
    fig, ax = plt.subplots()
    sns.barplot(x=med.index, y=med.values, color="steelblue", ax=ax)
    ax.set_ylabel("Median availability (days/year)")
    ax.set_title("Fig 03 — Median Availability by Month")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "fig03_availability_by_month", out_dir)


def fig04_hexbin_price_map(df: pd.DataFrame, out_dir: Path) -> None:
    """Fig 04 — Hex-bin map of average nightly price across London."""
    if "latitude" not in df.columns:
        print("[fig04] Skipped — no lat/lon columns")
        return
    gdf = gpd.GeoDataFrame(df,
          geometry=gpd.points_from_xy(df.longitude, df.latitude),
          crs="EPSG:4326").to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(gdf.geometry.x, gdf.geometry.y, C=df["price"],
                   gridsize=80, reduce_C_function=np.mean,
                   cmap="plasma", mincnt=4)
    try:
        cx.add_basemap(ax, crs=gdf.crs)
    except Exception:
        pass  # basemap optional — network may not be available
    ax.set_axis_off()
    fig.colorbar(hb, ax=ax).set_label("Mean nightly price")
    ax.set_title("Fig 04 — Hex-bin Map of Average Price Across London")
    save(fig, "fig04_hexbin_price_map", out_dir)


def fig05_spatial_kde(df: pd.DataFrame, out_dir: Path) -> None:
    """Fig 05 — Spatial KDE of listing density."""
    if "latitude" not in df.columns:
        print("[fig05] Skipped — no lat/lon columns")
        return
    gdf = gpd.GeoDataFrame(df,
          geometry=gpd.points_from_xy(df.longitude, df.latitude),
          crs="EPSG:4326").cx[-0.55:0.3, 51.25:51.70]

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.kdeplot(x=gdf.geometry.x, y=gdf.geometry.y,
                fill=True, thresh=0.05, levels=60,
                cmap="viridis", alpha=0.7, bw_adjust=0.4, ax=ax)
    try:
        cx.add_basemap(ax, crs="EPSG:4326",
                       source=cx.providers.CartoDB.Positron)
    except Exception:
        pass
    ax.set_axis_off()
    ax.set_title("Fig 05 — Spatial KDE — Listing Density Across London")
    save(fig, "fig05_spatial_kde", out_dir)


def fig06_capacity_vs_price(df: pd.DataFrame, out_dir: Path) -> None:
    """Fig 06 — Scatter of capacity vs price coloured by room type."""
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="accommodates", y="price",
                    hue="room_type", alpha=0.25, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Accommodates (guests)")
    ax.set_ylabel("Price (log scale)")
    ax.set_title("Fig 06 — Capacity vs Price by Room Type")
    save(fig, "fig06_capacity_vs_price", out_dir)


def fig07_host_pareto(df: pd.DataFrame, out_dir: Path) -> None:
    """Fig 07 — Pareto chart of top-20 host market share."""
    if "host_id" not in df.columns:
        print("[fig07] Skipped — no host_id column")
        return
    host_cnt = df["host_id"].value_counts()
    top20    = host_cnt.head(20)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=top20.values, y=[f"H{h}" for h in top20.index],
                color="steelblue", ax=ax)
    cum  = top20.cumsum() / host_cnt.sum() * 100
    ax2  = ax.twiny()
    ax2.plot(cum.values, range(len(cum)), c="crimson", marker="o", ms=4)
    ax.set_xlabel("Number of listings")
    ax2.set_xlabel("Cumulative market share (%)")
    ax.set_title("Fig 07 — Top-20 Host Market Share (Pareto)")
    save(fig, "fig07_host_pareto", out_dir)


def fig08_correlation_matrix(df: pd.DataFrame, out_dir: Path) -> None:
    """Fig 08 — Correlation heatmap of key numeric features."""
    num_cols = ["price", "accommodates", "bathrooms", "review_scores_rating",
                "availability_365", "number_of_reviews"]
    present  = [c for c in num_cols if c in df.columns]
    fig, ax  = plt.subplots(figsize=(7, 5))
    sns.heatmap(df[present].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Fig 08 — Correlation Matrix (Key Numeric Features)")
    save(fig, "fig08_correlation_matrix", out_dir)


def fig09_review_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Fig 09 — Review activity heatmap (month × day-of-week)."""
    if "last_review" not in df.columns:
        print("[fig09] Skipped — no last_review column")
        return
    df["review_date"] = pd.to_datetime(df["last_review"], errors="coerce")
    if df["review_date"].notna().sum() < 100:
        print("[fig09] Skipped — insufficient review dates")
        return

    cal = (df.dropna(subset=["review_date"])
             .groupby([df.review_date.dt.month_name(),
                       df.review_date.dt.day_of_week])
             .size()
             .unstack(fill_value=0)
             .reindex(index=MONTHS,
                      columns=["Monday", "Tuesday", "Wednesday",
                                "Thursday", "Friday", "Saturday", "Sunday"]))
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(cal, cmap="YlOrRd", ax=ax)
    ax.set_title("Fig 09 — Review Activity Heatmap (Month × Day-of-Week)")
    save(fig, "fig09_review_heatmap", out_dir)


def fig10_amenities_wordcloud(df: pd.DataFrame, out_dir: Path) -> None:
    """Fig 10 — Word cloud of most common amenities."""
    if "amenities" not in df.columns:
        print("[fig10] Skipped — no amenities column")
        return
    amenities = (df["amenities"].dropna()
                 .str.lower()
                 .str.replace(r'[{}\"]', "", regex=True))
    if amenities.str.len().sum() == 0:
        print("[fig10] Skipped — no amenities text")
        return

    text = " ".join([" ".join(a.split(",")) for a in amenities])
    wc   = WordCloud(width=900, height=400, background_color="white",
                     collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Fig 10 — Most Common Airbnb Amenities (Word Cloud)")
    save(fig, "fig10_amenities_wordcloud", out_dir)


def main(data_path: str, out_dir: str) -> None:
    """
    Generate all 10 figures.

    Parameters
    ----------
    data_path : str
        Path to Airbnb_clean.csv
    out_dir : str
        Directory to save figures.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)

    fig01_price_distribution(df, out)
    fig02_price_by_room_type(df, out)
    fig03_availability_by_month(df, out)
    fig04_hexbin_price_map(df, out)
    fig05_spatial_kde(df, out)
    fig06_capacity_vs_price(df, out)
    fig07_host_pareto(df, out)
    fig08_correlation_matrix(df, out)
    fig09_review_heatmap(df, out)
    fig10_amenities_wordcloud(df, out)

    print(f"\n[done] All figures saved to: {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Airbnb visual analytics figures")
    parser.add_argument("--data", type=str, default="data/Airbnb_clean.csv")
    parser.add_argument("--out",  type=str, default="figures/")
    args = parser.parse_args()

    main(args.data, args.out)
