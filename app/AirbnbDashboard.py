"""
AirbnbDashboard.py
==================
Airbnb London 2024 — Interactive Streamlit Dashboard.

Features:
    - Upload any Airbnb CSV (raw or clean)
    - Filter by borough, month, and room type
    - Live price distribution, boxplot, availability, and scatter charts
    - Price prediction using trained LightGBM model
    - PDF export of all charts

Usage:
    streamlit run app/AirbnbDashboard.py
"""

import datetime
import io
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airbnb London Price Analytics",
    page_icon="🏠",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH   = Path("models/lgbm_pipeline.pkl")
PRICE_MEAN   = 120.0
PRICE_STD    = 85.0
MONTHS_ORDER = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(csv_file) -> pd.DataFrame:
    """Load and clean uploaded CSV, removing price leakage columns."""
    df = pd.read_csv(csv_file)
    leak = [c for c in df.columns if "price" in c.lower() and c != "price"]
    df   = df.drop(columns=leak, errors="ignore")

    # Normalise price to numeric — handles both raw ("$85.00") and
    # pre-cleaned CSVs. Without this, an upload of the raw file leaves
    # price as a string column and crashes any .median()/.mean() call.
    if "price" in df.columns and not pd.api.types.is_numeric_dtype(df["price"]):
        df["price"] = pd.to_numeric(
            df["price"].astype(str).str.replace(r"[\$,]", "", regex=True),
            errors="coerce",
        )

    if "last_scraped" in df.columns:
        df["month"] = pd.to_datetime(df["last_scraped"]).dt.month_name()
    return df


@st.cache_resource
def load_model():
    """Load trained LightGBM pipeline if available."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_Bélo.svg/320px-Airbnb_Logo_Bélo.svg.png",
                 width=120)
st.sidebar.title("Airbnb London Analytics")
st.sidebar.markdown("---")

DATA_FILE = st.sidebar.file_uploader(
    "Upload Airbnb CSV",
    type="csv",
    help="Upload your Airbnb_clean.csv or raw Airbnb_london.csv"
)

if DATA_FILE is None:
    st.title("🏠 Airbnb London Price Analytics Dashboard")
    st.info("Upload your Airbnb CSV file in the sidebar to get started.")
    st.markdown("""
    **This dashboard provides:**
    - Price distribution analysis across 95,000+ London listings
    - Borough and room-type breakdowns
    - Geospatial availability patterns
    - ML-powered price prediction (LightGBM, R²=0.67)

    **Dataset:** Inside Airbnb — London 2024 (95,144 listings, 75 features)
    """)
    st.stop()

df  = load_data(DATA_FILE)
mdl = load_model()

# Filters
boros  = st.sidebar.multiselect(
    "Borough", sorted(df["neighbourhood_cleansed"].dropna().unique())
    if "neighbourhood_cleansed" in df.columns else []
)
months = st.sidebar.multiselect(
    "Month", sorted(df["month"].unique())
    if "month" in df.columns else []
)
rtypes = st.sidebar.multiselect(
    "Room type", sorted(df["room_type"].unique())
    if "room_type" in df.columns else []
)

# Apply filters
sub = df.copy()
if boros:  sub = sub[sub["neighbourhood_cleansed"].isin(boros)]
if months: sub = sub[sub["month"].isin(months)]
if rtypes: sub = sub[sub["room_type"].isin(rtypes)]

st.sidebar.markdown(f"**Listings shown:** {len(sub):,}")

# ── Main dashboard ────────────────────────────────────────────────────────────
st.title("🏠 Airbnb London — Price Analytics Dashboard")
st.caption(f"Dataset: {len(df):,} listings · Filtered: {len(sub):,} listings")

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Median Price",       f"£{sub['price'].median():.0f}")
k2.metric("Mean Price",         f"£{sub['price'].mean():.0f}")
k3.metric("Total Listings",     f"{len(sub):,}")
k4.metric("Unique Boroughs",
          str(sub["neighbourhood_cleansed"].nunique())
          if "neighbourhood_cleansed" in sub.columns else "N/A")

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(sub["price"], bins=50, kde=True, ax=ax1, color="#4C72B0")
    ax1.set_xscale("log")
    ax1.set_xlabel("Nightly price (log scale)")
    ax1.set_title("Price Distribution")
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    st.subheader("Price by Room Type")
    if "room_type" in sub.columns:
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=sub, x="price", y="room_type",
                    order=sub["room_type"].value_counts().index,
                    showfliers=False, ax=ax2, color="#55A868")
        ax2.set_xscale("log")
        ax2.set_title("Price by Room Type")
        st.pyplot(fig2)
        plt.close(fig2)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Availability by Month")
    if "month" in sub.columns:
        med = (sub.groupby("month")["availability_365"]
               .median().reindex(MONTHS_ORDER))
        fig3, ax3 = plt.subplots()
        sns.barplot(x=med.index, y=med.values, color="#8172B3", ax=ax3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        ax3.set_ylabel("Median availability (days)")
        ax3.set_title("Availability by Month")
        st.pyplot(fig3)
        plt.close(fig3)

with col4:
    st.subheader("Capacity vs Price")
    if "accommodates" in sub.columns:
        fig4, ax4 = plt.subplots()
        hue_col = "room_type" if "room_type" in sub.columns else None
        sns.scatterplot(data=sub.sample(min(2000, len(sub))),
                        x="accommodates", y="price",
                        hue=hue_col, alpha=0.4, ax=ax4)
        ax4.set_yscale("log")
        ax4.set_title("Capacity vs Price")
        st.pyplot(fig4)
        plt.close(fig4)

# ── Price Predictor ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🤖 Price Predictor — LightGBM Model")

if mdl is None:
    st.warning("Trained model not found. Run `python src/train.py` to train and save the model.")
else:
    st.caption("R² = 0.67 | MAE = £25 | RMSE = £49 on held-out test set (95,144 listings)")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        accommodates = st.slider("Guests", 1, 16, 2)
        bedrooms     = st.slider("Bedrooms", 0, 10, 1)
    with pc2:
        bathrooms    = st.slider("Bathrooms", 0.5, 5.0, 1.0, 0.5)
        min_nights   = st.slider("Minimum nights", 1, 30, 1)
    with pc3:
        room_type    = st.selectbox("Room type",
                                    ["Entire home/apt", "Private room",
                                     "Shared room", "Hotel room"])
        avail_365    = st.slider("Availability (days/year)", 0, 365, 180)

    if st.button("Predict Price"):
        # Build a minimal input row matching training features
        input_row = pd.DataFrame([{
            "accommodates":   accommodates,
            "bedrooms":       bedrooms,
            "bathrooms":      bathrooms,
            "minimum_nights": min_nights,
            "availability_365": avail_365,
            "days_since_review": 100,
            "no_review_flag": 0,
            "review_scores_rating": 4.7,
            "number_of_reviews": 10,
            "reviews_per_month": 1.0,
        }])

        try:
            # Align columns with training data
            train_cols = mdl.named_steps["prep"].feature_names_in_ \
                         if hasattr(mdl.named_steps["prep"], "feature_names_in_") \
                         else input_row.columns
            for col in train_cols:
                if col not in input_row.columns:
                    input_row[col] = 0
            input_row = input_row[[c for c in train_cols if c in input_row.columns]]

            pred_scaled = mdl.predict(input_row)[0]
            pred_gbp    = pred_scaled * PRICE_STD + PRICE_MEAN
            st.success(f"Estimated nightly price: **£{pred_gbp:.0f}**")
            st.caption("Based on LightGBM model trained on 95,144 London Airbnb listings.")
        except Exception as e:
            st.error(f"Prediction failed: {e}. Ensure model was trained on compatible features.")

# ── PDF Export ────────────────────────────────────────────────────────────────
st.markdown("---")

def make_pdf(figures: list) -> bytes:
    """Export dashboard charts to a multi-page PDF."""
    c_pdf = canvas.Canvas("brief.pdf", pagesize=A4)
    w, h  = A4
    for png_path in figures:
        c_pdf.drawImage(png_path, 30, 180, width=w - 60,
                        height=h - 260, preserveAspectRatio=True)
        c_pdf.showPage()
    c_pdf.save()
    with open("brief.pdf", "rb") as f:
        return f.read()


if st.button("📄 Download PDF Report"):
    figs  = []
    bufs  = []

    for name, color in [("Price dist", "#4C72B0"), ("Availability", "#8172B3")]:
        fig, ax = plt.subplots()
        if name == "Price dist":
            sns.histplot(sub["price"], bins=50, kde=True, ax=ax, color=color)
            ax.set_xscale("log")
        else:
            if "month" in sub.columns:
                med = sub.groupby("month")["availability_365"].median().reindex(MONTHS_ORDER)
                sns.barplot(x=med.index, y=med.values, color=color, ax=ax)
                ax.tick_params(axis="x", rotation=45)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        bufs.append(tmp.name)
        plt.close(fig)

    pdf_bytes = make_pdf(bufs)
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "⬇️ Save PDF Report",
        data=pdf_bytes,
        file_name=f"airbnb_london_report_{ts}.pdf",
        mime="application/pdf",
    )

st.markdown("---")
st.caption("Built by Farhan Hashmi · MSc Data Science & AI, Keele University · 2025 · "
           "[GitHub](https://github.com/Shaunfarhan16/Large-Scale-Airbnb-Price-Prediction-Spatial-Analytics-Platform)")
