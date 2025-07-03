import streamlit as st
st.set_page_config(page_title="Cohort Retention", layout="wide")

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import logging
from st_aggrid import AgGrid, GridOptionsBuilder  # clickable table component

logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)

# ───────── helper for tiny progress‑bar ─────────
def bar(p: float, w: int = 10) -> str:
    return "🟥" * int(round(p / 10)) + "⬜" * (w - int(round(p / 10)))

# ───────── file paths ─────────
BASE_DIR = Path(__file__).parent
FILE_SUBS = BASE_DIR / "subscriptions.tsv"
FILE_FB = BASE_DIR / "fb_export_it_all_time.csv"

@st.cache_data(show_spinner=False)
def load_subs(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, sep="\t")

@st.cache_data(show_spinner=False)
def load_fb(p: Path) -> pd.DataFrame:
    fb = pd.read_csv(p)
    fb["Day"] = pd.to_datetime(fb["Day"])
    fb["campaign_clean"] = (
        fb["Campaign name"].astype(str).str.split(" (", n=1, regex=False).str[0]
    )
    fb["Amount spent (USD)"] = pd.to_numeric(fb["Amount spent (USD)"], errors="coerce").fillna(0)
    fb["Impressions"] = pd.to_numeric(fb["Impressions"], errors="coerce").fillna(0)
    fb["Link clicks"] = pd.to_numeric(fb["Link clicks"], errors="coerce").fillna(0)
    return fb

# ───────── raw data ─────────
df_raw = load_subs(FILE_SUBS)
fb_raw = load_fb(FILE_FB)

# Clean and filter
df_raw["created_at"] = pd.to_datetime(df_raw["created_at"])
df_raw = df_raw[df_raw["user_visit.utm_source"].isin(["ig", "fb"])]
df_raw["campaign_clean"] = (
    df_raw["user_visit.utm_campaign"].astype(str).str.split(" (", n=1, regex=False).str[0]
)
fb_raw = fb_raw[fb_raw["campaign_clean"].isin(df_raw["campaign_clean"].unique())]

# ───────── UI filters ─────────
min_d, max_d = df_raw["created_at"].dt.date.agg(["min", "max"])
start, end = st.date_input("Date range", [min_d, max_d], min_d, max_d)

weekly = st.checkbox("Weekly cohorts", True)

price_col = "price.price_option_text"
sel_price = st.multiselect(
    "Price option",
    sorted(df_raw[price_col].dropna().unique()),
    default=sorted(df_raw[price_col].dropna().unique()),
)

# ───────── clickable campaign table ─────────
campaign_stats = (
    df_raw.groupby("campaign_clean")["charges_count"].sum().reset_index(name="Purchases")
)
gb = GridOptionsBuilder.from_dataframe(campaign_stats)
gb.configure_selection("single", use_checkbox=True)
grid = AgGrid(
    campaign_stats,
    gridOptions=gb.build(),
    height=300,
    allow_unsafe_jscode=True,
    theme="alpine",
)
sel_rows = grid["selected_rows"]
selected_campaign = sel_rows[0]["campaign_clean"] if len(sel_rows) > 0 else None

if selected_campaign:
    st.success(f"Filtered by campaign: {selected_campaign}")
    fb_selected = fb_raw[fb_raw["campaign_clean"] == selected_campaign]
    df_selected = df_raw[df_raw["campaign_clean"] == selected_campaign]

    daily_purchases = df_selected[df_selected["real_payment"] == 1].groupby(df_selected["created_at"].dt.date).size()
    fb_selected = fb_selected.groupby("Day").agg({
        "Amount spent (USD)": "sum",
        "Impressions": "sum",
        "Link clicks": "sum"
    }).reset_index()
    fb_selected["Purchases"] = fb_selected["Day"].map(daily_purchases).fillna(0)
    fb_selected["CTR"] = (fb_selected["Link clicks"] / fb_selected["Impressions"] * 100).round(2)
    fb_selected["CPM"] = (fb_selected["Amount spent (USD)"] / fb_selected["Impressions"] * 1000).round(2)
    fb_selected["CPC"] = (fb_selected["Amount spent (USD)"] / fb_selected["Link clicks"]).round(2)
    fb_selected["Cost per purchase"] = (fb_selected["Amount spent (USD)"] / fb_selected["Purchases"].replace(0, pd.NA)).round(2)

    st.dataframe(
        fb_selected.rename(columns={
            "Day": "Date",
            "Amount spent (USD)": "Spend",
            "Impressions": "Impr.",
            "Link clicks": "Clicks"
        })[
            ["Date", "Spend", "Impr.", "Clicks", "CTR", "CPM", "CPC", "Purchases", "Cost per purchase"]
        ]
    )

# (rest of your cohort logic continues here)
