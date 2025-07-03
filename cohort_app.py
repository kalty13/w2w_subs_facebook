import streamlit as st
import logging

# basic logging config
logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
st.set_page_config(page_title="Cohort Retention", layout="wide")

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ───────── helper for progress-bar in table ─────────
def bar(p, w=10):
    return "🟥"*int(round(p/10)) + "⬜"*(w-int(round(p/10)))

# ───────── load data ─────────
FILE = Path(__file__).parent / "subscriptions.tsv"
FB_FILE = Path(__file__).parent / "fb_export_it_all_time.csv"

ALLOWED_SOURCES = ["ig", "fb"]

# ───── sidebar debug switch ─────
debug = st.sidebar.checkbox("Debug mode")

@st.cache_data(show_spinner=False)
def load(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, sep="\t")


@st.cache_data(show_spinner=False)
def load_fb(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if "Day" in df.columns:
        df["Day"] = pd.to_datetime(df["Day"], errors="coerce")
    return df

df_raw = load(FILE)
fb_raw = load_fb(FB_FILE)
df_raw["created_at"] = pd.to_datetime(df_raw["created_at"])
df_raw = df_raw[df_raw["user_visit.utm_source"].isin(ALLOWED_SOURCES)]

if debug:
    logging.info("Subscriptions after IG/FB filter: %s rows", len(df_raw))
    st.write("Subscriptions (head)", df_raw.head())

# ─────────── UI filters ───────────
min_d, max_d = df_raw["created_at"].dt.date.agg(["min", "max"])
start, end   = st.date_input("Date range", [min_d, max_d], min_d, max_d)

# weekly checkbox ON by default
weekly = st.checkbox("Weekly cohorts", True)

utm_col       = "user_visit.utm_source"
campaign_col  = "user_visit.utm_campaign"
price_col     = "price.price_option_text"

fb_raw = fb_raw[fb_raw["Campaign name"].isin(df_raw[campaign_col].dropna().unique())]

if debug:
    logging.info("FB rows after campaign filter: %s", len(fb_raw))
    st.write("FB spend (head)", fb_raw.head())

sel_utm = st.multiselect(
    "UTM source",
    sorted(df_raw[utm_col].dropna().unique()),
    default=sorted(df_raw[utm_col].dropna().unique())
)

sel_campaign = st.multiselect(
    "UTM campaign",
    sorted(df_raw[campaign_col].dropna().unique()),
    default=sorted(df_raw[campaign_col].dropna().unique())
)

sel_price = st.multiselect(
    "Price option",
    sorted(df_raw[price_col].dropna().unique()),
    default=sorted(df_raw[price_col].dropna().unique())
)

# ─────────── filter dataframe ───────────
df = df_raw[
    (df_raw["real_payment"] == 1) &
    (df_raw["created_at"].dt.date.between(start, end)) &
    (df_raw[utm_col].isin(sel_utm)) &
    (df_raw[campaign_col].isin(sel_campaign)) &
    (df_raw[price_col].isin(sel_price))
].copy()

if debug:
    logging.info("Subs after UI filters: %s rows", len(df))

# ─────────── define cohort_date (daily / weekly) ───────────
df["cohort_date"] = (
    df["created_at"].dt.to_period("W").apply(lambda r: r.start_time.date())
    if weekly else
    df["created_at"].dt.date
)

# ─────────── expand rows into periods ───────────
exp = (
    df.loc[df.index.repeat(df["charges_count"].astype(int))]
      .assign(period=lambda d: d.groupby(level=0).cumcount())
)

size = exp[exp.period == 0].groupby("cohort_date").size()

if debug:
    st.write("Cohort sizes", size.head())

dead = (
    df[df["next_charge_date"].isna()]
      .groupby("cohort_date").size()
      .reindex(size.index, fill_value=0)
)
death_pct = (dead / size * 100).round(1)

revenue = (
    df.groupby("cohort_date")["send_event_amount"].sum()
      .reindex(size.index, fill_value=0).round(2)
)

if debug:
    st.write("Revenue per cohort", revenue.head())

# ─────────── FB spend per cohort ───────────
spend = (
    fb_raw.groupby(
        fb_raw["Day"].dt.to_period("W").apply(lambda r: r.start_time.date())
        if weekly
        else fb_raw["Day"].dt.date
    )["Amount spent (USD)"].sum()
      .reindex(size.index, fill_value=0)
)

if debug:
    st.write("Spend per cohort", spend.head())
ltv = (revenue / size).round(2)

pivot = exp.pivot_table(index="cohort_date", columns="period",
                        aggfunc="size", fill_value=0)
ret = pivot.div(size, axis=0).mul(100).round(1)
pivot.columns = ret.columns = [f"Period {p}" for p in pivot.columns]

# ─────────── build retention table ───────────
death_cell = (
     death_pct.map(lambda v: f"{v:.1f}%") + " "
    + death_pct.map(bar) + "<br>(" + dead.astype(str) + ")"
)

disp = pd.DataFrame(index=ret.index, columns=ret.columns)
for ix in ret.index:
    for col in ret.columns:
        if death_pct[ix] == 100 and ret.loc[ix, col] == 0:
            disp.loc[ix, col] = "💀"
        else:
            disp.loc[ix, col] = f"{ret.loc[ix, col]:.1f}%<br>({pivot.loc[ix, col]})"

combo = disp.copy()
combo.insert(0, "Cohort death", death_cell)
combo.insert(1, "Revenue USD", revenue.map(lambda v: f"${v:,.2f}"))
combo["LTV USD"] = ltv.map(lambda v: f"${v:,.2f}")

# TOTAL row
weighted = lambda s: (s * size).sum() / size.sum()
total = {
    "Cohort death": f"💀 {weighted(death_pct):.1f}% {bar(weighted(death_pct))}",
    "Revenue USD":  f"${revenue.sum():,.2f}",
    "LTV USD":      f"${weighted(ltv):,.2f}",
}
for col in ret.columns:
    total[col] = f"{weighted(ret[col]):.1f}%"
combo.loc["TOTAL"] = total
combo = pd.concat([combo.drop("TOTAL").sort_index(ascending=False), combo.loc[["TOTAL"]]])

# colour helpers
Y_R, Y_G, Y_B = 255, 212, 0
rgba = lambda a: f"rgba({Y_R},{Y_G},{Y_B},{a:.2f})"
txt  = lambda a: "black" if a > 0.5 else "white"
BASE = "#202020"; A0, A1 = .2, .8

header = ["Cohort"] + combo.columns.tolist()
rows, fills, fonts = [], [], []
for ix, row in combo.iterrows():
    rows.append([str(ix)] + row.tolist())
    if ix == "TOTAL":
        fills.append(["#444444"] * len(combo.columns))
        fonts.append(["white"] * len(combo.columns))
        continue
    c_row, f_row = ["#1e1e1e", "#333333", "#333333"], ["white"] * 3
    for p in ret.loc[ix].values / 100:
        if p == 0 or pd.isna(p):
            c_row.append(BASE); f_row.append("white")
        else:
            a = A0 + (A1 - A0) * p
            c_row.append(rgba(a)); f_row.append(txt(a))
    c_row.append("#333333"); f_row.append("white")
    fills.append(c_row); fonts.append(f_row)

fig_table = go.Figure(go.Table(
    header=dict(values=header, fill_color="#303030",
                font=dict(color="white", size=13), align="center"),
    cells=dict(values=list(map(list, zip(*rows))),
               fill_color=list(map(list, zip(*fills))),
               font=dict(size=13, color=list(map(list, zip(*fonts)))),
               align="center", height=34)
))
fig_table.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f")
st.title("Cohort Retention – real_payment = 1")
st.plotly_chart(fig_table, use_container_width=True)
