import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime as dt

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="Bharat ETF Dashboard", page_icon="📊", layout="wide")

# -------------------------------------------------------------
# TOP SCROLLING MESSAGE (Comic Sans + Scrolling)
# -------------------------------------------------------------
st.markdown("""
<style>
@keyframes scroll-left {
  0% { transform: translateX(100%); }
  100% { transform: translateX(-100%); }
}
.scrolling-text {
  width: 100%;
  overflow: hidden;
  white-space: nowrap;
  color: red;
  font-family: "Comic Sans MS";
  font-size: 30px;
  font-weight: bold;
  animation: scroll-left 12s linear infinite;
  padding: 10px;
}
</style>

<div class="scrolling-text">!! WORK IN PROGRESS — THANK YOU FOR YOUR PATIENCE !!</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# LOAD & CLEAN DATA
# -------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("etf_master.csv")

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Clean strings
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    return df

df = load_data()

# -------------------------------------------------------------
# STANDARDIZE AMC COLUMN
# -------------------------------------------------------------
if "amc" not in df.columns:
    for col in df.columns:
        if "amc" in col or "issuer" in col or "fund_house" in col:
            df = df.rename(columns={col: "amc"})
            break

df["amc"] = df["amc"].astype(str).str.strip()

# -------------------------------------------------------------
# CLEAN AUM COLUMN
# -------------------------------------------------------------
aum_col = "aum_in_cores" if "aum_in_cores" in df.columns else "aum(_in_cores)"

df[aum_col] = (
    df[aum_col]
    .astype(str)
    .str.replace(",", "")
    .str.replace(" ", "")
    .str.replace("\u2009", "")
)

df[aum_col] = pd.to_numeric(df[aum_col], errors="coerce")

# -------------------------------------------------------------
# DATE & FUND AGE CALCULATION
# -------------------------------------------------------------
df["date_of_inception"] = pd.to_datetime(df["date_of_inception"], errors="coerce")
today = dt.datetime.today()

df["fund_age_days"] = (today - df["date_of_inception"]).dt.days
df["fund_age_months"] = df["fund_age_days"] / 30.44
df["fund_age_years"] = df["fund_age_days"] / 365.25

df["fund_age_years"] = df["fund_age_years"].round(2)   # <-- ROUND TO 2 DECIMALS

df["overall_tracking_error"] = pd.to_numeric(df["overall_tracking_error"], errors="coerce")

# -------------------------------------------------------------
# CSS STYLING
# -------------------------------------------------------------
st.markdown("""
<style>
    body { background: linear-gradient(180deg, #f7fbff 0%, #ffffff 40%); }

    .main-header { 
        background: linear-gradient(90deg, #0E4C92, #1A73E8);
        color: white !important;
        padding: 25px;
        border-radius: 12px;
        text-align:center;
        margin-bottom: 25px;
    }

    .metric-card {
        background: rgba(255,255,255,0.95);
        padding: 16px;
        border-radius: 12px;
        color: #000 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #145DA0 !important;
    }

    .details-card {
        background: #FFF8DC; /* light yellow */
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .footer {
        text-align:center;
        color:#555;
        margin-top:40px;
        font-size:15px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# HEADER (Updated Text)
# -------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1 style="font-size:36px; font-weight:600;">📊 Bharat ETF Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# FILTER INSTRUCTION MESSAGE
# -------------------------------------------------------------
st.markdown("### 👉 Choose your ETF filters")

# -------------------------------------------------------------
# FILTER BAR
# -------------------------------------------------------------
colA, colB, colC = st.columns([1.2, 1.2, 1.2])

with colA:
    amc_list = ["All"] + sorted(df["amc"].dropna().unique())
    sel_amc = st.selectbox("AMC", amc_list)

with colB:
    cat_list = ["All"] + sorted(df["category"].dropna().unique())
    sel_cat = st.selectbox("Category", cat_list)

with colC:
    bench_list = ["All"] + sorted(df["benchmark_index"].dropna().unique())
    sel_bench = st.selectbox("Benchmark Index", bench_list)

search = st.text_input("Search ETF or Ticker")

# -------------------------------------------------------------
# APPLY FILTERS → filtered_df
# -------------------------------------------------------------
filtered_df = df.copy()

if sel_amc != "All":
    filtered_df = filtered_df[filtered_df["amc"] == sel_amc]

if sel_cat != "All":
    filtered_df = filtered_df[filtered_df["category"] == sel_cat]

if sel_bench != "All":
    filtered_df = filtered_df[filtered_df["benchmark_index"] == sel_bench]

if search:
    filtered_df = filtered_df[
        filtered_df["etf"].str.contains(search, case=False, na=False) |
        filtered_df["nse_ticker"].astype(str).str.contains(search, case=False, na=False)
    ]

# -------------------------------------------------------------
# DYNAMIC KPIs
# -------------------------------------------------------------
k1, k2 = st.columns(2)

filtered_count = len(filtered_df)
filtered_aum_total = filtered_df[aum_col].sum(skipna=True)
safe_aum = 0 if pd.isna(filtered_aum_total) else int(filtered_aum_total)

with k1:
    st.markdown(
        f'<div class="metric-card"><b>Number of ETFs</b><br>'
        f'<span class="metric-value">{filtered_count}</span></div>',
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f'<div class="metric-card"><b>Total AUM (Cr.)</b><br>'
        f'<span class="metric-value">{format(safe_aum,",")}</span></div>',
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------
# TABLE
# -------------------------------------------------------------
st.write(f"### Showing {filtered_count} ETFs:")
st.dataframe(filtered_df, use_container_width=True)

# -------------------------------------------------------------
# TABS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📑 ETF Details",
    "📊 Age vs AUM (Bar Chart)",
    "📉 Lowest Tracking Error",
    "🔥 Sectoral Heatmap"
])

# -------------------------------------------------------------
# TAB 1 — DETAILS
# -------------------------------------------------------------
with tab1:
    st.subheader("ETF Details")

    etf_list = [""] + sorted(filtered_df["etf"].dropna())
    chosen = st.selectbox("Select ETF", etf_list)

    if chosen:
        row = filtered_df[filtered_df["etf"] == chosen].iloc[0]

        st.markdown('<div class="details-card">', unsafe_allow_html=True)

        st.markdown("### ETF Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**AMC:**", row["amc"])
            st.write("**Category:**", row["category"])
            st.write("**Benchmark:**", row["benchmark_index"])

        with col2:
            st.write("**ISIN:**", row["isin_code"])
            st.write("**NSE Ticker:**", row["nse_ticker"])
            st.write("**BSE Ticker:**", row["bse_ticker"])

        with col3:
            st.write("**AUM (Cr):**", row[aum_col])
            st.write("**Expense Ratio:**", row.get("expense_ratio"))
            st.write("**Tracking Error:**", row["overall_tracking_error"])
            st.write("**Fund Age (Years):**", round(row["fund_age_years"], 2))

        if pd.notna(row["website_link"]):
            st.markdown("### For more information please visit:")
            st.markdown(f"[🌐 Website Link]({row['website_link']})")

        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2 — INTERACTIVE BAR CHART (Age vs AUM)
# -------------------------------------------------------------
with tab2:
    st.subheader("Interactive Bar Chart — ETF Age vs AUM")

    chart = alt.Chart(filtered_df).mark_bar(
        size=25,                   # BROADER BARS
        color="maroon",            # RED / MAROON COLOR
        opacity=0.85
    ).encode(
        x=alt.X("fund_age_years:Q", title="Fund Age (Years, rounded to 2 decimals)"),
        y=alt.Y(f"{aum_col}:Q", title="AUM (Crores)"),
        tooltip=["etf", "amc", "fund_age_years", aum_col]
    ).properties(
        width=950,
        height=500
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# -------------------------------------------------------------
# TAB 3 — LOWEST TRACKING ERROR
# -------------------------------------------------------------
with tab3:
    st.subheader("Lowest Tracking Error ETFs (Filtered)")

    filt = filtered_df.dropna(subset=["overall_tracking_error"]).copy()
    filt = filt.sort_values("overall_tracking_error").head(10)

    st.dataframe(filt, use_container_width=True)

# -------------------------------------------------------------
# TAB 4 — SECTORAL HEATMAP
# -------------------------------------------------------------
with tab4:
    st.subheader("Sectoral Heatmap – AUM by Category")

    heat = filtered_df.pivot_table(
        index="category",
        values=aum_col,
        aggfunc="sum"
    ).reset_index()

    heat_chart = alt.Chart(heat).mark_rect().encode(
        x=alt.X("category:N", title="ETF Category"),
        y=alt.Y("category:N", title=""),
        color=alt.Color(f"{aum_col}:Q", scale=alt.Scale(scheme='blues')),
        tooltip=["category", aum_col]
    ).properties(width=700, height=550)

    st.altair_chart(heat_chart, use_container_width=True)

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("""
<div class="footer">
Conceptualized by <b>Diganta Raychaudhuri</b>
</div>
""", unsafe_allow_html=True)

