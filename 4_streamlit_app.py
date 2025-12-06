# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Nifty ETFs Dashboard", page_icon="📊", layout="wide")

# -------------------------
# Helper: normalize column names to canonical keys
# -------------------------
def normalize_columns(df):
    # Create a simple mapping of canonical names -> possible variants
    candidates = {
        "amc": ["amc", "AMC", "fund_house", "asset_manager"],
        "etf": ["etf", "fund", "etf_name", "etf name"],
        "category": ["category", "etf_category", "sectoral", "type"],
        "benchmark_index": ["benchmark_index", "benchmark index", "benchmark", "index"],
        "website_link": ["website_link", "website", "url", "link"],
        "nse_ticker": ["nse_ticker", "nse ticker", "nse_ticker_symbol", "nse"],
        "bse_ticker": ["bse_ticker", "bse ticker", "bse"],
        "isin_code": ["isin_code", "isin code", "isin"],
        "date_of_inception": ["date_of_inception", "date of inception", "inception_date", "date_inception"],
        "aum_in_cores": ["aum_in_cores", "aum(_in_cores)", "aum", "aum_in_cr", "aum_in_crores", "aum_in_cores"],
        "expense_ratio": ["expense_ratio", "expense ratio", "expense"],
        "overall_tracking_error": ["overall_tracking_error", "tracking_error", "tracking error"],
        "overall_tracking_difference": ["overall_tracking_difference", "tracking_difference", "tracking difference"],
        "years_in_operation": ["years_in_operation", "years", "years in operation", "age"]
    }

    # standardize df columns (strip, lower, replace spaces/hyphens)
    clean_cols = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns}
    df = df.rename(columns=clean_cols)

    # build reverse index to map existing col names to canonical
    colset = set(df.columns)
    mapping = {}
    for canon, variants in candidates.items():
        for v in variants:
            v_norm = v.strip().lower().replace(" ", "_").replace("-", "_")
            if v_norm in colset:
                mapping[v_norm] = canon
                break
    # apply mapping
    rename_map = {}
    for existing, canon in mapping.items():
        if existing != canon:
            rename_map[existing] = canon
    if rename_map:
        df = df.rename(columns=rename_map)

    return df

# -------------------------
# Load & clean data
# -------------------------
@st.cache_data
def load_data(path="etf_master.csv"):
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = normalize_columns(df)
    return df

# try loading
try:
    df = load_data("etf_master.csv")
except FileNotFoundError:
    st.error("etf_master.csv not found. Please upload the file to the same folder as this app and restart.")
    st.stop()

# Ensure required canonical columns exist (provide friendly defaults if missing)
required = ["etf", "amc", "category", "benchmark_index"]
for r in required:
    if r not in df.columns:
        df[r] = np.nan

# find aum column (prefer canonical name)
if "aum_in_cores" in df.columns:
    aum_col = "aum_in_cores"
else:
    # fallback: first column name containing 'aum'
    aum_candidates = [c for c in df.columns if "aum" in c]
    aum_col = aum_candidates[0] if aum_candidates else None

# Normalize AUM numeric if found
if aum_col:
    df[aum_col] = df[aum_col].astype(str).str.replace(",", "").str.replace("\u2009", "").str.replace(" ", "")
    df[aum_col] = pd.to_numeric(df[aum_col], errors="coerce")
else:
    df["aum_in_cores"] = np.nan
    aum_col = "aum_in_cores"

# normalize common numeric columns
for col in ["expense_ratio", "overall_tracking_error", "overall_tracking_difference", "years_in_operation"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------
# CSS styling (attractive)
# -------------------------
st.markdown(
    """
    <style>
    body { background-color: #f4f6f8; }
    .main-header { text-align:center; padding:20px; background: linear-gradient(90deg,#0f4c75,#3282b8); color:white; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.15); }
    .metric-card { background: rgba(255,255,255,0.9); padding:14px; border-radius:10px; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.06); }
    .metric-value { font-size:22px; font-weight:700; color:#145DA0; }
    .section-title { color:#0f4c75; font-weight:700; margin-bottom:8px; }
    .footer { text-align:center; color:#6c757d; margin-top:30px; margin-bottom:20px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Header & KPIs
# -------------------------
st.markdown('<div class="main-header"><h1>📊 ETFs Dashboard</h1><div style="font-size:14px">Broader • Sectoral • Thematic</div></div>', unsafe_allow_html=True)
st.write("")

total_etfs = len(df)
total_amcs = int(df["amc"].nunique()) if "amc" in df.columns else 0
total_aum = int(df[aum_col].sum()) if aum_col else 0
avg_exp = round(df["expense_ratio"].mean(), 2) if "expense_ratio" in df.columns else None

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="metric-card"><div style="color:#6c757d">Total ETFs</div><div class="metric-value">{total_etfs}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="metric-card"><div style="color:#6c757d">Total AMCs</div><div class="metric-value">{total_amcs}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="metric-card"><div style="color:#6c757d">Avg Expense Ratio (%)</div><div class="metric-value">{avg_exp if avg_exp is not None else "N/A"}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="metric-card"><div style="color:#6c757d">Total AUM (Cr.)</div><div class="metric-value">{total_aum:,}</div></div>', unsafe_allow_html=True)

st.write("")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Explorer", "📑 Details", "📊 Charts"])

# -------------------------
# Explorer Tab
# -------------------------
with tab1:
    st.subheader("Explore ETFs")
    # Filters - use existing column names if available, else blank lists
    amc_list = ["All"] + sorted(df["amc"].dropna().unique().tolist()) if "amc" in df.columns else ["All"]
    cat_list = ["All"] + sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else ["All"]
    bench_list = ["All"] + sorted(df["benchmark_index"].dropna().unique().tolist()) if "benchmark_index" in df.columns else ["All"]

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        chosen_amc = st.selectbox("AMC", amc_list)
    with c2:
        chosen_cat = st.selectbox("Category", cat_list)
    with c3:
        chosen_bench = st.selectbox("Benchmark", bench_list)

    search = st.text_input("Search ETF name or ticker")

    filtered = df.copy()
    if chosen_amc != "All":
        filtered = filtered[filtered["amc"] == chosen_amc]
    if chosen_cat != "All":
        filtered = filtered[filtered["category"] == chosen_cat]
    if chosen_bench != "All":
        filtered = filtered[filtered["benchmark_index"] == chosen_bench]
    if search:
        # search ETF and NSE/BSE tickers if present
        mask = filtered["etf"].str.contains(search, case=False, na=False)
        for ticker_col in ["nse_ticker", "bse_ticker"]:
            if ticker_col in filtered.columns:
                mask = mask | filtered[ticker_col].astype(str).str.contains(search, case=False, na=False)
        filtered = filtered[mask]

    st.markdown(f"**Showing {len(filtered)} ETFs**")
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

# -------------------------
# Details Tab
# -------------------------
with tab2:
    st.subheader("ETF Details")
    etf_options = [""] + sorted(df["etf"].dropna().unique().tolist())
    chosen = st.selectbox("Choose ETF", etf_options)
    if chosen:
        r = df[df["etf"] == chosen].iloc[0]
        st.markdown(f"### {r.get('etf','-')}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**AMC:**", r.get("amc", "-"))
            st.write("**Category:**", r.get("category", "-"))
            st.write("**Benchmark:**", r.get("benchmark_index", "-"))
            st.write("**Inception:**", r.get("date_of_inception", "-"))
        with c2:
            st.write("**ISIN:**", r.get("isin_code", "-"))
            st.write("**NSE:**", r.get("nse_ticker", "-"))
            st.write("**BSE:**", r.get("bse_ticker", "-"))
        with c3:
            st.write("**AUM (Cr):**", r.get(aum_col, "-"))
            st.write("**Expense Ratio:**", r.get("expense_ratio", "-"))
            st.write("**Tracking Error:**", r.get("overall_tracking_error", "-"))
            st.write("**Tracking Diff:**", r.get("overall_tracking_difference", "-"))
        if pd.notna(r.get("website_link", "")):
            st.markdown(f"[Visit Website]({r.get('website_link')})")

# -------------------------
# Charts Tab
# -------------------------
with tab3:
    st.subheader("Summary Charts")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**AUM by AMC (Top 10)**")
        top = df.groupby("amc")[aum_col].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top)
    with cB:
        st.markdown("**ETFs by Category**")
        if "category" in df.columns:
            st.bar_chart(df["category"].value_counts())
        else:
            st.info("No category column found.")

# -------------------------
# Footer
# -------------------------
st.markdown('<div class="footer">Developed & Conceptualized by <b>Diganta Raychaudhuri</b></div>', unsafe_allow_html=True)

