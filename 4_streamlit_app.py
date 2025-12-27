import streamlit as st
import pandas as pd
import os
import altair as alt
latest_price_date = None

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Indian ETF Tracker",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# TOP SCROLLING MESSAGE
# ============================================================
st.markdown("""
<style>
.scrolling-container {
    width: 100%;
    overflow: hidden;
    height: 45px;
    position: relative;
    background: linear-gradient(90deg, #FF6B6B 0%, #FFD93D 50%, #6BCF7F 100%);
    border-radius: 8px;
    margin-bottom: 20px;
}
.scrolling-text {
    position: absolute;
    white-space: nowrap;
    font-family: "Comic Sans MS";
    font-size: 28px;
    font-weight: bold;
    color: white;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    animation: scroll-left 15s linear infinite;
}
@keyframes scroll-left {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}
</style>

<div class="scrolling-container">
    <div class="scrolling-text">
        ✨ WORK IN PROGRESS - THANK YOU FOR YOUR PATIENCE ✨
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 35px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
">
    <h1 style="color: white; margin: 0; font-size: 42px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        📊 Indian ETF Tracker
    </h1>
</div>
""", unsafe_allow_html=True)

# Info message with enhanced styling
st.markdown("""
<div style="
    background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #FF6F00;
    margin-bottom: 25px;
    box-shadow: 0 3px 6px rgba(0,0,0,0.1);
">
    <p style="color: #E65100; margin: 0; font-weight: 700; font-size: 16px;">
        ℹ️ Detailed holdings available for selected representative ETFs only
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAUVE BACKGROUND + GLOBAL FONT
# ============================================================
st.markdown("""
<style>
/* Global app background */
.stApp {
    background-color: #E6E6FA;
    font-family: "Comic Sans MS", "Comic Sans", cursive;
}

/* Ensure all text elements inherit the font */
html, body, [class*="css"] {
    font-family: "Comic Sans MS", "Comic Sans", cursive !important;
}

/* Enhanced button styling */
.stButton>button {
    background: linear-gradient(135deg, #8B4513 0%, #A0522D 100%);
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 12px 24px;
    border: none;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
FILE = "etf_master_new_cleaned.csv"
HOLDINGS_FILE = "holding_analysis.csv"

if not os.path.exists(FILE):
    st.error("❌ File etf_master_new_cleaned.csv not found")
    st.stop()

df = pd.read_csv(FILE, dtype=str)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

# ============================================================
# LOAD HOLDINGS DATA
# ============================================================
holdings_dict = {}

if os.path.exists(HOLDINGS_FILE):
    with open(HOLDINGS_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_header = None
    current_data = []

    for line in lines:
        line = line.strip()

        if not line or line == ',,,,,,,,,,,,,':
            if current_header and current_data:
                try:
                    temp_df = pd.DataFrame(current_data)
                    for _, row in temp_df.iterrows():
                        etf_name = row.get('ETF', '').strip()
                        if etf_name and etf_name != 'ETF':
                            holdings_dict[etf_name] = row.to_dict()
                except Exception as e:
                    pass

                current_header = None
                current_data = []
            continue

        parts = [p.strip() for p in line.split(',')]

        if parts[0] == 'ETF':
            if current_header and current_data:
                try:
                    temp_df = pd.DataFrame(current_data)
                    for _, row in temp_df.iterrows():
                        etf_name = row.get('ETF', '').strip()
                        if etf_name and etf_name != 'ETF':
                            holdings_dict[etf_name] = row.to_dict()
                except Exception as e:
                    pass

            current_header = parts
            current_data = []
        elif current_header and parts[0]:
            row_dict = {}
            for i, col_name in enumerate(current_header):
                if i < len(parts):
                    row_dict[col_name] = parts[i]
            current_data.append(row_dict)

    if current_header and current_data:
        try:
            temp_df = pd.DataFrame(current_data)
            for _, row in temp_df.iterrows():
                etf_name = row.get('ETF', '').strip()
                if etf_name and etf_name != 'ETF':
                    holdings_dict[etf_name] = row.to_dict()
        except Exception as e:
            pass

# AUM cleanup
aum_col = next((c for c in df.columns if "aum" in c), None)
if aum_col:
    df["aum"] = (
        df[aum_col]
        .str.replace(",", "")
        .str.replace("\u2009", "")
    )
    df["aum"] = pd.to_numeric(df["aum"], errors="coerce")
else:
    df["aum"] = 0.0

# Helper columns
df["_amc"] = df["amc"].str.lower()
df["_asset"] = df["asset_class"].str.lower()
df["_text"] = (
    df["category"].astype(str).str.lower() + " " +
    df["benchmark_index"].astype(str).str.lower()
)

# ============================================================
# ASSET MAP & CATEGORIES
# ============================================================
ASSET_MAP = {
    "EQUITY": r"\bequity\b",
    "GLOBAL EQUITY": r"global|international|overseas|nasdaq|s&p|msci",
    "DEBT": r"debt|fixed\s*income|bond|gilt|g-sec|sdl|liquid",
    "COMMODITIES": r"commodity|gold|silver"
}

BROADER = {
    "Nifty 50": r"nifty\s*50\b(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility|shariah))",
    "BSE Sensex": r"\bsensex\b(?!\s*next)(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility))",
    "Nifty Next 50": r"nifty\s*next\s*50(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility))",
    "Nifty Total Market": r"nifty\s*total\s*market(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility))",
    "BSE SENSEX Next 30": r"sensex\s*next\s*30(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility))",
    "Nifty LargeMidCap 250": r"largemidcap\s*250(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility))",
    "Nifty 200": r"nifty\s*200\b(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility|factor))",
    "Nifty SmallCap 250": r"smallcap\s*250\b(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility))",
    "Nifty Midcap 150": r"midcap\s*150\b(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility))",
    "Nifty 100": r"nifty\s*100\b(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility|factor))",
    "BSE 500": r"bse\s*500(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility))",
    "BSE Midcap Select": r"midcap\s*select(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility))",
    "Nifty Midcap 50": r"midcap\s*50\b(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility))",
    "Nifty MidCap 100": r"midcap\s*100\b(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility))",
    "Nifty 500 MultiCap 50:25:25": r"(?:50:25:25|nifty\s*500\s*multicap\s*50\s*25\s*25)(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility))",
    "Nifty 500": r"nifty\s*500\b(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility|factor|multicap\s*50\s*25\s*25|flexicap))",
    "BSE Sensex Next 50": r"sensex\s*next\s*50(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility))",
    "BSE 100": r"bse\s*100(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility))",
    "Nifty Smallcap 100": r"smallcap\s*100\b(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility))"
}

SECTORAL = {
    "BANK": r"\bbank\b",
    "FINANCIAL SERVICES": r"financial",
    "HEALTHCARE": r"healthcare|health",
    "IT": r"\bit\b|information\s*technology",
    "PRIVATE BANK": r"private.*bank",
    "PSU BANK": r"psu.*bank",
    "POWER": r"power",
    "REALTY": r"realty|real\s*estate",
    "AUTO": r"\bauto\b",
    "FINANCIAL SERVICES EX-BANK": r"financial.*ex.*bank",
    "FMCG": r"fmcg",
    "OIL AND GAS": r"oil|gas",
    "CHEMICALS": r"chemical",
    "Metal": r"metal",
    "PHARMA": r"pharma"
}


THEMATIC = {
    "PSE": r"\bpse\b(?!.*bank)",
    "India Cosumption": r"india.*consumption",
    "Capital Market & Insurance": r"capital.*market.*insurance",
    "EV and New Age Automotive": r"\bev\b|new.*age.*auto",
    "Defence": r"defence|defense",
    "Internet": r"internet",
    "Railways": r"railway",
    "PSU": r"\bpsu\b(?!.*bank)",
    "Capital Market": r"capital.*market(?!.*insurance)",
    "Commodities": r"commodit",
    "Infrastructure": r"infrastructure(?!.*india)|infra(?!.*india)",
    "BHARAT 22": r"bharat.*22",
    "Metal": r"\bmetal\b",
    "MNC": r"\bmnc\b",
    "Energy": r"\benergy\b(?!.*new)",
    "Manufacturing": r"manufacturing",
    "New Age Consumtion": r"new.*age.*consum",
    "ESG SECTOR LEADERS": r"\besg\b",
    "Tourism": r"tourism",
    "India Infrastructure": r"india.*infrastructure",
    "Nifty 50 shariah": r"nifty.*50.*shariah",
    "CPSE": r"\bcpse\b",
    "PSU BANK": r"psu.*bank",
    "Digital": r"digital"
}

STRATEGIC = {
    "Nifty 50 Factor Indices": r"nifty\s*50.*(?:equal\s*weight|value\s*20|shariah)",
    "Nifty 100 Factor Indices": r"nifty\s*100.*(?:quality\s*30|low\s*volatility\s*30|equal\s*weight)",
    "Nifty 200 Factor Indices": r"nifty\s*200.*(?:momentum\s*30|quality\s*30|value\s*30|alpha\s*30)",
    "Nifty 500 Factor Indices": r"nifty\s*500.*(?:value\s*50|flexicap\s*quality\s*30|multicap\s*momentum\s*quality\s*50|momentum\s*50|low\s*volatility\s*50)",
    "Nifty Alpha Factor Indices": r"nifty\s*alpha.*(?:low\s*volatility\s*30|alpha\s*50)|nifty\s*alpha\s*50",
    "Nifty Dividend Opportunities 50": r"nifty\s*dividend\s*opportunities\s*50",
    "Nifty Growth Sectors 15": r"nifty\s*growth\s*sectors\s*15",
    "Nifty Midcap 150 Factor Indices": r"nifty\s*midcap\s*150.*(?:quality\s*50|momentum\s*50)",
    "Nifty MidSmallcap400 Momentum Quality 100": r"nifty\s*midsmallcap\s*400\s*momentum\s*quality\s*100",
    "Nifty Smallcap250 Momentum Quality 100": r"nifty\s*smallcap\s*250\s*momentum\s*quality\s*100",
    "Nifty Top 10": r"nifty\s*top\s*10\s*equal\s*weight",
    "Nifty Top 15": r"nifty\s*top\s*15\s*equal\s*weight",
    "Nifty Total Market Factor Indices": r"nifty\s*total\s*market.*(?:momentum\s*quality\s*50)",
    "BSE Select IPO": r"bse\s*select\s*ipo",
    "BSE Quality": r"bse\s*quality",
    "BSE Low Volatility": r"bse\s*low\s*volatility",
    "BSE Enhanced Value": r"bse\s*enhanced\s*value",
    "BSE 200 Factor Indices": r"bse\s*200\s*equal\s*weight"
}

# ============================================================
# FILTER UI
# ============================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #FFF9C4 0%, #FFF59D 100%);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
">
    <h3 style="color: #F57F17; margin: 0 0 15px 0; text-align: center;">🔍 Filter ETFs</h3>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    amc_list = ["All"] + sorted(df["amc"].dropna().unique())
    selected_amc = st.selectbox("Select AMC", amc_list)

with c2:
    selected_asset = st.selectbox(
        "Select Asset Class",
        ["All", "EQUITY", "GLOBAL EQUITY", "DEBT", "COMMODITIES"]
    )

sub_cat = None
sub_sub_cat = None
lookup = None

if selected_asset == "EQUITY":
    sub_cat = st.selectbox(
        "Select Equity Category",
        ["Broader", "Sectoral", "Thematic", "Strategic"]
    )

    lookup = {
        "Broader": BROADER,
        "Sectoral": SECTORAL,
        "Thematic": THEMATIC,
        "Strategic": STRATEGIC
    }[sub_cat]

    sub_sub_cat = st.selectbox(
        f"Select {sub_cat}",
        ["All"] + sorted(lookup.keys())
    )

elif selected_asset == "DEBT":
    sub_cat = st.selectbox(
        "Select Debt Category",
        ["All", "Bharat Bond", "G-Sec", "Gilt", "Liquid", "SDL"]
    )

elif selected_asset == "COMMODITIES":
    sub_cat = st.selectbox(
        "Select Commodity",
        ["All", "Gold", "Silver"]
    )

# ============================================================
# FILTER LOGIC
# ============================================================
mask = pd.Series(True, index=df.index)

if selected_amc != "All":
    mask &= df["_amc"] == selected_amc.lower()

if selected_asset != "All" and selected_asset != "COMMODITIES":
    mask &= df["_asset"].str.contains(ASSET_MAP[selected_asset], na=False)

if selected_asset == "COMMODITIES":
    mask &= df["_text"].str.contains(r"commodity|gold|silver", na=False)
    if sub_cat and sub_cat != "All":
        commodity_map = {"Gold": r"gold", "Silver": r"silver"}
        mask &= df["_text"].str.contains(commodity_map[sub_cat], na=False)

if selected_asset == "EQUITY" and sub_sub_cat and sub_sub_cat != "All":
    mask &= df["_text"].str.contains(lookup[sub_sub_cat], na=False)

if selected_asset == "DEBT" and sub_cat and sub_cat != "All":
    debt_map = {
        "Bharat Bond": r"bharat\s*bond",
        "G-Sec": r"g[-\s]?sec|government",
        "Gilt": r"gilt",
        "Liquid": r"liquid|overnight|money\s*market",
        "SDL": r"sdl|state\s*development"
    }
    mask &= df["_text"].str.contains(debt_map[sub_cat], na=False)

result = df.loc[mask].copy()

# ============================================================
# METRICS - ENHANCED STYLING
# ============================================================
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #E1F5FE 0%, #81D4FA 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    ">
        <p style="margin: 0; font-size: 18px; color: #01579B; font-weight: bold;">📊 Number of ETFs</p>
        <p style="margin: 10px 0 0 0; font-size: 36px; color: #0277BD; font-weight: bold;">{result.shape[0]}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #C8E6C9 0%, #66BB6A 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    ">
        <p style="margin: 0; font-size: 18px; color: #1B5E20; font-weight: bold;">💰 Total AUM</p>
        <p style="margin: 10px 0 0 0; font-size: 36px; color: #2E7D32; font-weight: bold;">₹ {result['aum'].sum(skipna=True):,.2f} Cr</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# ETF SELECTOR
# ============================================================
selected_etf = st.selectbox(
    "🎯 Select an ETF to view details",
    [""] + sorted(result["etf"].dropna().unique().tolist())
)

# ============================================================
# IMPROVED LAYOUT: ETF DETAILS + HOLDINGS + PRICE CARD
# ============================================================
if selected_etf:
    row = result[result["etf"] == selected_etf].iloc[0]

    # Main container
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #FFFFFF 0%, #F5F5F5 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin-bottom: 25px;
    ">
        <h2 style="color: #4A235A; margin: 0 0 20px 0; text-align: center;">📄 ETF Details & Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    # TWO COLUMN LAYOUT: LEFT (Details + Holdings) | RIGHT (Price Card)
    left_col, right_col = st.columns([2, 1])

    with left_col:
        # ETF DETAILS CARD
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #E8EAF6 0%, #C5CAE9 100%);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <h3 style="color: #311B92; margin: 0 0 15px 0;">📊 Basic Information</h3>
        </div>
        """, unsafe_allow_html=True)

        # Details in two columns
        d1, d2 = st.columns(2)

        with d1:
            st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>ETF Name:</span> <span style='color:#000000;'>{row.get('etf', '-')}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>AMC:</span> <span style='color:#000000;'>{row.get('amc', '-')}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>Category:</span> <span style='color:#000000;'>{row.get('category', '-')}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>Asset Class:</span> <span style='color:#000000;'>{row.get('asset_class', '-')}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>Benchmark:</span> <span style='color:#000000;'>{row.get('benchmark_index', '-')}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>Launch Date:</span> <span style='color:#000000;'>{row.get('launch_date', '-')}</span></p>", unsafe_allow_html=True)

        with d2:
            # FIXED: Changed from 'nse_ticker' to 'symbol' to match the new column name
            st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>NSE Ticker:</span> <span style='color:#000000;'>{row.get('symbol', '-')}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>BSE Ticker:</span> <span style='color:#000000;'>{row.get('bse_ticker', '-')}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>ISIN Code:</span> <span style='color:#000000;'>{row.get('isin_code', '-')}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>AUM (₹ Cr):</span> <span style='color:#000000;'>{row.get('aum', '-')}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>Expense Ratio:</span> <span style='color:#000000;'>{row.get('expense_ratio', '-')}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>Tracking Error:</span> <span style='color:#000000;'>{row.get('overall_tracking_error', '-')}</span></p>", unsafe_allow_html=True)

        # Official Website Link
        website = row.get("website_link", "")
        if isinstance(website, str) and website.strip():
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #8B4513 0%, #A0522D 100%);
                    padding: 14px;
                    border-radius: 10px;
                    text-align: center;
                    margin-top: 15px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                ">
                    <a href="{website}" target="_blank"
                       style="color:white;font-weight:bold;text-decoration:none;font-size:16px;">
                        🌐 Visit Official Website
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )

        # HOLDINGS CHART (BELOW DETAILS)
        st.markdown("<br>", unsafe_allow_html=True)

        if selected_etf in holdings_dict:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 15px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            ">
                <h3 style="color: #B71C1C; margin: 0;">📈 Top 10 Holdings</h3>
            </div>
            """, unsafe_allow_html=True)

            etf_data = holdings_dict[selected_etf]
            holdings_data = []
            skip_keys = ['ETF', 'ISIN CODE', 'NSE TICKER', 'Total']

            for key, value in etf_data.items():
                if key not in skip_keys and value and value.strip():
                    try:
                        company_name = key.replace('Ltd.', '').replace('Ltd,', '').replace('Ltd', '').replace('Ordinary Shares', '').replace('Class A', '').replace('Class B', '').replace('Class H', '').strip()
                        weight = float(value)
                        holdings_data.append({
                            'Company': company_name,
                            'Weight': weight
                        })
                    except:
                        pass

            if holdings_data:
                holdings_chart_df = pd.DataFrame(holdings_data)
                holdings_chart_df = holdings_chart_df.sort_values('Weight', ascending=False)

                color_scale = alt.Scale(
                    domain=[holdings_chart_df['Weight'].min(), holdings_chart_df['Weight'].max()],
                    range=['#FFCDD2', '#8B0000']
                )

                base = alt.Chart(holdings_chart_df).encode(
                    y=alt.Y('Company:N',
                           sort=alt.EncodingSortField(field='Weight', order='descending'),
                           axis=alt.Axis(title='Stock/Issuer', labelFontSize=11, labelFontWeight='bold')),
                    x=alt.X('Weight:Q',
                           axis=alt.Axis(title='Weight (%) to NAV', titleFontSize=13, labelFontSize=10))
                )

                bars = base.mark_bar().encode(
                    color=alt.Color('Weight:Q', scale=color_scale, legend=None),
                    tooltip=[
                        alt.Tooltip('Company:N', title='Company'),
                        alt.Tooltip('Weight:Q', title='Weight (%)', format='.2f')
                    ]
                )

                text = base.mark_text(
                    align='left',
                    baseline='middle',
                    dx=5,
                    fontSize=11,
                    fontWeight='bold'
                ).encode(
                    text=alt.Text('Weight:Q', format='.2f')
                )

                final_chart = alt.layer(bars, text).properties(
                    height=500
                ).configure_view(
                    strokeWidth=0
                ).configure_axis(
                    gridColor='rgba(128, 128, 128, 0.2)'
                )

                st.altair_chart(final_chart, width='stretch')

                total_value = etf_data.get('Total', '')
                if total_value and total_value.strip():
                    try:
                        total_pct = float(total_value)
                        st.markdown(
                            f"""
                            <div style="
                                background: linear-gradient(135deg, #F44336 0%, #B71C1C 100%);
                                padding: 16px;
                                border-radius: 10px;
                                text-align: center;
                                margin-top: 15px;
                                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            ">
                                <p style="color: white; font-size: 17px; margin: 0; font-weight: bold;">
                                    💎 Contribution from Top 10 Holdings: {total_pct}%
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    except:
                        pass
            else:
                st.info("📊 Holdings data available but could not be parsed.")
        else:
            st.info("📊 Holdings data not available for this ETF.")

    # RIGHT COLUMN - PRICE CARD
    with right_col:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #E3F2FD 0%, #90CAF9 100%);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
        ">
            <h3 style="color: #0D47A1; margin: 0;">💹 Price Information</h3>
        </div>
        """, unsafe_allow_html=True)

        # Load price data
        PRICE_FILE="data/nse_etf_prices.csv"
        if os.path.exists(PRICE_FILE):
            df_price = pd.read_csv(PRICE_FILE)
            # 1️⃣ Normalize column names FIRST
            df_price.columns = df_price.columns.str.strip().str.lower().str.replace(" ", "_")  
            # 2️⃣ Parse date with correct format (dd-mm-yyyy)
            df_price["date"] = pd.to_datetime(
            df_price["date"],
            format="%d-%m-%Y",
             errors="coerce"
             )
            # 3️⃣ Now safely compute latest date
            latest_price_date = df_price["date"].max()


        # FIXED: Price file uses 'symbol' column, standardize it
        if 'symbol' in df_price.columns:
            df_price["symbol"] = (
            df_price["symbol"]
            .astype(str)
            .str.strip()
            .str.upper()
        )    
        elif 'nse_ticker' in df_price.columns:
            df_price["symbol"] = (
            df_price["nse_ticker"]
            .astype(str)
            .str.strip()
            .str.upper()
        )


            # FIXED: Get the ticker from the 'symbol' column in the master file
            ticker = str(row.get("symbol", "")).strip().upper()
            price_row = df_price[df_price["symbol"] == ticker]
            st.write("DEBUG ticker from master:", ticker)
            st.write("DEBUG unique symbols in price file (sample):", df_price["symbol"].unique()[:10])
            st.write("DEBUG price file columns:", df_price.columns.tolist())


            if not price_row.empty:
                # Initialize session state for price visibility
                if f'show_price_{ticker}' not in st.session_state:
                    st.session_state[f'show_price_{ticker}'] = False

                # Toggle button
                if st.button("📊 View LTP vs NAV", key=f"btn_{ticker}"):
                    st.session_state[f'show_price_{ticker}'] = not st.session_state[f'show_price_{ticker}']

                # Show price chart if toggled
                if st.session_state[f'show_price_{ticker}']:
                    ltp_val = pd.to_numeric(price_row.iloc[0]["ltp"], errors="coerce")
                    nav_val = pd.to_numeric(price_row.iloc[0]["nav"], errors="coerce")
                if pd.isna(ltp_val) or pd.isna(nav_val):
                    st.warning("⚠️ Price data unavailable or invalid for this ETF.")
                    st.stop()
                    st.markdown(f"""
                    <div style="
                        background: white;
                        padding: 20px;
                        border-radius: 10px;
                        margin-top: 15px;
                        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
                    ">
                        <p style="margin: 8px 0; font-size: 16px;">
                            <span style="color: #1976D2; font-weight: bold;">LTP:</span>
                            <span style="color: #000; font-size: 20px; font-weight: bold;">₹ {ltp_val}</span>
                        </p>
                        <p style="margin: 8px 0; font-size: 16px;">
                            <span style="color: #388E3C; font-weight: bold;">NAV:</span>
                            <span style="color: #000; font-size: 20px; font-weight: bold;">₹ {nav_val}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Bar chart
                    price_df = pd.DataFrame({
                        "Metric": ["LTP", "NAV"],
                        "Value": [ltp_val, nav_val]
                    })

                    price_chart = alt.Chart(price_df).mark_bar(
                        size=60
                    ).encode(
                        x=alt.X("Metric:N", axis=alt.Axis(labelFontWeight="bold", labelFontSize=14)),
                        y=alt.Y("Value:Q", title="₹ Value"),
                        color=alt.Color("Metric:N", scale=alt.Scale(range=["#42A5F5", "#66BB6A"]), legend=None),
                        tooltip=["Metric", alt.Tooltip("Value:Q", format=".2f")]
                    ).properties(height=300)

                    st.altair_chart(price_chart, width='stretch')
                    st.caption(f"Data as of: {latest_price_date.strftime('%d %b %Y')}")

                    # Close button
                    if st.button("❌ Close", key=f"close_{ticker}"):
                        st.session_state[f'show_price_{ticker}'] = False
                        st.rerun()
            else:
                st.warning("⚠️ Price data not available for this ticker.")
        else:
            st.warning("⚠️ Price data file not found.")

# ============================================================
# AI ASSISTANT SECTION - ENHANCED
# ============================================================
st.markdown("---")
st.markdown("""
<div style="
    background: linear-gradient(135deg, #FFF3E0 0%, #FFB74D 100%);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin: 30px 0;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
">
    <h2 style="color: #E65100; margin: 0 0 15px 0;">🤖 AI-Powered ETF Assistant</h2>
    <p style="color: #BF360C; font-size: 16px; margin: 0 0 20px 0;">
        Get intelligent insights and analysis on Indian ETFs
    </p>
    <a href="https://chatgpt.com/g/g-6942d299b4648191a8acc98e68636cb9-indiaetf" target="_blank"
       style="
           display: inline-block;
           background: linear-gradient(135deg, #8B4513 0%, #A0522D 100%);
           color: white;
           font-weight: bold;
           text-decoration: none;
           font-size: 18px;
           padding: 15px 40px;
           border-radius: 10px;
           box-shadow: 0 4px 8px rgba(0,0,0,0.2);
           transition: all 0.3s ease;
       "
       onmouseover="this.style.transform='translateY(-3px)';this.style.boxShadow='0 6px 12px rgba(0,0,0,0.3)';"
       onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='0 4px 8px rgba(0,0,0,0.2)';">
        🚀 Launch AI Assistant
    </a>
</div>
""", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="
    background: linear-gradient(135deg, #D1C4E9 0%, #B39DDB 100%);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
">
    <p style="font-size: 16px; color: #4A148C; font-weight: bold; margin: 0;">
        ✨ Conceptualized by Diganta Raychaudhuri ✨
    </p>
    <p style="font-size: 13px; color: #6A1B9A; margin: 5px 0 0 0;">
        let's invest in passives, actively
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style="
    font-size: 11px;
    color: #555;
    text-align: left;
    margin-top: 8px;
    padding-left: 6px;
">
    Data taken from NSE, AMFI and AMC websites.
</div>
""", unsafe_allow_html=True)
# ============================================================
# LEGAL + COPYRIGHT + LAST UPDATED
# ============================================================

st.markdown("""
<div style="
    font-size: 11px;
    color: #444;
    text-align: left;
    margin-top: 10px;
    line-height: 1.5;
">
    <strong>Disclaimer:</strong><br>
    This website is for informational and educational purposes only.
    It does not constitute investment advice.
    Please consult a registered financial advisor before making investment decisions.
</div>
""", unsafe_allow_html=True)
if latest_price_date is not None:
    st.markdown(f"""
    <div style="
        font-size: 11px;
        color: #444;
        text-align: left;
        margin-top: 6px;
    ">
        © {latest_price_date.year} Diganta Raychaudhuri. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="
        font-size: 11px;
        color: #444;
        text-align: left;
        margin-top: 4px;
    ">
        Last updated: {latest_price_date.strftime('%d %b %Y')} (EOD, NSE)
    </div>
    """, unsafe_allow_html=True)
