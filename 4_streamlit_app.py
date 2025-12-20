import streamlit as st
import pandas as pd
import os
import altair as alt

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
}
.scrolling-text {
    position: absolute;
    white-space: nowrap;
    font-family: "Comic Sans MS";
    font-size: 30px;
    font-weight: bold;
    color: red;
    animation: scroll-left 12s linear infinite;
}
@keyframes scroll-left {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}
</style>

<div class="scrolling-container">
    <div class="scrolling-text">
        WORK IN PROGRESS THANK YOU FOR YOUR PATIENCE
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
">
    <h1 style="color: white; margin: 0;">📊 India ETF Tracker</h1>
</div>
""", unsafe_allow_html=True)

# Black info message
st.markdown("""
<div style="
    background-color: #f0f0f0;
    padding: 12px;
    border-radius: 8px;
    border-left: 5px solid #000000;
    margin-bottom: 20px;
">
    <p style="color: #000000; margin: 0; font-weight: 600; font-size: 15px;">
        ℹ️ Detailed holding available for selected representative ETFs only
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
# LOAD HOLDINGS DATA - SPECIAL HANDLING FOR MULTIPLE GROUPS
# ============================================================
holdings_dict = {}

if os.path.exists(HOLDINGS_FILE):
    # Read the raw CSV file
    with open(HOLDINGS_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_header = None
    current_data = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line or line == ',,,,,,,,,,,,,':
            # If we have accumulated data, process it
            if current_header and current_data:
                try:
                    # Create DataFrame from accumulated data
                    temp_df = pd.DataFrame(current_data)
                    # Store each ETF's data in the dictionary
                    for _, row in temp_df.iterrows():
                        etf_name = row.get('ETF', '').strip()
                        if etf_name and etf_name != 'ETF':
                            holdings_dict[etf_name] = row.to_dict()
                except Exception as e:
                    pass

                # Reset for next group
                current_header = None
                current_data = []
            continue

        # Split the line
        parts = [p.strip() for p in line.split(',')]

        # Check if this is a header line (starts with "ETF")
        if parts[0] == 'ETF':
            # Process previous group if exists
            if current_header and current_data:
                try:
                    temp_df = pd.DataFrame(current_data)
                    for _, row in temp_df.iterrows():
                        etf_name = row.get('ETF', '').strip()
                        if etf_name and etf_name != 'ETF':
                            holdings_dict[etf_name] = row.to_dict()
                except Exception as e:
                    pass

            # Start new group
            current_header = parts
            current_data = []
        elif current_header and parts[0]:  # Data row (has ETF name)
            # Create row dictionary
            row_dict = {}
            for i, col_name in enumerate(current_header):
                if i < len(parts):
                    row_dict[col_name] = parts[i]
            current_data.append(row_dict)

    # Process last group
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
# ASSET MAP
# ============================================================
ASSET_MAP = {
    "EQUITY": r"\bequity\b",
    "GLOBAL EQUITY": r"global|international|overseas|nasdaq|s&p|msci",
    "DEBT": r"debt|fixed\s*income|bond|gilt|g-sec|sdl|liquid",
    "COMMODITIES": r"commodity|gold|silver"
}

# ============================================================
# BROADER CATEGORIES - EXCLUDE FACTOR INDICES
# ============================================================
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

# ============================================================
# SECTORAL
# ============================================================
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

# ============================================================
# THEMATIC
# ============================================================
THEMATIC = {
    "BHARAT 22": r"bharat.*22",
    "Capital Market": r"capital.*market(?!.*insurance)",
    "Capital Market & Insurance": r"capital.*market.*insurance",
    "Commodities": r"commodit",
    "CPSE": r"\bcpse\b",
    "Defence": r"defence|defense",
    "Digital": r"digital",
    "Energy": r"energy",
    "ESG SECTOR LEADERS": r"esg",
    "EV and New Age Automotive": r"ev|new.*age.*auto",
    "India Cosumption": r"india.*consumption",
    "India Infrastructure": r"india.*infrastructure",
    "Infrastructure": r"infrastructure|infra",
    "Internet": r"internet",
    "Manufacturing": r"manufacturing",
    "MNC": r"\bmnc\b",
    "New Age Consumtion": r"new.*age.*consum",
    "PSE": r"\bpse\b",
    "Railways": r"railway",
    "PSU": r"\bpsu\b",
    "Tourism": r"tourism"
}

# ============================================================
# STRATEGIC - FIXED WITH NON-CAPTURING GROUPS
# ============================================================
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
st.write("### 🔍 Filter ETFs")

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
        ["All"] + list(lookup.keys())
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
# METRICS
# ============================================================
st.write("---")
c1, c2 = st.columns(2)
c1.metric("📊 Number of ETFs", result.shape[0])
c2.metric("💰 Total AUM (₹ Crores)", f"₹ {result['aum'].sum(skipna=True):,.2f}")
st.write("---")

# ============================================================
# ETF SELECTOR
# ============================================================
selected_etf = st.selectbox(
    "Select an ETF to view details",
    [""] + result["etf"].tolist()
)

# ============================================================
# ETF CARD WITH HOLDINGS CHART
# ============================================================
if selected_etf:
    row = result[result["etf"] == selected_etf].iloc[0]

    st.markdown("### 📄 ETF Details")

    left, right = st.columns([1, 1])

    with left:
        # Display ETF details with colored labels
        st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>ETF Name:</span> <span style='color:#000000;'>{row.get('etf', '-')}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>AMC:</span> <span style='color:#000000;'>{row.get('amc', '-')}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>Category:</span> <span style='color:#000000;'>{row.get('category', '-')}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>Asset Class:</span> <span style='color:#000000;'>{row.get('asset_class', '-')}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>Benchmark Index:</span> <span style='color:#000000;'>{row.get('benchmark_index', '-')}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>Launch Date:</span> <span style='color:#000000;'>{row.get('launch_date', '-')}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>NSE Ticker:</span> <span style='color:#000000;'>{row.get('nse_ticker', '-')}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>BSE Ticker:</span> <span style='color:#000000;'>{row.get('bse_ticker', '-')}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>ISIN Code:</span> <span style='color:#000000;'>{row.get('isin_code', '-')}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>AUM (₹ Crores):</span> <span style='color:#000000;'>{row.get('aum', '-')}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>Expense Ratio:</span> <span style='color:#000000;'>{row.get('expense_ratio', '-')}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>Tracking Error:</span> <span style='color:#000000;'>{row.get('overall_tracking_error', '-')}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:5px 0;'><span style='color:#4A235A;font-weight:bold;'>Tracking Difference:</span> <span style='color:#000000;'>{row.get('overall_tracking_difference', '-')}</span></p>", unsafe_allow_html=True)

        # Official Website Link at bottom left
        st.write("")
        website = row.get("website_link", "")
        if isinstance(website, str) and website.strip():
            st.markdown(
                f"""
                <div style="background:#8B4513;padding:14px;border-radius:8px;text-align:center;margin-top:15px;">
                    <a href="{website}" target="_blank"
                       style="color:white;font-weight:bold;text-decoration:none;">
                        🌐 Official Website
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ============================================================
    # TOP 10 HOLDINGS CHART - RIGHT SIDE (RED GRADIENT - HIGHEST AT TOP)
    # ============================================================
    with right:
        if selected_etf in holdings_dict:
            st.markdown("### 📈 Top 10 Holdings")

            etf_data = holdings_dict[selected_etf]

            # Extract holdings (skip ETF, ISIN CODE, NSE TICKER, Total)
            holdings_data = []
            skip_keys = ['ETF', 'ISIN CODE', 'NSE TICKER', 'Total']

            for key, value in etf_data.items():
                if key not in skip_keys and value and value.strip():
                    try:
                        # Clean company name
                        company_name = key.replace('Ltd.', '').replace('Ltd,', '').replace('Ltd', '').replace('Ordinary Shares', '').replace('Class A', '').replace('Class B', '').replace('Class H', '').strip()
                        weight = float(value)

                        holdings_data.append({
                            'Company': company_name,
                            'Weight': weight
                        })
                    except:
                        pass

            if holdings_data:
                # Create DataFrame and sort by weight (DESCENDING - highest at top)
                holdings_chart_df = pd.DataFrame(holdings_data)
                holdings_chart_df = holdings_chart_df.sort_values('Weight', ascending=False)

                # Create red gradient color scale - light to dark red
                color_scale = alt.Scale(
                    domain=[holdings_chart_df['Weight'].min(), holdings_chart_df['Weight'].max()],
                    range=['#FFCDD2', '#8B0000']
                )

                # Create the base chart layer
                base = alt.Chart(holdings_chart_df).encode(
                    y=alt.Y('Company:N', 
                           sort=alt.EncodingSortField(field='Weight', order='descending'),
                           axis=alt.Axis(title='', labelFontSize=12, labelFontWeight='bold')),
                    x=alt.X('Weight:Q', 
                           axis=alt.Axis(title='Weight (%)', titleFontSize=14, labelFontSize=11))
                )

                # Bar layer
                bars = base.mark_bar().encode(
                    color=alt.Color('Weight:Q', scale=color_scale, legend=None),
                    tooltip=[
                        alt.Tooltip('Company:N', title='Company'),
                        alt.Tooltip('Weight:Q', title='Weight (%)', format='.2f')
                    ]
                )

                # Text layer
                text = base.mark_text(
                    align='left',
                    baseline='middle',
                    dx=5,
                    fontSize=12,
                    fontWeight='bold'
                ).encode(
                    text=alt.Text('Weight:Q', format='.2f')
                )

                # Layer the charts
                final_chart = alt.layer(bars, text).properties(
                    height=550
                ).configure_view(
                    strokeWidth=0
                ).configure_axis(
                    gridColor='rgba(128, 128, 128, 0.2)'
                )

                st.altair_chart(final_chart, width='stretch')

                # Display Total contribution with enhanced styling (red gradient)
                total_value = etf_data.get('Total', '')
                if total_value and total_value.strip():
                    try:
                        total_pct = float(total_value)
                        st.markdown(
                            f"""
                            <div style="
                                background: linear-gradient(135deg, #F44336 0%, #B71C1C 100%);
                                padding: 18px;
                                border-radius: 10px;
                                text-align: center;
                                margin-top: 15px;
                                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            ">
                                <p style="color: white; font-size: 18px; margin: 0; font-weight: bold; font-family: 'Comic Sans MS';">
                                    💎 Total contribution from top 10 holdings: {total_pct}%
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
            st.info("📊 Holdings data not available for this ETF in the dataset.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 10px;">
    <p style="font-size: 14px; color: #4A235A; font-weight: bold;">
        Conceptualized by Diganta Raychaudhuri
    </p>
</div>
""", unsafe_allow_html=True)
