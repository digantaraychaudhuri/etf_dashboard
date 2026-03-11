import streamlit as st
import pandas as pd
import os
import altair as alt
import datetime
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import seaborn as sns
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Suppress font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Indian ETF Tracker",
    layout="wide"
)
# ============================================================
# GOOGLE SEARCH CONSOLE VERIFICATION
# ============================================================
st.markdown(
    """
    <meta name="google-site-verification" content="a5U3VkvBtjUfe3hihHvAFn8JFn7rQsqeCG3IfehNgSQ" />
    """,
    unsafe_allow_html=True
)
# ============================================================
# MATPLOTLIB FONT CONFIGURATION
# ============================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']

# ============================================================
# TECHNICAL ANALYSIS FUNCTIONS
# ============================================================
def calculate_returns(data):
    """Calculate multi-period returns"""

    if data is None or data.empty:
        return None

    close = data['Close'].dropna()

    if len(close) < 10:
        return None

    latest = close.iloc[-1]

    periods = {
        "3D": 3,
        "5D": 5,
        "1M": 21,
        "1Y": 252,
        "3Y": 756,
        "5Y": 1260
    }

    results = {}            

    for name, days in periods.items():
        if len(close) > days:
            past = close.iloc[-days]
            results[name] = ((latest / past) - 1) * 100
        else:
            results[name] = None

    return results

def calculate_cagr(data):
    """Calculate CAGR values - uses available data intelligently"""
    
    if data is None or data.empty:
        return None

    close = data['Close'].dropna()

    if len(close) < 2:
        return None

    latest = close.iloc[-1]
    latest_date = close.index[-1]
    earliest_date = close.index[0]
    
    # Calculate actual years of data available
    total_years = (latest_date - earliest_date).days / 365.25
    
    cagr_periods = {
        "2Y CAGR": 2,
        "3Y CAGR": 3,
        "5Y CAGR": 5
    }

    results = {}

    for label, target_years in cagr_periods.items():
        # Strategy: Use whatever data we have if it's close enough
        # If we have at least 80% of the required period, calculate CAGR
        min_required_years = target_years * 0.80  # 80% threshold
        
        if total_years < min_required_years:
            results[label] = None
            continue
        
        # Calculate the target date (going back target_years from latest)
        target_date = latest_date - pd.Timedelta(days=int(target_years * 365.25))
        
        try:
            if target_date < earliest_date:
                # Target date is before our data starts
                # Use all available data and mark it
                past_price = close.iloc[0]
                past_date = earliest_date
            else:
                # Find closest date to target
                idx = close.index.get_indexer([target_date], method='nearest')[0]
                past_price = close.iloc[idx]
                past_date = close.index[idx]
            
            # Calculate actual years between dates
            actual_years = (latest_date - past_date).days / 365.25
            
            # Calculate CAGR
            if actual_years > 0.1:  # Need at least ~1 month of data
                cagr = ((latest / past_price) ** (1/actual_years) - 1) * 100
                results[label] = cagr
            else:
                results[label] = None
                
        except Exception as e:
            results[label] = None

    return results  

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_mfi(high, low, close, volume, period=14):
    """Calculate Money Flow Index"""
    if isinstance(high, pd.Series) and isinstance(high.index, pd.MultiIndex):
        high = high.droplevel(0) if high.index.nlevels > 1 else high
    if isinstance(low, pd.Series) and isinstance(low.index, pd.MultiIndex):
        low = low.droplevel(0) if low.index.nlevels > 1 else low
    if isinstance(close, pd.Series) and isinstance(close.index, pd.MultiIndex):
        close = close.droplevel(0) if close.index.nlevels > 1 else close
    if isinstance(volume, pd.Series) and isinstance(volume.index, pd.MultiIndex):
        volume = volume.droplevel(0) if volume.index.nlevels > 1 else volume

    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    price_increased = typical_price > typical_price.shift(1)
    price_decreased = typical_price < typical_price.shift(1)

    positive_mf = (money_flow * price_increased).rolling(window=period).sum()
    negative_mf = (money_flow * price_decreased).rolling(window=period).sum()

    mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, np.nan)))
    return mfi

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def calculate_emv(data, period=14):
    """Calculate Ease of Movement"""
    dm = ((data['High'] + data['Low']) / 2) - ((data['High'].shift(1) + data['Low'].shift(1)) / 2)
    br = (data['Volume'] / 100000000) / (data['High'] - data['Low'])
    emv = dm / br
    emv_ma = emv.rolling(period).mean()
    return emv_ma

def calculate_mean_deviation(high, low, close, period=20):
    """Calculate Mean Deviation using typical price"""
    typical_price = (high + low + close) / 3
    typical_price_avg = typical_price.rolling(window=period).mean()
    deviations = (typical_price - typical_price_avg).abs()
    mean_deviation = deviations.rolling(window=period).mean()
    return mean_deviation

def calculate_cci(high, low, close, period=40):
    """Calculate Commodity Channel Index (CCI)"""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (typical_price - sma) / (0.015 * mad)
    return cci

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD (12,26,9)"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

def get_etf_volume_signals(data):
    """ETF-optimized volume signals"""

    if data is None or data.empty or len(data) < 40:
        return {"Status": "Not enough data"}

    vol = data['Volume'].dropna()
    close = data['Close'].dropna()

    if len(vol) < 40:
        return {"Status": "Not enough data"}

    # --- Rolling averages ---
    avg5 = vol.rolling(5).mean().iloc[-1]
    avg20 = vol.rolling(20).mean().iloc[-1]
    avg40 = vol.rolling(40).mean().iloc[-1]

    latest_vol = vol.iloc[-1]

    if pd.isna(avg5) or pd.isna(avg20) or pd.isna(avg40):
        return {"Status": "Insufficient data"}

    # -------------------------
    # 1️⃣ Volume Spike (relative)
    # -------------------------
    spike_ratio = latest_vol / avg20

    if spike_ratio >= 2:
        spike = "Unusual Spike"
    elif spike_ratio >= 1.5:
        spike = "High Activity"
    elif spike_ratio >= 1.1:
        spike = "Above Normal"
    else:
        spike = "Normal"

    # -------------------------
    # 2️⃣ Trend (short vs medium)
    # -------------------------
    trend_ratio = avg5 / avg40

    if trend_ratio > 1.2:
        trend = "Strong Rising Interest"
    elif trend_ratio > 1.05:
        trend = "Rising Interest"
    elif trend_ratio < 0.85:
        trend = "Falling Interest"
    else:
        trend = "Stable"

    # -------------------------
    # 3️⃣ Price confirmation
    # -------------------------
    price_change = close.pct_change().iloc[-1]

    if price_change > 0 and spike_ratio > 1.2:
        confirm = "Bullish Participation"
    elif price_change < 0 and spike_ratio > 1.2:
        confirm = "Distribution Pressure"
    else:
        confirm = "Low Conviction"

    # -------------------------
    # 4️⃣ Liquidity context
    # -------------------------
    if avg20 > 1_000_000:
        liquidity = "Highly Liquid"
    elif avg20 > 200_000:
        liquidity = "Moderately Liquid"
    else:
        liquidity = "Thinly Traded"

    return {
        "Volume Activity": spike,
        "Volume Trend": trend,
        "Participation Signal": confirm,
        "Liquidity Profile": liquidity
    }
def analyze_volume_sentiment(vol):
    """Analyze volume sentiment across different time periods"""
    periods = [15, 21, 63, 126, 252]
    period_names = ['15-Day', '21-Day', '63-Day', '126-Day', '252-Day']
    sentiments = []
    sentiment_colors = []

    for w in periods:
        buffer = min(w * 2, len(vol))
        vol_subset = vol.tail(buffer)

        rolling_avg = vol_subset.rolling(w, min_periods=1).mean().dropna()
        if len(rolling_avg) < 2:
            sentiments.append("Insufficient data")
            sentiment_colors.append("gray")
            continue

        x = np.arange(len(rolling_avg))
        y = rolling_avg.values

        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            sentiments.append("Insufficient data")
            sentiment_colors.append("gray")
            continue

        z = np.polyfit(x, y, 1)
        slope = z[0]

        pct_change = ((y[-1] - y[0]) / y[0]) * 100 if y[0] != 0 else 0

        if slope > 0 and pct_change > 1:
            sentiments.append(f"Positive ↑ ({pct_change:.1f}%)")
            sentiment_colors.append("green")
        elif slope < 0 and pct_change < -1:
            sentiments.append(f"Negative ↓ ({pct_change:.1f}%)")
            sentiment_colors.append("red")
        else:
            sentiments.append(f"Neutral → ({pct_change:.2f}%)")
            sentiment_colors.append("gray")

    positive_count = sum(1 for s in sentiments if "Positive" in s)
    negative_count = sum(1 for s in sentiments if "Negative" in s)

    if positive_count > negative_count:
        overall = "Overall: Positive Volume Sentiment (Bullish) 📈"
        overall_color = "green"
    elif negative_count > positive_count:
        overall = "Overall: Negative Volume Sentiment (Bearish) 📉"
        overall_color = "red"
    else:
        overall = "Overall: Neutral Volume Sentiment ➡️"
        overall_color = "blue"

    return period_names, sentiments, sentiment_colors, overall, overall_color

def create_volume_charts(data, symbol):
    """Create volume analysis charts using matplotlib"""
    vol = data['Volume']
    close = data['Close']

    periods = [15, 21, 63, 126, 252]
    period_names = ['15-Day', '21-Day', '63-Day (3 months)', '126-Day (6 months)', '252-Day (1 year)']

    figs = []

    for i, (w, name) in enumerate(zip(periods, period_names)):
        buffer = min(w * 2, len(vol))
        vol_subset = vol.tail(buffer)
        close_subset = close.tail(buffer)

        vol_avg = vol_subset.rolling(w, min_periods=1).mean()

        fig = Figure(figsize=(10, 4))
        ax1 = fig.add_subplot(111)

        ax1.bar(vol_subset.index, vol_subset.values, alpha=0.7, color='#4169E1', label='Volume', width=0.8)
        ax1.plot(vol_avg.index, vol_avg.values, color='#FF8C00', linewidth=2, label=f'{w}-Day Vol Avg')
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('Volume', color='#4169E1', fontsize=10)
        ax1.tick_params(axis='y', labelcolor='#4169E1')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"{name} Analysis (Last {buffer} days)", fontsize=11, fontweight='bold')

        ax2 = ax1.twinx()
        ax2.plot(close_subset.index, close_subset.values, color='#DC143C', linewidth=2, label='Close Price')
        ax2.set_ylabel('Price (₹)', color='#DC143C', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='#DC143C')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

        fig.tight_layout()
        figs.append(fig)

    return figs

def fetch_etf_data(ticker, period="5y"):
    """Fetch ETF data from yfinance"""
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def create_technical_charts(data, symbol):
    """Create interactive technical analysis charts using Plotly with hover tooltips"""
    data = data.copy()

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data)
    data['MFI'] = calculate_mfi(data['High'], data['Low'], data['Close'], data['Volume'])
    data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'])
    data['EMV'] = calculate_emv(data)
    data['MD'] = calculate_mean_deviation(data['High'], data['Low'], data['Close'], period=20)
    data['CCI'] = calculate_cci(data['High'], data['Low'], data['Close'], period=40)
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = calculate_macd(data['Close'])

    charts = []

    # Chart 1: Price with Moving Averages
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='#2D3748', width=2), hovertemplate='Date: %{x}<br>Close: ₹%{y:.2f}<extra></extra>'))
    fig1.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='#FF8C00', width=2), hovertemplate='Date: %{x}<br>SMA 20: ₹%{y:.2f}<extra></extra>'))
    fig1.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='#4169E1', width=2), hovertemplate='Date: %{x}<br>SMA 50: ₹%{y:.2f}<extra></extra>'))
    fig1.update_layout(title=f'{symbol} - Price Chart with Moving Averages', xaxis_title='Date', yaxis_title='Price (₹)', hovermode='x unified', height=500)
    charts.append(fig1)

    # Chart 2: Price with RSI
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(f'{symbol} - Price', 'RSI (14)'), row_heights=[0.6, 0.4])
    fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='#2D3748', width=2), hovertemplate='Close: ₹%{y:.2f}<extra></extra>'), row=1, col=1)
    fig2.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='#8B008B', width=2), hovertemplate='RSI: %{y:.2f}<extra></extra>'), row=2, col=1)
    fig2.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
    fig2.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
    fig2.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig2.update_yaxes(title_text="RSI", row=2, col=1)
    fig2.update_xaxes(title_text="Date", row=2, col=1)
    fig2.update_layout(hovermode='x unified', height=700)
    charts.append(fig2)

    # Chart 3: Price with MFI
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(f'{symbol} - Price', 'MFI (14)'), row_heights=[0.6, 0.4])
    fig3.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='#2D3748', width=2), hovertemplate='Close: ₹%{y:.2f}<extra></extra>'), row=1, col=1)
    fig3.add_trace(go.Scatter(x=data.index, y=data['MFI'], name='MFI', line=dict(color='#FF8C00', width=2), hovertemplate='MFI: %{y:.2f}<extra></extra>'), row=2, col=1)
    fig3.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
    fig3.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
    fig3.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig3.update_yaxes(title_text="MFI", row=2, col=1)
    fig3.update_xaxes(title_text="Date", row=2, col=1)
    fig3.update_layout(hovermode='x unified', height=700)
    charts.append(fig3)

    # Chart 4: Price with EMV
    fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(f'{symbol} - Price', 'EMV (14)'), row_heights=[0.6, 0.4])
    fig4.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='#2D3748', width=2), hovertemplate='Close: ₹%{y:.2f}<extra></extra>'), row=1, col=1)
    fig4.add_trace(go.Scatter(x=data.index, y=data['EMV'], name='EMV', line=dict(color='#228B22', width=2), hovertemplate='EMV: %{y:.4f}<extra></extra>'), row=2, col=1)
    fig4.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7, row=2, col=1)
    fig4.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig4.update_yaxes(title_text="EMV", row=2, col=1)
    fig4.update_xaxes(title_text="Date", row=2, col=1)
    fig4.update_layout(hovermode='x unified', height=700)
    charts.append(fig4)

    # Chart 5: Price with ATR
    fig5 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(f'{symbol} - Price', 'ATR (14)'), row_heights=[0.6, 0.4])
    fig5.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='#2D3748', width=2), hovertemplate='Close: ₹%{y:.2f}<extra></extra>'), row=1, col=1)
    fig5.add_trace(go.Scatter(x=data.index, y=data['ATR'], name='ATR', line=dict(color='#DC143C', width=2), hovertemplate='ATR: %{y:.2f}<extra></extra>'), row=2, col=1)
    fig5.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig5.update_yaxes(title_text="ATR", row=2, col=1)
    fig5.update_xaxes(title_text="Date", row=2, col=1)
    fig5.update_layout(hovermode='x unified', height=700)
    charts.append(fig5)

    # Chart 6: Price with Mean Deviation
    fig6 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(f'{symbol} - Price', 'Mean Deviation (20)'), row_heights=[0.6, 0.4])
    fig6.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='#2D3748', width=2), hovertemplate='Close: ₹%{y:.2f}<extra></extra>'), row=1, col=1)
    fig6.add_trace(go.Scatter(x=data.index, y=data['MD'], name='Mean Deviation', line=dict(color='#9333EA', width=2), hovertemplate='MD: %{y:.2f}<extra></extra>'), row=2, col=1)
    fig6.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig6.update_yaxes(title_text="MD", row=2, col=1)
    fig6.update_xaxes(title_text="Date", row=2, col=1)
    fig6.update_layout(hovermode='x unified', height=700)
    charts.append(fig6)

    # Chart 7: Price with CCI
    fig7 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(f'{symbol} - Price', 'CCI (40)'), row_heights=[0.6, 0.4])
    fig7.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='#2D3748', width=2), hovertemplate='Close: ₹%{y:.2f}<extra></extra>'), row=1, col=1)
    fig7.add_trace(go.Scatter(x=data.index, y=data['CCI'], name='CCI', line=dict(color='#059669', width=2), hovertemplate='CCI: %{y:.2f}<extra></extra>'), row=2, col=1)
    fig7.add_hline(y=100, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
    fig7.add_hline(y=-100, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
    fig7.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5, row=2, col=1)
    fig7.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig7.update_yaxes(title_text="CCI", row=2, col=1)
    fig7.update_xaxes(title_text="Date", row=2, col=1)
    fig7.update_layout(hovermode='x unified', height=700)
    charts.append(fig7)

    # Chart 8: Price with MACD
    fig8 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} - Price', 'MACD (12,26,9)'),
        row_heights=[0.6, 0.4]
    )

    fig8.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            name='Close Price',
            line=dict(color='#2D3748', width=2)
        ),
        row=1, col=1
    )

    fig8.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MACD'],
            name='MACD',
            line=dict(color='#2563EB', width=2)
        ),
        row=2, col=1
    )

    fig8.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MACD_signal'],
            name='Signal',
            line=dict(color='#DC2626', width=2)
        ),
        row=2, col=1
    )

    fig8.add_trace(
        go.Bar(
            x=data.index,
            y=data['MACD_hist'],
            name='Histogram'
        ),
        row=2, col=1
    )

    fig8.update_layout(hovermode='x unified', height=700)
    charts.append(fig8)

    return charts, data

def display_indicator_summary(data, symbol):
    """Display latest indicator values in a formatted box"""
    try:
        latest_rsi = data['RSI'].dropna().iloc[-1] if not data['RSI'].dropna().empty else None
        latest_mfi = data['MFI'].dropna().iloc[-1] if not data['MFI'].dropna().empty else None
        latest_atr = data['ATR'].dropna().iloc[-1] if not data['ATR'].dropna().empty else None
        latest_md = data['MD'].dropna().iloc[-1] if not data['MD'].dropna().empty else None
        latest_cci = data['CCI'].dropna().iloc[-1] if not data['CCI'].dropna().empty else None
        latest_emv = data['EMV'].dropna().iloc[-1] if not data['EMV'].dropna().empty else None
        latest_sma_20 = data['SMA_20'].dropna().iloc[-1] if not data['SMA_20'].dropna().empty else None
        latest_sma_50 = data['SMA_50'].dropna().iloc[-1] if not data['SMA_50'].dropna().empty else None
        latest_macd = data['MACD'].dropna().iloc[-1] if not data['MACD'].dropna().empty else None

        close_val = data['Close'].iloc[-1]
        if isinstance(close_val, pd.Series):
            latest_close = close_val.values[0]
        else:
            latest_close = close_val

        latest_date = data.index[-1].strftime('%Y-%m-%d')
    except Exception as e:
        st.error(f"Error extracting indicator values: {e}")
        return

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    ">
        <h3 style="color: white; margin: 0 0 15px 0; text-align: center;">📊 {symbol} - Latest Technical Indicators</h3>
        <p style="color: white; text-align: center; font-style: italic;">As of: {latest_date}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Price & Moving Averages")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <p style="margin: 5px 0; font-weight: bold; color: #333;">Close Price</p>
            <p style="margin: 5px 0; font-size: 24px; color: #2196F3; font-weight: bold;">₹{latest_close:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if latest_sma_20 is not None:
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 5px 0; font-weight: bold; color: #333;">SMA 20</p>
                <p style="margin: 5px 0; font-size: 24px; color: #FF8C00; font-weight: bold;">₹{latest_sma_20:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        if latest_sma_50 is not None:
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 5px 0; font-weight: bold; color: #333;">SMA 50</p>
                <p style="margin: 5px 0; font-size: 24px; color: #4169E1; font-weight: bold;">₹{latest_sma_50:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        if latest_atr is not None:
            atr_pct = (latest_atr / latest_close) * 100
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 5px 0; font-weight: bold; color: #333;">ATR (14)</p>
                <p style="margin: 5px 0; font-size: 20px; color: #9C27B0; font-weight: bold;">{latest_atr:.2f}</p>
                <p style="margin: 5px 0; color: #9C27B0; font-size: 14px;">{atr_pct:.2f}% of price</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Momentum Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if latest_rsi is not None:
            rsi_color = "#dc2626" if latest_rsi > 70 else ("#16a34a" if latest_rsi < 30 else "#0f172a")
            rsi_status = "Overbought" if latest_rsi > 70 else ("Oversold" if latest_rsi < 30 else "Neutral")
            st.markdown(f"""
            <div style="background: white; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 2px 0; font-weight: bold; color: #333; font-size: 11px;">RSI (14)</p>
                <p style="margin: 5px 0; font-size: 18px; color: {rsi_color}; font-weight: bold;">{latest_rsi:.2f}</p>
                <p style="margin: 2px 0; color: {rsi_color}; font-size: 12px; font-weight: 600;">{rsi_status}</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        if latest_mfi is not None:
            mfi_color = "#dc2626" if latest_mfi > 80 else ("#16a34a" if latest_mfi < 20 else "#0f172a")
            mfi_status = "Overbought" if latest_mfi > 80 else ("Oversold" if latest_mfi < 20 else "Neutral")
            st.markdown(f"""
            <div style="background: white; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 2px 0; font-weight: bold; color: #333; font-size: 11px;">MFI (14)</p>
                <p style="margin: 5px 0; font-size: 18px; color: {mfi_color}; font-weight: bold;">{latest_mfi:.2f}</p>
                <p style="margin: 2px 0; color: {mfi_color}; font-size: 12px; font-weight: 600;">{mfi_status}</p>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        if latest_cci is not None:
            cci_color = "#dc2626" if latest_cci > 100 else ("#16a34a" if latest_cci < -100 else "#0f172a")
            cci_status = "Overbought" if latest_cci > 100 else ("Oversold" if latest_cci < -100 else "Neutral")
            st.markdown(f"""
            <div style="background: white; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 2px 0; font-weight: bold; color: #333; font-size: 11px;">CCI (40)</p>
                <p style="margin: 5px 0; font-size: 18px; color: {cci_color}; font-weight: bold;">{latest_cci:.2f}</p>
                <p style="margin: 2px 0; color: {cci_color}; font-size: 12px; font-weight: 600;">{cci_status}</p>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        if latest_emv is not None:
            emv_color = "#16a34a" if latest_emv > 0 else ("#dc2626" if latest_emv < 0 else "#64748b")
            emv_status = "Positive" if latest_emv > 0 else ("Negative" if latest_emv < 0 else "Neutral")
            st.markdown(f"""
            <div style="background: white; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 2px 0; font-weight: bold; color: #333; font-size: 11px;">EMV (14)</p>
                <p style="margin: 5px 0; font-size: 18px; color: {emv_color}; font-weight: bold;">{latest_emv:.2f}</p>
                <p style="margin: 2px 0; color: {emv_color}; font-size: 12px; font-weight: 600;">{emv_status}</p>
            </div>
            """, unsafe_allow_html=True)

    with col5:
        if latest_macd is not None:
            macd_color = "#16a34a" if latest_macd > 0 else "#dc2626"
            macd_status = "Bullish" if latest_macd > 0 else "Bearish"
            st.markdown(f"""
            <div style="background: white; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 2px 0; font-weight: bold; color: #333; font-size: 11px;">MACD</p>
                <p style="margin: 5px 0; font-size: 18px; color: {macd_color}; font-weight: bold;">{latest_macd:.2f}</p>
                <p style="margin: 2px 0; color: {macd_color}; font-size: 12px; font-weight: 600;">{macd_status}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    return {
        'close': latest_close,
        'sma_20': latest_sma_20,
        'sma_50': latest_sma_50,
        'rsi': latest_rsi,
        'mfi': latest_mfi,
        'cci': latest_cci,
        'atr': latest_atr,
        'md': latest_md,
        'emv': latest_emv,
        'macd': latest_macd
    }

def get_technical_indicators(etf_row):
    """Helper function to get technical indicators for an ETF"""
    yf_ticker = str(etf_row.get('yfinance_ticker', '')).strip()

    if not yf_ticker:
        return None

    try:
        etf_data = fetch_etf_data(yf_ticker)
        if etf_data is not None and not etf_data.empty:
            etf_data['SMA_20'] = etf_data['Close'].rolling(window=20).mean()
            etf_data['SMA_50'] = etf_data['Close'].rolling(window=50).mean()
            etf_data['RSI'] = calculate_rsi(etf_data)
            etf_data['MFI'] = calculate_mfi(etf_data['High'], etf_data['Low'], etf_data['Close'], etf_data['Volume'])
            etf_data['ATR'] = calculate_atr(etf_data['High'], etf_data['Low'], etf_data['Close'])
            etf_data['CCI'] = calculate_cci(etf_data['High'], etf_data['Low'], etf_data['Close'], period=40)
            etf_data['MD'] = calculate_mean_deviation(etf_data['High'], etf_data['Low'], etf_data['Close'], period=20)
            etf_data['EMV'] = calculate_emv(etf_data)

            return {
                'sma_20': etf_data['SMA_20'].dropna().iloc[-1] if not etf_data['SMA_20'].dropna().empty else None,
                'sma_50': etf_data['SMA_50'].dropna().iloc[-1] if not etf_data['SMA_50'].dropna().empty else None,
                'rsi': etf_data['RSI'].dropna().iloc[-1] if not etf_data['RSI'].dropna().empty else None,
                'mfi': etf_data['MFI'].dropna().iloc[-1] if not etf_data['MFI'].dropna().empty else None,
                'cci': etf_data['CCI'].dropna().iloc[-1] if not etf_data['CCI'].dropna().empty else None,
                'atr': etf_data['ATR'].dropna().iloc[-1] if not etf_data['ATR'].dropna().empty else None,
                'md': etf_data['MD'].dropna().iloc[-1] if not etf_data['MD'].dropna().empty else None,
                'emv': etf_data['EMV'].dropna().iloc[-1] if not etf_data['EMV'].dropna().empty else None,
            }
    except:
        return None

# ============================================================
# GLOBAL CSS STYLING - PROFESSIONAL GREY THEME
# ============================================================
st.markdown("""
<style>
.scrolling-container {
    width: 100%;
    overflow: hidden;
    height: 45px;
    position: relative;
    background: linear-gradient(90deg, #4A5568 0%, #2D3748 100%);
    border-radius: 8px;
    margin-bottom: 20px;
}
.scrolling-text {
    position: absolute;
    white-space: nowrap;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    font-size: 24px;
    font-weight: 600;
    color: white;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    animation: scroll-left 15s linear infinite;
}
@keyframes scroll-left {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}
/* Mobile sidebar hint - only visible on mobile */
.mobile-sidebar-hint {
    display: none;
    position: fixed;
    top: 60px;
    left: 10px;
    background: #FEF3C7;
    color: #92400E;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    z-index: 999;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    border: 1px solid #F59E0B;
}

@media (max-width: 768px) {
    .mobile-sidebar-hint {
        display: block;
    }
}
.stApp {
    background-color: #F7FAFC;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif !important;
}

.stButton>button {
    background: linear-gradient(135deg, #4A5568 0%, #2D3748 100%);
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    background: linear-gradient(135deg, #2D3748 0%, #1A202C 100%);
}

/* Custom Tab Styling - Make tabs more vibrant */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #EDF2F7;
    padding: 8px;
    border-radius: 10px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    background-color: white;
    border-radius: 8px;
    padding: 0px 24px;
    font-weight: 600;
    font-size: 15px;
    color: #4A5568;
    border: 2px solid transparent;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #E2E8F0;
    color: #2D3748;
    transform: translateY(-2px);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    color: white !important;
    border: 2px solid #1E40AF !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

/* Details and Technical Analysis tabs - special styling */
.stTabs [data-baseweb="tab"]:nth-child(3),
.stTabs [data-baseweb="tab"]:nth-child(4) {
    background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
}

.stTabs [data-baseweb="tab"]:nth-child(3):hover,
.stTabs [data-baseweb="tab"]:nth-child(4):hover {
    background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%);
    color: #1E40AF;
}

/* Tab panel content */
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 20px;
}
/* Global Search Bar */
.global-search-wrapper {
    position: relative;
    margin-bottom: 20px;
}
.search-result-count {
    font-size: 12px;
    color: #718096;
    margin-top: 4px;
    text-align: right;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'show_comparison' not in st.session_state:
    st.session_state.show_comparison = False
if 'compare_etf_1' not in st.session_state:
    st.session_state.compare_etf_1 = ""
if 'compare_etf_2' not in st.session_state:
    st.session_state.compare_etf_2 = ""

# ============================================================
# TOP SCROLLING MESSAGE
# ============================================================
st.markdown("""
<div class="scrolling-container">
    <div class="scrolling-text">
        ✨ WORK IN PROGRESS - THANK YOU FOR YOUR PATIENCE ✨
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="mobile-sidebar-hint">
    👉 Slide to choose ETF
</div>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #2D3748 0%, #1A202C 100%);
    padding: 35px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
">
    <h1 style="color: white; margin: 0; font-size: 38px; font-weight: 700;">
          Indian ETF Tracker
    </h1>
    <p style="color: #E2E8F0; margin: 10px 0 0 0; font-size: 16px; text-align: center; font-style: italic;">
        let's invest in passives, actively
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div style="
    background: #EDF2F7;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #4A5568;
    margin-bottom: 25px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
">
    <p style="color: #2D3748; margin: 0; font-weight: 600; font-size: 14px;">
        ℹ️ Detailed holdings and technical analysis available for selected ETFs
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
FILE = "etf_master_new.csv"
HOLDINGS_FILE = "holding_analysis.csv"

if not os.path.exists(FILE):
    st.error("❌ File etf_master_new.csv not found")
    st.stop()

df = pd.read_csv(FILE, dtype=str)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
# ============================================================
# LOAD HOLDINGS DATA
# ============================================================
holdings_dict = {}
marketcap_dict = {}

if os.path.exists(HOLDINGS_FILE):
    try:
        df_holdings = pd.read_csv(HOLDINGS_FILE)
        df_holdings.columns = df_holdings.columns.str.strip()

        for isin_code, group in df_holdings.groupby('isin_code'):
            if pd.isna(isin_code) or str(isin_code).strip() == '':
                continue

            first_row = group.iloc[0]
            etf_name = str(first_row.get('ETF', '')).strip()

            if not etf_name or etf_name.lower() == 'nan':
                continue

            holdings_data = {}
            holdings_data['ISIN CODE'] = str(isin_code).strip()
            holdings_data['ETF'] = etf_name
            holdings_data['NSE TICKER'] = str(first_row.get('SYMBOL', '')).strip()

            valid_holdings = group[
                (group['Holding'].notna()) &
                (group['Holding'] != '') &
                (group['Amount'].notna()) &
                (group['Amount'] != '')
            ].copy()

            valid_holdings['Amount'] = pd.to_numeric(valid_holdings['Amount'], errors='coerce')
            valid_holdings = valid_holdings[valid_holdings['Amount'].notna()]
            valid_holdings = valid_holdings.sort_values('Amount', ascending=False).head(10)

            total_weight = 0
            for idx, row in valid_holdings.iterrows():
                holding_name = str(row['Holding']).strip()

                if not holding_name or holding_name.lower() == 'nan':
                    continue

                try:
                    amount = float(row['Amount'])

                    if amount <= 0:
                        continue

                    holdings_data[holding_name] = str(amount)
                    total_weight += amount
                except (ValueError, TypeError):
                    continue

            if total_weight > 0:
                holdings_data['Total'] = str(round(total_weight, 2))

            holdings_dict[str(isin_code).strip()] = holdings_data

            marketcap_data = {}
            for idx, row in group.iterrows():
                row_list = row.tolist()
                if len(row_list) >= 12:
                    market_cap_cat = str(row_list[10]).strip() if pd.notna(row_list[10]) and str(row_list[10]).strip() else None
                    market_cap_pct = str(row_list[11]).strip() if pd.notna(row_list[11]) and str(row_list[11]).strip() else None

                    if market_cap_cat and market_cap_pct and market_cap_cat in ['Large Cap', 'Mid Cap', 'Small Cap']:
                        try:
                            marketcap_data[market_cap_cat] = float(market_cap_pct)
                        except (ValueError, TypeError):
                            pass

            if marketcap_data:
                marketcap_dict[str(isin_code).strip()] = marketcap_data

    except Exception as e:
        st.warning(f"⚠️ Could not load holdings data: {e}")

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
# LOAD PRICE DATA
# ============================================================
PRICE_FILE = "nse_etf_prices.csv"
df_price = pd.DataFrame()
latest_price_date = None
price_file_mtime = None

if os.path.exists(PRICE_FILE):
    price_file_mtime = os.path.getmtime(PRICE_FILE)

    try:
        df_price = pd.read_csv(PRICE_FILE)
        df_price.columns = df_price.columns.str.strip().str.lower().str.replace(" ", "_")

        date_col = next((c for c in df_price.columns if 'date' in c), None)

        if date_col:
            df_price["date"] = pd.to_datetime(
                df_price[date_col],
                format="%d-%m-%Y",
                errors="coerce"
            )
            if not df_price.empty and "date" in df_price.columns:
                latest_price_date = df_price["date"].max()
                if pd.isna(latest_price_date):
                    latest_price_date = None

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
    except Exception as e:
        st.warning(f"⚠️ Could not load price data: {e}")
        df_price = pd.DataFrame()

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
    "Nifty 100": r"nifty\s*100\b(?!.*(?:value|equal\s*weight|quality|momentum|alpha|low\s*volatility|factor|esg))",
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
    "FINANCIAL SERVICES": r"financial.*services\b(?!.*ex.*bank)",
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
    "India Consumption": r"india.*consumption",
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
    "New Age Consumption": r"new.*age.*consum",
    "ESG SECTOR LEADERS": r"\besg\b",
    "Tourism": r"tourism",
    "India Infrastructure": r"india.*infrastructure",
    "Nifty 50 shariah": r"nifty.*50.*shariah",
    "CPSE": r"\bcpse\b",
    "PSU BANK": r"psu.*bank",
    "Digital": r"digital",
    "Service Sector": r"service sector"
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
# FILTER UI - SIDEBAR
# ============================================================
# Initialize session state for active filter section
if 'active_filter_section' not in st.session_state:
    st.session_state.active_filter_section = None

with st.sidebar:
    st.markdown("""
    <div style="
        background: #2D3748;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    ">
        <h3 style="color: white; margin: 0; text-align: center; font-size: 18px;">🔍 Filter ETFs</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Helper info
    st.markdown("""
    <div style="
        background: #EDF2F7;
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 15px;
    ">
        <p style="color: #4A5568; margin: 0; font-size: 11px; font-weight: 500;">
            💡 Tip: Only one filter type can be active at a time
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Reset all filters button
    if st.button("🔄 Reset All Filters", width='stretch'):
        st.session_state.active_filter_section = None
        for key in list(st.session_state.keys()):
            if 'te_range' in key:
                del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    
    # ==================== ASSET CLASS FILTER SECTION ====================
    # Determine if this section should be disabled
    asset_section_disabled = (st.session_state.active_filter_section == 'tracking_error')
    
    if asset_section_disabled:
        st.markdown("""
        <div style="
            background: #F1F5F9;
            padding: 12px;
            border-radius: 8px;
            opacity: 0.6;
            margin-bottom: 15px;
        ">
            <p style="color: #64748B; margin: 0; font-size: 12px; font-weight: 600; text-align: center;">
                🔒 Asset Class Filters (Disabled)
            </p>
            <p style="color: #64748B; margin: 5px 0 0 0; font-size: 10px; text-align: center;">
                Tracking Error filter is active
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background: #2D3748;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
        ">
            <p style="color: white; margin: 0; font-size: 14px; font-weight: 600; text-align: center;">
                📊 Asset Class Filters
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    amc_list = ["All"] + sorted(df["amc"].dropna().unique())
    selected_amc = st.selectbox("Select AMC", amc_list, disabled=asset_section_disabled)

    selected_asset = st.selectbox(
        "Select Asset Class",
        ["All", "EQUITY", "GLOBAL EQUITY", "DEBT", "COMMODITIES"],
        disabled=asset_section_disabled
    )
    
    # Track if asset filters are being used
    if not asset_section_disabled and (selected_amc != "All" or selected_asset != "All"):
        st.session_state.active_filter_section = 'asset_class'

    sub_cat = None
    sub_sub_cat = None
    lookup = None

    if selected_asset == "EQUITY" and not asset_section_disabled:
        sub_cat = st.selectbox(
            "Select Equity Category",
            ["Broader", "Sectoral", "Thematic", "Strategic"],
            disabled=asset_section_disabled
        )

        lookup = {
            "Broader": BROADER,
            "Sectoral": SECTORAL,
            "Thematic": THEMATIC,
            "Strategic": STRATEGIC
        }[sub_cat]

        sub_sub_cat = st.selectbox(
            f"Select {sub_cat}",
            ["All"] + sorted(lookup.keys()),
            disabled=asset_section_disabled
        )

    elif selected_asset == "DEBT" and not asset_section_disabled:
        sub_cat = st.selectbox(
            "Select Debt Category",
            ["All", "Bharat Bond", "G-Sec", "Gilt", "Liquid", "SDL"],
            disabled=asset_section_disabled
        )

    elif selected_asset == "COMMODITIES" and not asset_section_disabled:
        sub_cat = st.selectbox(
            "Select Commodity",
            ["All", "Gold", "Silver"],
            disabled=asset_section_disabled
        )
    # ============================================================
    # ETF COMPARISON SELECTOR (SIDEBAR)
    # ============================================================
    st.markdown("---")
    st.markdown("""
    <div style="
        background: #2D3748;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        margin-top: 20px;
    ">
        <h3 style="color: white; margin: 0; text-align: center; font-size: 18px;">⚖️ Compare ETFs</h3>
    </div>
    """, unsafe_allow_html=True)

    all_etf_names = sorted(df["etf"].dropna().unique().tolist())

    st.markdown("**First ETF:**")
    search_term_1 = st.text_input(
        "Type to search First ETF",
        value=st.session_state.compare_etf_1,
        key="search_etf_1",
        placeholder="Start typing ETF name...",
        label_visibility="collapsed"
    )

    if search_term_1:
        filtered_etfs_1 = [etf for etf in all_etf_names if search_term_1.lower() in etf.lower()]
        if filtered_etfs_1:
            st.session_state.compare_etf_1 = st.selectbox(
                "Select from matches",
                filtered_etfs_1,
                key="filtered_select_1",
                label_visibility="collapsed"
            )
        else:
            st.warning("No matching ETFs found")
            st.session_state.compare_etf_1 = ""
    else:
        st.session_state.compare_etf_1 = ""

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**Second ETF:**")
    search_term_2 = st.text_input(
        "Type to search Second ETF",
        value=st.session_state.compare_etf_2,
        key="search_etf_2",
        placeholder="Start typing ETF name...",
        label_visibility="collapsed"
    )

    if search_term_2:
        filtered_etfs_2 = [etf for etf in all_etf_names if search_term_2.lower() in etf.lower()]
        if filtered_etfs_2:
            st.session_state.compare_etf_2 = st.selectbox(
                "Select from matches",
                filtered_etfs_2,
                key="filtered_select_2",
                label_visibility="collapsed"
            )
        else:
            st.warning("No matching ETFs found")
            st.session_state.compare_etf_2 = ""
    else:
        st.session_state.compare_etf_2 = ""

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.compare_etf_1 and st.session_state.compare_etf_2:
        col_open, col_close = st.columns(2)
        with col_open:
            if st.button("🔍 Open Comparison", width='stretch'):
                st.session_state.show_comparison = True
                st.rerun()
        with col_close:
            if st.session_state.show_comparison:
                if st.button("❌ Close", width='stretch'):
                    st.session_state.show_comparison = False
                    st.rerun()

    # ============================================================
    # FILTER BY TRACKING ERROR (SIDEBAR)
    # ============================================================
    
    # ============================================================
    # FILTER BY TRACKING ERROR (SIDEBAR)
    # ============================================================
    st.markdown("---")
    
    # Determine if this section should be disabled
    te_section_disabled = (st.session_state.active_filter_section == 'asset_class')

    if te_section_disabled:
        st.markdown("""
        <div style="
            background: #F1F5F9;
            padding: 12px;
            border-radius: 8px;
            opacity: 0.6;
            margin-bottom: 15px;
        ">
            <p style="color: #64748B; margin: 0; font-size: 12px; font-weight: 600; text-align: center;">
                🔒 Tracking Error Filter (Disabled)
            </p>
            <p style="color: #64748B; margin: 5px 0 0 0; font-size: 10px; text-align: center;">
                Asset Class filter is active
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background: #2D3748;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
        ">
            <p style="color: white; margin: 0; font-size: 14px; font-weight: 600; text-align: center;">
                📊 Tracking Error Filter
            </p>
        </div>
        """, unsafe_allow_html=True)

    te_ranges = {
        "0.00 to 0.05": (0.00, 0.05),
        "0.06 to 0.10": (0.06, 0.10),
        "0.11 to 0.20": (0.11, 0.20),
        "0.21 to 0.40": (0.21, 0.40),
        "0.41 to 0.60": (0.41, 0.60),
        "0.61 to 1.00": (0.61, 1.00),
        "1.01+": (1.01, float('inf'))
    }
    
    if not te_section_disabled:
        st.markdown("**Select Tracking Error Ranges:**")
        st.markdown("<small style='color: #94A3B8;'>Multiple ranges can be selected</small>", unsafe_allow_html=True)

    # Create checkboxes for each range
    selected_te_ranges = []
    for range_label in te_ranges.keys():
        if st.checkbox(range_label, key=f"te_range_{range_label}", disabled=te_section_disabled):
            selected_te_ranges.append(range_label)
    
    # Track if TE filters are being used
    if selected_te_ranges and not te_section_disabled:
        st.session_state.active_filter_section = 'tracking_error'

    # Display selected count if any
    if selected_te_ranges and not te_section_disabled:
        st.markdown(f"""
        <div style="
            background: #DBEAFE;
            padding: 8px;
            border-radius: 6px;
            margin-top: 10px;
            text-align: center;
        ">
            <p style="margin: 0; font-size: 12px; color: #1E40AF; font-weight: 600;">
                ✓ {len(selected_te_ranges)} range(s) selected
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    
# ============================================================
# COMPARISON MODAL/WINDOW (Full Width - Main Area)
# ============================================================
@st.dialog("ETF Comparison", width="large")
def show_comparison_modal():
    compare_etf_1 = st.session_state.compare_etf_1
    compare_etf_2 = st.session_state.compare_etf_2

    etf1_data = df[df["etf"] == compare_etf_1].iloc[0]
    etf2_data = df[df["etf"] == compare_etf_2].iloc[0]

    benchmark1 = str(etf1_data.get("benchmark_index", "")).strip()
    benchmark2 = str(etf2_data.get("benchmark_index", "")).strip()

    if benchmark1.lower() == benchmark2.lower():
        st.markdown("""
        <div style="
            background: #D4EDDA;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #28A745;
            margin-bottom: 20px;
        ">
            <p style="color: #155724; margin: 0; font-weight: 600; font-size: 14px;">
                ✅ These funds are from the same benchmark: <strong>{}</strong>
            </p>
        </div>
        """.format(benchmark1), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background: #FFF3CD;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #FFC107;
            margin-bottom: 20px;
        ">
            <p style="color: #856404; margin: 0; font-weight: 600; font-size: 14px;">
                ⚠️ These funds are not from the same benchmark
            </p>
            <p style="color: #856404; margin: 5px 0 0 0; font-size: 13px;">
                <strong>{}</strong>: {}<br>
                <strong>{}</strong>: {}
            </p>
        </div>
        """.format(compare_etf_1, benchmark1, compare_etf_2, benchmark2), unsafe_allow_html=True)

    def get_price(etf_row):
        ticker = str(etf_row.get("symbol", "")).strip().upper()
        if not df_price.empty:
            price_row = df_price[df_price["symbol"] == ticker]
            if not price_row.empty:
                ltp_val = pd.to_numeric(price_row.iloc[0]["ltp"], errors="coerce")
                if not pd.isna(ltp_val):
                    return ltp_val
            else:
                if ticker:
                    alt_matches = df_price[df_price["symbol"].str.contains(ticker[:5], na=False, case=False)]
                    if not alt_matches.empty:
                        ltp_val = pd.to_numeric(alt_matches.iloc[0]["ltp"], errors="coerce")
                        if not pd.isna(ltp_val):
                            return ltp_val
        return None

    col1, col2 = st.columns(2)

    with st.spinner("Fetching technical indicators..."):
        tech1 = get_technical_indicators(etf1_data)
        tech2 = get_technical_indicators(etf2_data)

    price1 = get_price(etf1_data)
    price2 = get_price(etf2_data)

    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #64748b 0%, #475569 100%);
            padding: 25px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            margin-bottom: 20px;
            border: 1px solid #cbd5e1;
        ">
            <h3 style="color: #f1f5f9; margin: 0; text-align: center; font-size: 16px; font-weight: 600; letter-spacing: 0.5px;">{compare_etf_1}</h3>
        </div>
        """, unsafe_allow_html=True)

        if price1:
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">CURRENT PRICE</p>
                <p style="margin: 8px 0 0 0; font-size: 28px; color: #1e40af; font-weight: 800;">₹{price1:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">CURRENT PRICE</p>
                <p style="margin: 8px 0 0 0; font-size: 18px; color: #94a3b8; font-weight: 600;">N/A</p>
            </div>
            """, unsafe_allow_html=True)

        aum1 = etf1_data.get('aum', 'N/A')
        st.markdown(f"""
        <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
            <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">AUM</p>
            <p style="margin: 8px 0 0 0; font-size: 24px; color: #0f172a; font-weight: 800;">₹{aum1} Cr</p>
        </div>
        """, unsafe_allow_html=True)

        years1 = etf1_data.get('years_active', 'N/A')
        st.markdown(f"""
        <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
            <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">YEARS ACTIVE</p>
            <p style="margin: 8px 0 0 0; font-size: 24px; color: #0f172a; font-weight: 800;">{years1}</p>
        </div>
        """, unsafe_allow_html=True)

        if tech1 and tech1.get("rsi") is not None:
            rsi1 = tech1["rsi"]
            rsi_color = "#dc2626" if rsi1 > 70 else ("#16a34a" if rsi1 < 30 else "#0f172a")
            rsi_status = "Overbought" if rsi1 > 70 else ("Oversold" if rsi1 < 30 else "Neutral")
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">RSI (14)</p>
                <p style="margin: 8px 0 0 0; font-size: 24px; color: {rsi_color}; font-weight: 800;">{rsi1:.2f}</p>
                <p style="margin: 6px 0 0 0; color: {rsi_color}; font-size: 13px; font-weight: 600;">{rsi_status}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">RSI (14)</p>
                <p style="margin: 8px 0 0 0; font-size: 18px; color: #94a3b8; font-weight: 600;">N/A</p>
            </div>
            """, unsafe_allow_html=True)

        if tech1 and tech1.get("mfi") is not None:
            mfi1 = tech1["mfi"]
            mfi_color = "#dc2626" if mfi1 > 80 else ("#16a34a" if mfi1 < 20 else "#0f172a")
            mfi_status = "Overbought" if mfi1 > 80 else ("Oversold" if mfi1 < 20 else "Neutral")
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">MFI (14)</p>
                <p style="margin: 8px 0 0 0; font-size: 24px; color: {mfi_color}; font-weight: 800;">{mfi1:.2f}</p>
                <p style="margin: 6px 0 0 0; color: {mfi_color}; font-size: 13px; font-weight: 600;">{mfi_status}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">MFI (14)</p>
                <p style="margin: 8px 0 0 0; font-size: 18px; color: #94a3b8; font-weight: 600;">N/A</p>
            </div>
            """, unsafe_allow_html=True)

        te1 = etf1_data.get('overall_tracking_error', 'N/A')
        st.markdown(f"""
        <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
            <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">TRACKING ERROR</p>
            <p style="margin: 8px 0 0 0; font-size: 24px; color: #0f172a; font-weight: 800;">{te1}</p>
        </div>
        """, unsafe_allow_html=True)

        td1 = etf1_data.get('overall_tracking_difference', 'N/A')
        st.markdown(f"""
        <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
            <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">TRACKING DIFFERENCE</p>
            <p style="margin: 8px 0 0 0; font-size: 24px; color: #0f172a; font-weight: 800;">{td1}</p>
        </div>
        """, unsafe_allow_html=True)

        if tech1 and tech1.get("cci") is not None:
            cci1 = tech1["cci"]
            cci_color = "#dc2626" if cci1 > 100 else ("#16a34a" if cci1 < -100 else "#0f172a")
            cci_status = "Overbought" if cci1 > 100 else ("Oversold" if cci1 < -100 else "Neutral")
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">CCI (40)</p>
                <p style="margin: 8px 0 0 0; font-size: 24px; color: {cci_color}; font-weight: 800;">{cci1:.2f}</p>
                <p style="margin: 6px 0 0 0; color: {cci_color}; font-size: 13px; font-weight: 600;">{cci_status}</p>
            </div>
            """, unsafe_allow_html=True)

        if tech1 and tech1.get("sma_20") is not None:
            sma20 = tech1["sma_20"]
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">SMA 20</p>
                <p style="margin: 8px 0 0 0; font-size: 24px; color: #0f172a; font-weight: 800;">₹{sma20:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        if tech1 and tech1.get("sma_50") is not None:
            sma50 = tech1["sma_50"]
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">SMA 50</p>
                <p style="margin: 8px 0 0 0; font-size: 24px; color: #0f172a; font-weight: 800;">₹{sma50:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #64748b 0%, #475569 100%);
            padding: 25px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            margin-bottom: 20px;
            border: 1px solid #cbd5e1;
        ">
            <h3 style="color: #f1f5f9; margin: 0; text-align: center; font-size: 16px; font-weight: 600; letter-spacing: 0.5px;">{compare_etf_2}</h3>
        </div>
        """, unsafe_allow_html=True)

        if price2:
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">CURRENT PRICE</p>
                <p style="margin: 8px 0 0 0; font-size: 28px; color: #1e40af; font-weight: 800;">₹{price2:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">CURRENT PRICE</p>
                <p style="margin: 8px 0 0 0; font-size: 18px; color: #94a3b8; font-weight: 600;">N/A</p>
            </div>
            """, unsafe_allow_html=True)

        aum2 = etf2_data.get('aum', 'N/A')
        st.markdown(f"""
        <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
            <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">AUM</p>
            <p style="margin: 8px 0 0 0; font-size: 24px; color: #0f172a; font-weight: 800;">₹{aum2} Cr</p>
        </div>
        """, unsafe_allow_html=True)

        years2 = etf2_data.get('years_active', 'N/A')
        st.markdown(f"""
        <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
            <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">YEARS ACTIVE</p>
            <p style="margin: 8px 0 0 0; font-size: 24px; color: #0f172a; font-weight: 800;">{years2}</p>
        </div>
        """, unsafe_allow_html=True)

        if tech2 and tech2.get("rsi") is not None:
            rsi2 = tech2["rsi"]
            rsi_color = "#dc2626" if rsi2 > 70 else ("#16a34a" if rsi2 < 30 else "#0f172a")
            rsi_status = "Overbought" if rsi2 > 70 else ("Oversold" if rsi2 < 30 else "Neutral")
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">RSI (14)</p>
                <p style="margin: 8px 0 0 0; font-size: 24px; color: {rsi_color}; font-weight: 800;">{rsi2:.2f}</p>
                <p style="margin: 6px 0 0 0; color: {rsi_color}; font-size: 13px; font-weight: 600;">{rsi_status}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">RSI (14)</p>
                <p style="margin: 8px 0 0 0; font-size: 18px; color: #94a3b8; font-weight: 600;">N/A</p>
            </div>
            """, unsafe_allow_html=True)

        if tech2 and tech2.get("mfi") is not None:
            mfi2 = tech2["mfi"]
            mfi_color = "#dc2626" if mfi2 > 80 else ("#16a34a" if mfi2 < 20 else "#0f172a")
            mfi_status = "Overbought" if mfi2 > 80 else ("Oversold" if mfi2 < 20 else "Neutral")
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">MFI (14)</p>
                <p style="margin: 8px 0 0 0; font-size: 24px; color: {mfi_color}; font-weight: 800;">{mfi2:.2f}</p>
                <p style="margin: 6px 0 0 0; color: {mfi_color}; font-size: 13px; font-weight: 600;">{mfi_status}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">MFI (14)</p>
                <p style="margin: 8px 0 0 0; font-size: 18px; color: #94a3b8; font-weight: 600;">N/A</p>
            </div>
            """, unsafe_allow_html=True)

        te2 = etf2_data.get('overall_tracking_error', 'N/A')
        st.markdown(f"""
        <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
            <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">TRACKING ERROR</p>
            <p style="margin: 8px 0 0 0; font-size: 24px; color: #0f172a; font-weight: 800;">{te2}</p>
        </div>
        """, unsafe_allow_html=True)

        td2 = etf2_data.get('overall_tracking_difference', 'N/A')
        st.markdown(f"""
        <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
            <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">TRACKING DIFFERENCE</p>
            <p style="margin: 8px 0 0 0; font-size: 24px; color: #0f172a; font-weight: 800;">{td2}</p>
        </div>
        """, unsafe_allow_html=True)

        if tech2 and tech2.get("cci") is not None:
            cci2 = tech2["cci"]
            cci_color = "#dc2626" if cci2 > 100 else ("#16a34a" if cci2 < -100 else "#0f172a")
            cci_status = "Overbought" if cci2 > 100 else ("Oversold" if cci2 < -100 else "Neutral")
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">CCI (40)</p>
                <p style="margin: 8px 0 0 0; font-size: 24px; color: {cci_color}; font-weight: 800;">{cci2:.2f}</p>
                <p style="margin: 6px 0 0 0; color: {cci_color}; font-size: 13px; font-weight: 600;">{cci_status}</p>
            </div>
            """, unsafe_allow_html=True)
        if tech2 and tech2.get("sma_20") is not None:
            sma20 = tech2["sma_20"]
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">SMA 20</p>
                <p style="margin: 8px 0 0 0; font-size: 24px; color: #0f172a; font-weight: 800;">₹{sma20:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        if tech2 and tech2.get("sma_50") is not None:
            sma50 = tech2["sma_50"]
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">SMA 50</p>
                <p style="margin: 8px 0 0 0; font-size: 24px; color: #0f172a; font-weight: 800;">₹{sma50:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

# Show the comparison modal if the flag is set
if st.session_state.show_comparison and st.session_state.compare_etf_1 and st.session_state.compare_etf_2:
    show_comparison_modal()

# ============================================================
# FILTER LOGIC
# ============================================================
mask = pd.Series(True, index=df.index)

# Only apply filters from the active section
if st.session_state.active_filter_section == 'asset_class':
    # Only apply asset class filters
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

elif st.session_state.active_filter_section == 'tracking_error':
    # Only apply tracking error filter
    if selected_te_ranges:
        df_temp = df.copy()
        df_temp['te_numeric'] = pd.to_numeric(df_temp['overall_tracking_error'], errors='coerce')
        
        te_mask = pd.Series(False, index=df_temp.index)
        
        for range_label in selected_te_ranges:
            min_val, max_val = te_ranges[range_label]
            
            if min_val is None:
                te_mask |= df_temp['te_numeric'].isna()
            else:
                te_mask |= ((df_temp['te_numeric'] >= min_val) & (df_temp['te_numeric'] <= max_val))
        
        mask &= te_mask

result = df.loc[mask].copy()

# Sort by tracking error if TE filter is active
if selected_te_ranges:
    result['te_numeric'] = pd.to_numeric(result['overall_tracking_error'], errors='coerce')
    result = result.sort_values('te_numeric', na_position='first')

# ============================================================
# METRICS & ETF SELECTOR
# ============================================================  
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style="
        background: #E2E8F0;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    ">
        <p style="margin: 0; font-size: 14px; color: #2D3748; font-weight: 600;">📊 Number of ETFs</p>
        <p style="margin: 10px 0 0 0; font-size: 32px; color: #1A202C; font-weight: 700;">{result.shape[0]}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="
        background: #E2E8F0;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    ">
        <p style="margin: 0; font-size: 14px; color: #2D3748; font-weight: 600;">💰 Total AUM</p>
        <p style="margin: 10px 0 0 0; font-size: 32px; color: #1A202C; font-weight: 700;">₹ {result['aum'].sum(skipna=True):,.2f} Cr</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

#============================================================
# GLOBAL ETF SEARCH BAR
# ============================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #2D3748 0%, #1A202C 100%);
    padding: 20px 25px;
    border-radius: 12px;
    margin-bottom: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
">
    <h3 style="color: white; margin: 0 0 4px 0; font-size: 17px; font-weight: 700;">
        🔍 Search ETFs
    </h3>
    <p style="color: #A0AEC0; margin: 0; font-size: 12px;">
        Search by name, AMC, benchmark, category, or ticker
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize global search state
if 'global_search_query' not in st.session_state:
    st.session_state.global_search_query = ""

# Full ETF list (search across ALL ETFs, not just filtered result)
all_etfs_df = df.copy()

# Search input
search_query = st.text_input(
    "Search ETFs",
    value=st.session_state.global_search_query,
    placeholder="e.g. Nifty 50, HDFC, Gold, Mirae, NIFTYBEES...",
    label_visibility="collapsed",
    key="global_etf_search_input"
)

st.session_state.global_search_query = search_query

# Build a searchable combined text column across key fields
search_fields = ['etf', 'amc', 'benchmark_index', 'category', 'asset_class', 'symbol']
for field in search_fields:
    if field not in all_etfs_df.columns:
        all_etfs_df[field] = ""

all_etfs_df['_search_text'] = (
    all_etfs_df['etf'].fillna('') + ' ' +
    all_etfs_df['amc'].fillna('') + ' ' +
    all_etfs_df['benchmark_index'].fillna('') + ' ' +
    all_etfs_df['category'].fillna('') + ' ' +
    all_etfs_df['asset_class'].fillna('') + ' ' +
    all_etfs_df['symbol'].fillna('')
).str.lower()

# Filter logic
if search_query.strip():
    query_lower = search_query.strip().lower()
    # Split multi-word queries for AND matching
    query_terms = query_lower.split()
    search_mask = pd.Series(True, index=all_etfs_df.index)
    for term in query_terms:
        search_mask &= all_etfs_df['_search_text'].str.contains(term, na=False)
    
    search_results = all_etfs_df[search_mask]['etf'].dropna().unique().tolist()
    search_results = sorted(search_results)
    
    match_count = len(search_results)
    
    if match_count == 0:
        st.markdown("""
        <div style="
            background: #FFF5F5;
            border: 1px solid #FEB2B2;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 12px;
        ">
            <p style="color: #C53030; margin: 0; font-size: 13px; font-weight: 600;">
                ❌ No ETFs matched your search. Try different keywords.
            </p>
        </div>
        """, unsafe_allow_html=True)
        search_results_for_select = [""]
    else:
        st.markdown(f"""
        <div style="
            background: #F0FFF4;
            border: 1px solid #9AE6B4;
            padding: 8px 16px;
            border-radius: 8px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <p style="color: #276749; margin: 0; font-size: 13px; font-weight: 600;">
                ✅ {match_count} ETF{"s" if match_count != 1 else ""} found
            </p>
            <p style="color: #48BB78; margin: 0; font-size: 11px;">
                Select from dropdown below ↓
            </p>
        </div>
        """, unsafe_allow_html=True)
        search_results_for_select = [""] + search_results
else:
    # No search query — show full filtered result list
    search_results_for_select = None
    match_count = None

# ── ETF selector (uses search results if query exists, else uses sidebar-filtered result) ──
st.markdown("""
<div style="
    background: #EDF2F7;
    padding: 10px 15px;
    border-radius: 8px;
    margin-bottom: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
">
    <h3 style="color: #2D3748; margin: 0; font-size: 14px; text-align: center;">
        🎯 Select an ETF to View Details
        <span style="font-weight: 400; color: #718096; font-size: 12px;">
            {"— showing search results" if search_query.strip() else "— use search above or sidebar filters"}
        </span>
    </h3>
</div>
""", unsafe_allow_html=True)

# Check if ETF was selected from Tracking Error tool
if 'te_selected_etf' in st.session_state and st.session_state.te_selected_etf:
    selected_etf = st.session_state.te_selected_etf
    st.session_state.te_selected_etf = None
elif search_results_for_select is not None:
    # Use search results
    selected_etf = st.selectbox(
        "Select ETF from search results",
        search_results_for_select,
        label_visibility="collapsed",
        key="etf_selector_search"
    )
else:
    # Use sidebar-filtered result
    selected_etf = st.selectbox(
        "Select ETF",
        [""] + sorted(result["etf"].dropna().unique().tolist()),
        label_visibility="collapsed",
        key="etf_selector_filter"
    )

# ============================================================
# ETF DETAILS + HOLDINGS + PRICE + TECHNICAL ANALYSIS
# ============================================================
if selected_etf:
    row = result[result["etf"] == selected_etf].iloc[0]

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #2D3748 0%, #1A202C 100%);
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    ">
        <h2 style="color: white; margin: 0 0 10px 0; font-size: 22px; font-weight: 700;">{row.get('etf', '-')}</h2>
        <div style="display: flex; gap: 30px; flex-wrap: wrap;">
            <div>
                <span style="color: #A0AEC0; font-size: 12px;">AMC</span>
                <p style="color: white; margin: 5px 0 0 0; font-size: 14px; font-weight: 600;">{row.get('amc', '-')}</p>
            </div>
            <div>
                <span style="color: #A0AEC0; font-size: 12px;">AUM</span>
                <p style="color: white; margin: 5px 0 0 0; font-size: 14px; font-weight: 600;">₹{row.get('aum', '-')} Cr</p>
            </div>
            <div>
                <span style="color: #A0AEC0; font-size: 12px;">Expense Ratio</span>
                <p style="color: white; margin: 5px 0 0 0; font-size: 14px; font-weight: 600;">{row.get('expense_ratio', '-')}%</p>
            </div>
            <div>
                <span style="color: #A0AEC0; font-size: 12px;">Tracking Error</span>
                <p style="color: white; margin: 5px 0 0 0; font-size: 14px; font-weight: 600;">{row.get('overall_tracking_error', '-')}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview & Holdings", "💹 NAV Info", "📈 Technical Analysis", "ℹ️ Details", "📈 Return Analysis", "🧮 Return Calculator"])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); height: 100%;">
                <p style="margin: 0; font-size: 12px; color: #718096; font-weight: 600;">ASSET CLASS</p>
                <p style="margin: 8px 0 0 0; font-size: 16px; color: #2D3748; font-weight: 700;">{row.get('asset_class', '-')}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); height: 100%;">
                <p style="margin: 0; font-size: 12px; color: #718096; font-weight: 600;">CATEGORY</p>
                <p style="margin: 8px 0 0 0; font-size: 16px; color: #2D3748; font-weight: 700;">{row.get('category', '-')}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); height: 100%;">
                <p style="margin: 0; font-size: 12px; color: #718096; font-weight: 600;">LAUNCH DATE</p>
                <p style="margin: 8px 0 0 0; font-size: 16px; color: #2D3748; font-weight: 700;">{row.get('launch_date', '-')}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: #EDF2F7; padding: 12px; border-radius: 8px; margin-bottom: 20px;">
            <span style="color: #4A5568; font-size: 12px; font-weight: 600;">BENCHMARK INDEX</span>
            <p style="margin: 5px 0 0 0; color: #2D3748; font-size: 14px;">{row.get('benchmark_index', '-')}</p>
        </div>
        """, unsafe_allow_html=True)

        etf_isin = str(row.get('isin_code', '')).strip()

        if etf_isin and etf_isin in holdings_dict:
            st.markdown("""
            <div style="
                background: #EDF2F7;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <h3 style="color: #2D3748; margin: 0; font-size: 16px;">📈 Portfolio Composition</h3>
            </div>
            """, unsafe_allow_html=True)

            etf_data = holdings_dict[etf_isin]
            holdings_data = []
            skip_keys = ['ETF', 'ISIN CODE', 'NSE TICKER', 'Total']

            for key, value in etf_data.items():
                if key not in skip_keys:
                    if value is None or str(value).strip() == '' or str(value).strip().lower() == 'nan':
                        continue
                    try:
                        company_name = key.replace('Ltd.', '').replace('Ltd,', '').replace('Ltd', '').replace('Ordinary Shares', '').replace('Class A', '').replace('Class B', '').replace('Class H', '').replace('Limited', '').strip()

                        if not company_name:
                            continue

                        weight = float(value)

                        if weight <= 0:
                            continue

                        holdings_data.append({
                            'Company': company_name,
                            'Weight': weight
                        })
                    except (ValueError, TypeError) as e:
                        continue

            if holdings_data:
                col_pie, col_bar = st.columns([1, 1.5])

                num_holdings = len(holdings_data)
                if num_holdings < 10:
                    st.info(f"ℹ️ Found {num_holdings} holdings for this ETF")

                with col_pie:
                    if etf_isin in marketcap_dict:
                        marketcap_data = marketcap_dict[etf_isin]

                        st.markdown("""
                        <div style="
                            background: white;
                            padding: 15px;
                            border-radius: 8px;
                            margin-bottom: 10px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                        ">
                            <h4 style="color: #2D3748; margin: 0 0 10px 0; font-size: 14px; font-weight: bold;">Market Cap Distribution</h4>
                        </div>
                        """, unsafe_allow_html=True)

                        pie_labels = list(marketcap_data.keys())
                        pie_values = list(marketcap_data.values())

                        colors = {
                            'Large Cap': '#DC2626',
                            'Mid Cap': '#16A34A',
                            'Small Cap': '#2563EB'
                        }
                        pie_colors = [colors.get(label, '#CBD5E0') for label in pie_labels]

                        fig_pie = Figure(figsize=(6, 4))
                        ax_pie = fig_pie.add_subplot(111)

                        wedges, texts, autotexts = ax_pie.pie(
                            pie_values,
                            labels=pie_labels,
                            colors=pie_colors,
                            autopct='%1.1f%%',
                            startangle=90,
                            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
                            textprops={'fontsize': 10, 'weight': 'bold'}
                        )

                        for text in texts:
                            text.set_fontweight('bold')
                            text.set_fontsize(10)

                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontsize(10)
                            autotext.set_fontweight('bold')

                        ax_pie.axis('equal')
                        fig_pie.tight_layout()

                        st.pyplot(fig_pie, width='content')

                        st.markdown("""
                        <div style="
                            background: white;
                            padding: 10px;
                            border-radius: 6px;
                            margin-top: 10px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                        ">
                        """, unsafe_allow_html=True)

                        for cap_type, percentage in marketcap_data.items():
                            color = colors.get(cap_type, '#CBD5E0')
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 5px 0;">
                                <span style="color: #4A5568; font-size: 13px; font-weight: bold;">
                                    <span style="display: inline-block; width: 12px; height: 12px; background: {color}; border-radius: 2px; margin-right: 6px;"></span>
                                    {cap_type}
                                </span>
                                <span style="color: #2D3748; font-weight: 600; font-size: 13px;">{percentage:.2f}%</span>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info("Market cap data not available")

                with col_bar:
                    st.markdown("""
                    <div style="
                        background: white;
                        padding: 15px;
                        border-radius: 8px;
                        margin-bottom: 10px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                    ">
                        <h4 style="color: #2D3748; margin: 0 0 10px 0; font-size: 14px; font-weight: bold;">Top 10 Holdings</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    holdings_chart_df = pd.DataFrame(holdings_data)

                    holdings_chart_df = holdings_chart_df.sort_values('Weight', ascending=False)

                    if len(holdings_chart_df) > 10:
                        holdings_chart_df = holdings_chart_df.head(10)

                    color_scale = alt.Scale(
                        domain=[holdings_chart_df['Weight'].min(), holdings_chart_df['Weight'].max()],
                        range=['#93C5FD', '#1E40AF']
                    )

                    base = alt.Chart(holdings_chart_df).encode(
                        y=alt.Y('Company:N',
                               sort=alt.EncodingSortField(field='Weight', order='descending'),
                               axis=alt.Axis(title='', labelFontSize=10, labelLimit=150)),
                        x=alt.X('Weight:Q',
                               axis=alt.Axis(title='Weight (%)', titleFontSize=11, labelFontSize=9))
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
                        dx=3,
                        fontSize=9,
                        fontWeight='normal',
                        color='#2D3748'
                    ).encode(
                        text=alt.Text('Weight:Q', format='.2f')
                    )

                    final_chart = alt.layer(bars, text).properties(
                        height=300
                    ).configure_view(
                        strokeWidth=0
                    ).configure_axis(
                        gridColor='rgba(0, 0, 0, 0.05)'
                    )

                    st.altair_chart(final_chart, width='stretch')

                total_value = etf_data.get('Total', '')
                if total_value and str(total_value).strip():
                    try:
                        total_pct = float(total_value)
                        st.markdown(
                            f"""
                            <div style="
                                background: #EDF2F7;
                                padding: 12px;
                                border-radius: 6px;
                                text-align: center;
                                margin-top: 10px;
                                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
                            ">
                                <p style="color: #2D3748; font-size: 14px; margin: 0; font-weight: 600;">
                                    📊 Top 10 Holdings Total: {total_pct}%
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

    with tab2:
        st.subheader("💹 Current NAV/LTP Information")

        if df_price.empty:
            st.warning("⚠️ Price data file not found.")
            st.stop()

        ticker = str(row.get("symbol", "")).strip().upper()
        price_row = df_price[df_price["symbol"] == ticker]

        if price_row.empty:
            st.warning(f"⚠️ Price data not available for ticker: {ticker}")
            st.stop()

        ltp_val = pd.to_numeric(price_row.iloc[0]["ltp"], errors="coerce")
        nav_val = pd.to_numeric(price_row.iloc[0]["nav"], errors="coerce")

        col1, col2 = st.columns(2)

        # LTP / NAV
        with col1:
            st.metric("LTP", f"₹ {ltp_val:.2f}" if not pd.isna(ltp_val) else "N/A")
            st.metric("NAV", f"₹ {nav_val:.2f}" if not pd.isna(nav_val) else "N/A")

        # Premium / Discount
        with col2:
            if pd.isna(ltp_val) or pd.isna(nav_val) or nav_val == 0:
                st.info("Premium/Discount requires both LTP and NAV")
            else:
                premium_pct = ((ltp_val - nav_val) / nav_val) * 100

                if -0.5 <= premium_pct <= 0.5:
                    label = "Fair Value"
                elif abs(premium_pct) <= 1.5:
                    label = "Mild"
                elif abs(premium_pct) <= 3:
                    label = "Moderate"
                else:
                    label = "HIGH"

                st.metric("Premium / Discount", f"{premium_pct:+.2f}%", label)

            st.caption("""
            Guide:
            - -0.5% to 0.5% → Fair  
            - 0.5–1.5% → Mild  
            - 1.5–3% → Moderate  
            - >3% → High
            """)

            # Bar Chart
            if not pd.isna(ltp_val) and not pd.isna(nav_val):
                chart_df = pd.DataFrame({
                    "Metric": ["LTP", "NAV"],
                    "Value": [ltp_val, nav_val]
                })

                chart = alt.Chart(chart_df).mark_bar(size=50).encode(
                    x=alt.X('Metric', axis=alt.Axis(title=None)),
                    y=alt.Y('Value', axis=alt.Axis(title='Value (₹)')),
                    color=alt.Color('Metric',
                            scale=alt.Scale(
                                domain=['LTP','NAV'],
                                range=['#FEB2B2','#90CDF4']
                            ),
                            legend=None)
                ).properties(height=300)

                st.altair_chart(chart, width='stretch')

    with tab3:
        yfinance_ticker = str(row.get('yfinance_ticker', '')).strip()

        if yfinance_ticker:
            with st.spinner(f"Fetching data for {yfinance_ticker}..."):
                etf_data = fetch_etf_data(yfinance_ticker)

                if etf_data is not None and not etf_data.empty:
                    charts, analyzed_data = create_technical_charts(etf_data, selected_etf)

                    display_indicator_summary(analyzed_data, selected_etf)

                    subtab1, subtab2, subtab3, subtab4, subtab5, subtab6, subtab7, subtab8, subtab9 = st.tabs([
                        "📊 Moving Averages",
                        "📈 RSI",
                        "💰 MFI",
                        "📉 EMV",
                        "📊 ATR",
                        "📊 Mean Deviation",
                        "📊 CCI",
                        "📊 Volume Analysis",
                        "📊 MACD"
                    ])

                    with subtab1:

                        # ==========================
                        # 5Y PRICE CHART (NEW)
                        # ==========================
                        st.markdown("### 📈 5-Year Price Trend")

                        price_fig = go.Figure()

                        price_fig.add_trace(
                            go.Scatter(
                                x=etf_data.index,
                                y=etf_data['Close'],
                                mode='lines',
                                name='Price',
                                line=dict(width=2),
                                hovertemplate='Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
                            )
                        )

                        price_fig.update_layout(
                            height=400,
                            hovermode='x unified',
                            xaxis_title="Date",
                            yaxis_title="Price (₹)"
                        )

                        st.plotly_chart(price_fig, width='stretch')

                        # ==========================
                        # EXISTING MA CHART
                        # ==========================
                        st.markdown("### 📊 Moving Averages")

                        st.plotly_chart(charts[0], width='stretch')

                        

                    with subtab2:
                        st.plotly_chart(charts[1], width='stretch')

                    with subtab3:
                        st.plotly_chart(charts[2], width='stretch')

                    with subtab4:
                        st.plotly_chart(charts[3], width='stretch')

                    with subtab5:
                        st.plotly_chart(charts[4], width='stretch')

                    with subtab6:
                        st.plotly_chart(charts[5], width='stretch')

                    with subtab7:
                        st.plotly_chart(charts[6], width='stretch')

                    
                    with subtab8:

                        # ----------- ADD THIS PART -----------
                        vol_signals = get_etf_volume_signals(analyzed_data)

                        st.markdown("### 📊 Volume Signals")

                        col1, col2, col3 = st.columns(3)

                        for col, (k, v) in zip([col1, col2, col3], vol_signals.items()):

                            if "Bullish" in v or "Rising" in v or "High" in v:
                                col.success(f"{k}: {v}")

                            elif "Selling" in v or "Declining" in v:
                                col.error(f"{k}: {v}")

                            else:
                                col.info(f"{k}: {v}")


                        st.markdown("---")
                        # ----------- END ADD -----------

                        period_names, sentiments, sentiment_colors, overall, overall_color = analyze_volume_sentiment(analyzed_data['Volume'])

                        st.markdown("""
                        <div style="
                            background: white;
                            padding: 15px;
                            border-radius: 8px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                            margin-bottom: 15px;
                        ">
                            <h4 style="color: #2D3748; margin: 0 0 10px 0; font-size: 14px;">Volume Sentiment Summary</h4>
                        """, unsafe_allow_html=True)

                        for period, sentiment, color in zip(period_names, sentiments, sentiment_colors):
                            st.markdown(f"<p style='margin: 5px 0;'><span style='font-weight: 600; color: #4A5568;'>{period}:</span> <span style='color: {color}; font-weight: 600;'>{sentiment}</span></p>", unsafe_allow_html=True)

                        st.markdown(f"""
                            <div style="
                                background: #EDF2F7;
                                padding: 12px;
                                border-radius: 6px;
                                margin-top: 10px;
                                text-align: center;
                            ">
                                <p style="margin: 0; font-size: 14px; color: {overall_color}; font-weight: 700;">{overall}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        volume_charts = create_volume_charts(analyzed_data, selected_etf)

                        for i, fig in enumerate(volume_charts):
                            st.pyplot(fig, width='content')

                    with subtab9:
                        st.plotly_chart(charts[7], width='stretch')
                else:
                    st.error(f"⚠️ Could not fetch data for {yfinance_ticker}. Please check if the ticker is correct.")
        else:
            st.info("📊 Technical analysis not available - yfinance ticker not found for this ETF.")

    with tab4:
        st.markdown("""
        <div style="
            background: #EDF2F7;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        ">
            <h3 style="color: #2D3748; margin: 0; font-size: 16px;">ℹ️ Complete ETF Information</h3>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"**ETF Name:** {row.get('etf', '-')}")
            st.markdown(f"**AMC:** {row.get('amc', '-')}")
            st.markdown(f"**Asset Class:** {row.get('asset_class', '-')}")
            st.markdown(f"**Category:** {row.get('category', '-')}")
            st.markdown(f"**Sub-Category:** {row.get('sub-category', '-')}")
            st.markdown(f"**Benchmark Index:** {row.get('benchmark_index', '-')}")

        with col_b:
            st.markdown(f"**NSE Ticker:** {row.get('symbol', '-')}")
            st.markdown(f"**BSE Ticker:** {row.get('bse_ticker', '-')}")
            st.markdown(f"**ISIN Code:** {row.get('isin_code', '-')}")
            st.markdown(f"**Launch Date:** {row.get('launch_date', '-')}")
            st.markdown(f"**AUM (₹ Cr):** {row.get('aum', '-')}")
            st.markdown(f"**Years Active:** {row.get('years_active', '-')}")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        ">
            <h4 style="color: #2D3748; margin: 0 0 10px 0; font-size: 14px;">📊 Performance Metrics</h4>
        """, unsafe_allow_html=True)

        perf_col1, perf_col2 = st.columns(2)

        with perf_col1:
            st.markdown(f"**Expense Ratio:** {row.get('expense_ratio', '-')}%")
            st.markdown(f"**Tracking Error:** {row.get('overall_tracking_error', '-')}")

        with perf_col2:
            st.markdown(f"**Tracking Difference:** {row.get('overall_tracking_difference', '-')}")

        st.markdown("</div>", unsafe_allow_html=True)

        website = row.get("website_link", "")
        if isinstance(website, str) and website.strip():
            st.markdown(f"""
            <div style="text-align: center; margin-top: 20px;">
                <a href="{website}" target="_blank" style="
                    display: inline-block;
                    background: linear-gradient(135deg, #4A5568 0%, #2D3748 100%);
                    color: white;
                    font-weight: 600;
                    text-decoration: none;
                    font-size: 14px;
                    padding: 12px 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    🌐 Visit Official Website
                </a>
            </div>
            """, unsafe_allow_html=True)
    
    # ============================================================
    # RETURN ANALYSIS TAB
    # ============================================================
    with tab5:

        st.subheader("📈 Return Analysis")

        yfinance_ticker = str(row.get('yfinance_ticker', '')).strip()

        if yfinance_ticker:

            with st.spinner("Calculating returns..."):
                etf_hist = fetch_etf_data(yfinance_ticker)

            # ======================
            # ABSOLUTE RETURNS
            # ======================
            st.markdown("### 📊 Absolute Returns")

            returns = calculate_returns(etf_hist)

            if returns:

                cols = st.columns(3)

                for i, (period, val) in enumerate(returns.items()):

                    if val is None:
                        cols[i % 3].metric(period, "N/A")
                    else:
                        cols[i % 3].metric(period, f"{val:.2f}%", f"{val:.2f}%")

            # ======================
            # CAGR
            # ======================
            st.markdown("### 📈 CAGR")

            cagr_values = calculate_cagr(etf_hist)

            if cagr_values:

                cols = st.columns(3)

                for i, (label, val) in enumerate(cagr_values.items()):

                    if val is None:
                        cols[i % 3].metric(label, "N/A")
                    else:
                        cols[i % 3].metric(label, f"{val:.2f}%")

        else:
            st.warning("No yfinance ticker available for this ETF")         
    
    with tab6:

        st.markdown("""
        <style>
        .rc-header {
            background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
            border: 1px solid #C7D2FE;
            border-radius: 14px;
            padding: 22px 28px;
            margin-bottom: 24px;
        }
        .rc-header h2 { font-size: 22px; font-weight: 700; color: #3730A3; margin: 0 0 4px 0; }
        .rc-header p  { font-size: 13px; color: #6366F1; margin: 0; }
        .rc-cagr-pill {
            display: inline-block; background: #F0FDF4;
            border: 1px solid #86EFAC; border-radius: 20px;
            padding: 4px 14px; font-size: 12px; font-weight: 600;
            color: #15803D; margin: 3px 4px;
        }
        .rc-cagr-section {
            background: #F0FDF4; border: 1px solid #BBF7D0;
            border-radius: 10px; padding: 14px 16px; margin-bottom: 20px;
        }
        .rc-cagr-section p {
            font-size: 11px; font-weight: 700; color: #166534;
            letter-spacing: 0.8px; text-transform: uppercase; margin: 0 0 8px 0;
        }
        .rc-result-card {
            background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%);
            border: 2px solid #A5B4FC; border-radius: 16px;
            padding: 24px 20px; text-align: center;
            box-shadow: 0 4px 20px rgba(99,102,241,0.10);
            margin-bottom: 12px;
        }
        .rc-result-card-real {
            background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%);
            border: 2px solid #FCA5A5; border-radius: 16px;
            padding: 20px; text-align: center;
            box-shadow: 0 4px 12px rgba(239,68,68,0.10);
        }
        .rc-result-label  { font-size: 11px; font-weight: 700; color: #6366F1; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px; }
        .rc-result-label-real { font-size: 11px; font-weight: 700; color: #EA580C; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px; }
        .rc-result-value  { font-size: 38px; font-weight: 800; color: #3730A3; margin: 0; letter-spacing: -1px; line-height: 1.1; }
        .rc-result-value-real { font-size: 32px; font-weight: 800; color: #C2410C; margin: 0; letter-spacing: -1px; line-height: 1.1; }
        .rc-result-invested { font-size: 13px; color: #6B7280; margin: 8px 0 4px 0; }
        .rc-gain-pos { color: #16A34A; font-weight: 700; font-size: 15px; }
        .rc-gain-neg { color: #DC2626; font-weight: 700; font-size: 15px; }
        .rc-stat { background: white; border: 1px solid #E5E7EB; border-radius: 10px; padding: 14px 12px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
        .rc-stat-label { font-size: 10px; font-weight: 700; color: #9CA3AF; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 6px; }
        .rc-stat-value { font-size: 18px; font-weight: 700; color: #1F2937; }
        .rc-section-title { font-size: 14px; font-weight: 700; color: #374151; margin: 20px 0 12px 0; padding-bottom: 6px; border-bottom: 2px solid #E5E7EB; }
        .rc-return-card  { background: white; border: 1px solid #E5E7EB; border-radius: 8px; padding: 10px 8px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
        .rc-return-period { font-size: 10px; font-weight: 600; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.5px; }
        .rc-inflation-box {
            background: #FFF7ED; border: 1px solid #FED7AA;
            border-radius: 10px; padding: 14px 16px; margin: 16px 0;
        }
        .rc-disclaimer { background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 10px 14px; font-size: 11px; color: #92400E; text-align: center; margin-top: 24px; }
        </style>
        """, unsafe_allow_html=True)

        yfinance_ticker = str(row.get('yfinance_ticker', '')).strip()

        if not yfinance_ticker:
            st.warning("No price data ticker available for this ETF — return calculator unavailable.")
        else:
            # ── Header ────────────────────────────────────────────────────
            st.markdown(f"""
            <div class="rc-header">
                <h2>🧮 Return Calculator</h2>
                <p>{selected_etf}</p>
            </div>
            """, unsafe_allow_html=True)

            # ── Fetch data ────────────────────────────────────────────────
            with st.spinner("Loading historical data…"):
                etf_hist    = fetch_etf_data(yfinance_ticker)
            cagr_values = calculate_cagr(etf_hist)   or {}
            returns     = calculate_returns(etf_hist) or {}

            # ── CAGR pills ────────────────────────────────────────────────
            if cagr_values:
                pills = "".join(
                    f'<span class="rc-cagr-pill">{lbl}: {val:.2f}%</span>'
                    for lbl, val in cagr_values.items() if val is not None
                )
                st.markdown(f"""
                <div class="rc-cagr-section">
                    <p>📈 Historical CAGR Reference</p>
                    {pills}
                </div>
                """, unsafe_allow_html=True)

            # ── Mode + Rate ───────────────────────────────────────────────
            col_mode, col_rate = st.columns([1, 2])
            with col_mode:
                calc_mode = st.radio(
                    "Mode",
                    ["💰 Lumpsum", "📅 SIP (Monthly)"],
                    key="rc_mode"
                )
            with col_rate:
                rate_options = {
                    f"{lbl} ({val:.2f}%)": val
                    for lbl, val in cagr_values.items() if val is not None
                }
                rate_options["✏️ Custom Rate"] = None
                chosen_label = st.selectbox(
                    "Expected Annual Return (CAGR)",
                    list(rate_options.keys()),
                    key="rc_rate_label"
                )
                if rate_options[chosen_label] is None:
                    annual_rate = st.number_input(
                        "Custom rate (%)",
                        min_value=-50.0, max_value=100.0,
                        value=12.0, step=0.5, format="%.1f",
                        key="rc_custom_rate"
                    )
                else:
                    annual_rate = rate_options[chosen_label]

            # ── Inflation toggle ──────────────────────────────────────────
            st.markdown(
                "<hr style='border:none;border-top:1px solid #E5E7EB;margin:16px 0'>",
                unsafe_allow_html=True
            )
            use_inflation = st.checkbox(
                "📉 Adjust for Inflation (show real purchasing power)",
                value=False,
                key="rc_use_inflation"
            )
            inflation_rate = 0.0
            if use_inflation:
                st.markdown("""
                <div class="rc-inflation-box">
                    <b style="color:#C2410C;font-size:13px;">🌡️ Inflation Settings</b>
                    <p style="color:#6B7280;font-size:12px;margin:4px 0 10px 0;">
                        India's long-run CPI inflation averages ~5–7%. Adjust to your expectation.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                inflation_rate = st.slider(
                    "Expected Annual Inflation Rate (%)",
                    min_value=1.0, max_value=15.0,
                    value=6.0, step=0.5,
                    key="rc_inflation_rate"
                )
                real_rate = ((1 + annual_rate / 100) / (1 + inflation_rate / 100) - 1) * 100
                st.info(
                    f"📌 **Real (inflation-adjusted) return = {real_rate:.2f}%**  "
                    f"(Nominal {annual_rate:.1f}% − Inflation {inflation_rate:.1f}%  "
                    f"using Fisher equation)"
                )

            st.markdown(
                "<hr style='border:none;border-top:1px solid #E5E7EB;margin:16px 0'>",
                unsafe_allow_html=True
            )

            # ════════════════════════════════════════════════════════════
            # LUMPSUM
            # ════════════════════════════════════════════════════════════
            if calc_mode == "💰 Lumpsum":

                col_inp, col_res = st.columns([1, 1], gap="large")

                with col_inp:
                    st.markdown('<p class="rc-section-title">Investment Details</p>', unsafe_allow_html=True)
                    lump_amount = st.number_input(
                        "Lumpsum Amount (₹)",
                        min_value=1_000, max_value=10_000_000,
                        value=100_000, step=1_000,
                        format="%d", key="rc_lump_amount"
                    )
                    lump_years = st.slider(
                        "Investment Horizon (Years)",
                        min_value=1, max_value=30, value=5,
                        key="rc_lump_years"
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                    s1, s2, s3 = st.columns(3)
                    doubling = round(72 / annual_rate, 1) if annual_rate > 0 else "—"
                    with s1:
                        st.markdown(f'<div class="rc-stat"><div class="rc-stat-label">Nominal Rate</div><div class="rc-stat-value">{annual_rate:.1f}%</div></div>', unsafe_allow_html=True)
                    with s2:
                        st.markdown(f'<div class="rc-stat"><div class="rc-stat-label">Horizon</div><div class="rc-stat-value">{lump_years}Y</div></div>', unsafe_allow_html=True)
                    with s3:
                        st.markdown(f'<div class="rc-stat"><div class="rc-stat-label">Rule of 72</div><div class="rc-stat-value">{doubling}Y</div></div>', unsafe_allow_html=True)

                with col_res:
                    # ── Nominal calculation ──────────────────────────────
                    fv       = lump_amount * ((1 + annual_rate / 100) ** lump_years)
                    gain     = fv - lump_amount
                    gain_pct = (gain / lump_amount) * 100
                    g_class  = "rc-gain-pos" if gain >= 0 else "rc-gain-neg"
                    g_sign   = "+" if gain >= 0 else ""

                    st.markdown('<p class="rc-section-title">Projected Outcome</p>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="rc-result-card">
                        <div class="rc-result-label">💰 Nominal Future Value</div>
                        <div class="rc-result-value">₹{fv:,.0f}</div>
                        <div class="rc-result-invested">Amount Invested: ₹{lump_amount:,.0f}</div>
                        <div class="{g_class}">{g_sign}₹{gain:,.0f} &nbsp;({g_sign}{gain_pct:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Inflation-adjusted calculation ───────────────────
                    if use_inflation:
                        real_fv      = fv / ((1 + inflation_rate / 100) ** lump_years)
                        real_gain    = real_fv - lump_amount
                        real_gain_pct = (real_gain / lump_amount) * 100
                        rg_class     = "rc-gain-pos" if real_gain >= 0 else "rc-gain-neg"
                        rg_sign      = "+" if real_gain >= 0 else ""
                        # purchasing power of invested amount in today's money already = lump_amount
                        st.markdown(f"""
                        <div class="rc-result-card-real">
                            <div class="rc-result-label-real">📉 Real Value (Today's Purchasing Power @ {inflation_rate:.1f}% inflation)</div>
                            <div class="rc-result-value-real">₹{real_fv:,.0f}</div>
                            <div class="rc-result-invested">Equivalent to today's ₹{lump_amount:,.0f}</div>
                            <div class="{rg_class}">{rg_sign}₹{real_gain:,.0f} &nbsp;({rg_sign}{real_gain_pct:.1f}% real gain)</div>
                        </div>
                        """, unsafe_allow_html=True)

                # ── Growth chart ─────────────────────────────────────────
                st.markdown('<p class="rc-section-title">📈 Projected Growth</p>', unsafe_allow_html=True)
                yr_range  = list(range(0, lump_years + 1))
                proj_vals = [lump_amount * ((1 + annual_rate / 100) ** y) for y in yr_range]

                proj_fig = go.Figure()
                proj_fig.add_trace(go.Scatter(
                    x=yr_range, y=proj_vals,
                    mode="lines+markers", fill="tozeroy",
                    fillcolor="rgba(99,102,241,0.08)",
                    line=dict(color="#6366F1", width=2.5),
                    marker=dict(size=5, color="#6366F1"),
                    name="Nominal Value",
                    hovertemplate="Year %{x}: ₹%{y:,.0f}<extra></extra>"
                ))
                proj_fig.add_hline(
                    y=lump_amount, line_dash="dot", line_color="#9CA3AF",
                    annotation_text="Invested Amount",
                    annotation_font_color="#6B7280",
                    annotation_position="bottom right"
                )

                if use_inflation:
                    real_vals = [
                        lump_amount * ((1 + annual_rate / 100) ** y) / ((1 + inflation_rate / 100) ** y)
                        for y in yr_range
                    ]
                    proj_fig.add_trace(go.Scatter(
                        x=yr_range, y=real_vals,
                        mode="lines+markers",
                        line=dict(color="#EA580C", width=2, dash="dash"),
                        marker=dict(size=4, color="#EA580C"),
                        name=f"Real Value ({inflation_rate:.1f}% inflation)",
                        hovertemplate="Year %{x} (Real): ₹%{y:,.0f}<extra></extra>"
                    ))

                proj_fig.update_layout(
                    height=320, paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                    xaxis=dict(title="Years", gridcolor="#F3F4F6", linecolor="#E5E7EB", color="#6B7280"),
                    yaxis=dict(title="Value (₹)", gridcolor="#F3F4F6", linecolor="#E5E7EB", color="#6B7280", tickformat=",.0f"),
                    hovermode="x unified",
                    legend=dict(bgcolor="white", bordercolor="#E5E7EB", borderwidth=1, font=dict(color="#374151")),
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(proj_fig, use_container_width=True)

            # ════════════════════════════════════════════════════════════
            # SIP
            # ════════════════════════════════════════════════════════════
            else:
                col_inp, col_res = st.columns([1, 1], gap="large")

                with col_inp:
                    st.markdown('<p class="rc-section-title">SIP Details</p>', unsafe_allow_html=True)
                    monthly_sip = st.number_input(
                        "Monthly SIP Amount (₹)",
                        min_value=500, max_value=1_000_000,
                        value=10_000, step=500,
                        format="%d", key="rc_sip_amount"
                    )
                    sip_years = st.slider(
                        "Investment Horizon (Years)",
                        min_value=1, max_value=30, value=10,
                        key="rc_sip_years"
                    )
                    monthly_rate   = annual_rate / 100 / 12
                    n_months       = sip_years * 12
                    total_invested = monthly_sip * n_months
                    st.markdown("<br>", unsafe_allow_html=True)
                    s1, s2, s3 = st.columns(3)
                    with s1:
                        st.markdown(f'<div class="rc-stat"><div class="rc-stat-label">Nominal Rate</div><div class="rc-stat-value">{annual_rate:.1f}%</div></div>', unsafe_allow_html=True)
                    with s2:
                        st.markdown(f'<div class="rc-stat"><div class="rc-stat-label">Total Invested</div><div class="rc-stat-value">₹{total_invested/100000:.1f}L</div></div>', unsafe_allow_html=True)
                    with s3:
                        st.markdown(f'<div class="rc-stat"><div class="rc-stat-label">Horizon</div><div class="rc-stat-value">{sip_years}Y</div></div>', unsafe_allow_html=True)

                with col_res:
                    # ── Nominal SIP calculation ──────────────────────────
                    if monthly_rate == 0:
                        sip_fv = total_invested
                    else:
                        sip_fv = monthly_sip * (
                            ((1 + monthly_rate) ** n_months - 1) / monthly_rate
                        ) * (1 + monthly_rate)

                    sip_gain     = sip_fv - total_invested
                    sip_gain_pct = (sip_gain / total_invested) * 100 if total_invested else 0
                    g_class      = "rc-gain-pos" if sip_gain >= 0 else "rc-gain-neg"
                    g_sign       = "+" if sip_gain >= 0 else ""

                    st.markdown('<p class="rc-section-title">Projected Outcome</p>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="rc-result-card">
                        <div class="rc-result-label">💰 Nominal Future Value</div>
                        <div class="rc-result-value">₹{sip_fv:,.0f}</div>
                        <div class="rc-result-invested">Total Invested: ₹{total_invested:,.0f}</div>
                        <div class="{g_class}">{g_sign}₹{sip_gain:,.0f} &nbsp;({g_sign}{sip_gain_pct:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Inflation-adjusted SIP calculation ───────────────
                    if use_inflation:
                        # Deflate nominal corpus by inflation over the horizon
                        real_sip_fv      = sip_fv / ((1 + inflation_rate / 100) ** sip_years)
                        real_sip_gain    = real_sip_fv - total_invested
                        real_sip_gain_pct = (real_sip_gain / total_invested) * 100 if total_invested else 0
                        rg_class         = "rc-gain-pos" if real_sip_gain >= 0 else "rc-gain-neg"
                        rg_sign          = "+" if real_sip_gain >= 0 else ""
                        st.markdown(f"""
                        <div class="rc-result-card-real">
                            <div class="rc-result-label-real">📉 Real Value (Today's Purchasing Power @ {inflation_rate:.1f}% inflation)</div>
                            <div class="rc-result-value-real">₹{real_sip_fv:,.0f}</div>
                            <div class="rc-result-invested">Total Invested: ₹{total_invested:,.0f} (today's ₹)</div>
                            <div class="{rg_class}">{rg_sign}₹{real_sip_gain:,.0f} &nbsp;({rg_sign}{real_sip_gain_pct:.1f}% real gain)</div>
                        </div>
                        """, unsafe_allow_html=True)

                # ── SIP chart ────────────────────────────────────────────
                st.markdown('<p class="rc-section-title">📈 SIP Growth Projection</p>', unsafe_allow_html=True)
                sip_m_range   = list(range(0, n_months + 1))
                sip_inv_curve = [monthly_sip * m for m in sip_m_range]
                sip_val_curve = []
                for m in sip_m_range:
                    if m == 0:
                        sip_val_curve.append(0)
                    elif monthly_rate == 0:
                        sip_val_curve.append(monthly_sip * m)
                    else:
                        sip_val_curve.append(
                            monthly_sip * (
                                ((1 + monthly_rate) ** m - 1) / monthly_rate
                            ) * (1 + monthly_rate)
                        )

                sip_fig = go.Figure()
                sip_fig.add_trace(go.Scatter(
                    x=sip_m_range, y=sip_val_curve,
                    mode="lines", name="Nominal Value",
                    fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
                    line=dict(color="#6366F1", width=2.5),
                    hovertemplate="Month %{x}: ₹%{y:,.0f}<extra></extra>"
                ))
                sip_fig.add_trace(go.Scatter(
                    x=sip_m_range, y=sip_inv_curve,
                    mode="lines", name="Amount Invested",
                    line=dict(color="#9CA3AF", width=1.5, dash="dot"),
                    hovertemplate="Month %{x}: ₹%{y:,.0f}<extra></extra>"
                ))

                if use_inflation:
                    monthly_inflation = inflation_rate / 100 / 12
                    real_sip_curve = [
                        v / ((1 + monthly_inflation) ** m) if m > 0 else 0
                        for m, v in zip(sip_m_range, sip_val_curve)
                    ]
                    sip_fig.add_trace(go.Scatter(
                        x=sip_m_range, y=real_sip_curve,
                        mode="lines", name=f"Real Value ({inflation_rate:.1f}% inflation)",
                        line=dict(color="#EA580C", width=2, dash="dash"),
                        hovertemplate="Month %{x} (Real): ₹%{y:,.0f}<extra></extra>"
                    ))

                sip_fig.update_layout(
                    height=320, paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                    xaxis=dict(title="Month", gridcolor="#F3F4F6", linecolor="#E5E7EB", color="#6B7280"),
                    yaxis=dict(title="Value (₹)", gridcolor="#F3F4F6", linecolor="#E5E7EB", color="#6B7280", tickformat=",.0f"),
                    hovermode="x unified",
                    legend=dict(bgcolor="white", bordercolor="#E5E7EB", borderwidth=1, font=dict(color="#374151")),
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(sip_fig, use_container_width=True)

            # ── Historical price chart (always shown) ────────────────────
            st.markdown('<p class="rc-section-title">📊 ETF Historical Price Chart</p>', unsafe_allow_html=True)
            if etf_hist is not None and not etf_hist.empty:
                sma50     = etf_hist['Close'].rolling(50).mean()
                price_fig = go.Figure()
                price_fig.add_trace(go.Scatter(
                    x=etf_hist.index, y=etf_hist['Close'],
                    mode='lines', name='Close Price',
                    fill='tozeroy', fillcolor='rgba(99,102,241,0.07)',
                    line=dict(color='#6366F1', width=2),
                    hovertemplate='%{x|%d %b %Y}: ₹%{y:.2f}<extra></extra>'
                ))
                price_fig.add_trace(go.Scatter(
                    x=etf_hist.index, y=sma50,
                    mode='lines', name='SMA 50',
                    line=dict(color='#F59E0B', width=1.5, dash='dot'),
                    hovertemplate='SMA50: ₹%{y:.2f}<extra></extra>'
                ))
                price_fig.update_layout(
                    height=350, paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                    xaxis=dict(title="Date", gridcolor="#F3F4F6", linecolor="#E5E7EB", color="#6B7280"),
                    yaxis=dict(title="Price (₹)", gridcolor="#F3F4F6", linecolor="#E5E7EB", color="#6B7280"),
                    hovermode="x unified",
                    legend=dict(bgcolor="white", bordercolor="#E5E7EB", borderwidth=1, font=dict(color="#374151")),
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(price_fig, use_container_width=True)
            else:
                st.info("Historical price data not available for this ETF.")

            # ── Actual returns strip ─────────────────────────────────────
            if returns:
                st.markdown('<p class="rc-section-title">📊 Actual Historical Returns</p>', unsafe_allow_html=True)
                ret_items = list(returns.items())
                ret_cols  = st.columns(len(ret_items))
                for col, (period, val) in zip(ret_cols, ret_items):
                    if val is not None:
                        color = "#16A34A" if val >= 0 else "#DC2626"
                        sign  = "+" if val >= 0 else ""
                        col.markdown(f"""
                        <div class="rc-return-card">
                            <div class="rc-return-period">{period}</div>
                            <div style="font-size:15px;font-weight:700;color:{color};margin-top:4px;">
                                {sign}{val:.1f}%
                            </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        col.markdown(f"""
                        <div class="rc-return-card">
                            <div class="rc-return-period">{period}</div>
                            <div style="font-size:14px;color:#9CA3AF;margin-top:4px;">N/A</div>
                        </div>""", unsafe_allow_html=True)

            # ── Disclaimer ───────────────────────────────────────────────
            st.markdown("""
            <div class="rc-disclaimer">
                ⚠️ Projections are illustrative only. Past returns do not guarantee future performance.
                Inflation adjustment shows estimated purchasing power — actual inflation may vary.
                Please consult a financial advisor before investing.
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# AI ASSISTANT SECTION
# ============================================================
st.markdown("---")
st.markdown("""
<div style="
    background: #EDF2F7;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin: 30px 0;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
">
    <h2 style="color: #2D3748; margin: 0 0 15px 0;">🤖 AI-Powered ETF Assistant</h2>
    <p style="color: #4A5568; font-size: 16px; margin: 0 0 20px 0;">
        Get intelligent insights and analysis on Indian ETFs
    </p>
    <a href="https://chatgpt.com/g/g-6942d299b4648191a8acc98e68636cb9-indiaetf" target="_blank"
       style="
           display: inline-block;
           background: linear-gradient(135deg, #4A5568 0%, #2D3748 100%);
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
