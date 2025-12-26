import requests
import pandas as pd
from datetime import date
import os

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================
# CONFIG
# ============================================================
OUTPUT_FILE = "data/nse_etf_prices.csv"
NSE_API_URL = "https://www.nseindia.com/api/etf"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

# ============================================================
# CREATE SESSION (NSE REQUIRES COOKIES)
# ============================================================
session = requests.Session()

retry_strategy = Retry(
    total=3,
    backoff_factor=5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)

# Hit homepage to set cookies
session.get("https://www.nseindia.com", headers=headers, timeout=30)
print("Homepage cookies acquired")
# ============================================================
# FETCH ETF DATA
# ============================================================
try:
    response = session.get(NSE_API_URL, headers=headers, timeout=40)
    print("API status code:", response.status_code)
    print("API response content type:", response.headers.get("Content-Type"))
    response.raise_for_status()
except Exception as e:
    print("NSE API failed or timed out:", e)
    print("Skipping update for today to avoid failure")
    exit(0)


json_data = response.json()
data = json_data.get("data", [])
print("Number of records received:", len(data))
if not data:
    raise RuntimeError("NSE ETF API returned empty data")

df_new = pd.DataFrame(data)

# ============================================================
# STANDARDIZE & ADD DATE
# ============================================================
df_new.columns = df_new.columns.str.strip().str.lower()
df_new["symbol"] = df_new["symbol"].astype(str).str.strip()
df_new["date"] = pd.to_datetime(date.today())

# ============================================================
# APPEND OR CREATE CSV
# ============================================================
if os.path.exists(OUTPUT_FILE):
    df_old = pd.read_csv(OUTPUT_FILE)
    df = pd.concat([df_old, df_new], ignore_index=True)
else:
    df = df_new

df.to_csv(OUTPUT_FILE, index=False)

print("NSE ETF EOD data downloaded and saved successfully")

