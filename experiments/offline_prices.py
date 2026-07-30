"""
offline_prices.py — offline replacement for models.<model>.fetch_data, backed by
the committed DONNEE~1.XLS workbook (daily Close for the 5 benchmark assets,
2018-02 -> 2026-07, verified identical to yfinance auto_adjust Close against
Run/*/prices.parquet: max abs deviation 0.000004%).

Why: sandboxed / air-gapped environments can't reach Yahoo Finance. Every
experiment runner that needs price history can do

    from offline_prices import fetch_data_offline as fetch_data

and get the exact same Series (tz-naive DatetimeIndex, float Close) that
models.arima_model.fetch_data(ticker, start, end) would have returned.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLS_PATH = ROOT / "DONNEE~1.XLS"

SHEET_BY_TICKER = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SPY": "S&P 500 (SPY)",
    "ZN=F": "US Treasury 10Y Note Futures",
    "TLT": "US Treasury 20+Y (ETF)",
}

_cache: dict = {}


def fetch_data_offline(ticker: str, start: str, end: str) -> pd.Series:
    """Daily Close prices from DONNEE~1.XLS -- same contract as
    models.arima_model.fetch_data (end is EXCLUSIVE, like yf.download)."""
    if ticker not in SHEET_BY_TICKER:
        raise KeyError(f"No offline sheet for ticker {ticker!r} "
                       f"(available: {sorted(SHEET_BY_TICKER)})")
    if ticker not in _cache:
        df = pd.read_excel(XLS_PATH, sheet_name=SHEET_BY_TICKER[ticker],
                           parse_dates=["Date"])
        close = pd.to_numeric(df.set_index("Date")["Close"], errors="coerce")
        close = close.replace([np.inf, -np.inf], np.nan).dropna()
        close.index = pd.DatetimeIndex(close.index).tz_localize(None)
        _cache[ticker] = close.astype(float).sort_index()
    s = _cache[ticker]
    out = s.loc[(s.index >= pd.Timestamp(start)) & (s.index < pd.Timestamp(end))]
    if out.empty:
        raise SystemExit(f"No offline data for {ticker} between {start} and {end}.")
    return out.copy()
