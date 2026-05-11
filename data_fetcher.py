from __future__ import annotations
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "BRK-B", "JPM", "JNJ", "V",
    "UNH", "XOM", "PG", "MA", "HD",
    "TSLA", "MRK", "ABBV", "LLY", "AVGO",
]


def fetch_stock_data(ticker: str, period: str = "1y") -> dict:
    """Fetch price history and fundamentals for a single ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info

        if hist.empty:
            return None

        return {
            "ticker": ticker,
            "history": hist,
            "info": info,
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
        }
    except Exception:
        return None


def fetch_watchlist(tickers: list[str]) -> dict:
    """Fetch data for a list of tickers. Returns dict keyed by ticker."""
    results = {}
    for ticker in tickers:
        data = fetch_stock_data(ticker)
        if data:
            results[ticker] = data
    return results


def get_price_metrics(hist: pd.DataFrame) -> dict:
    """Extract price-based metrics from history DataFrame."""
    if hist.empty or len(hist) < 5:
        return {}

    close = hist["Close"]
    current = float(close.iloc[-1])

    def pct_change(days):
        if len(close) <= days:
            return None
        return float((current - close.iloc[-days]) / close.iloc[-days] * 100)

    sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    hi_52 = float(close.rolling(252).max().iloc[-1]) if len(close) >= 52 else float(close.max())
    lo_52 = float(close.rolling(252).min().iloc[-1]) if len(close) >= 52 else float(close.min())

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = float(100 - (100 / (1 + rs)).iloc[-1]) if not rs.empty else None

    return {
        "current_price": current,
        "return_1m": pct_change(21),
        "return_3m": pct_change(63),
        "return_6m": pct_change(126),
        "return_1y": pct_change(252),
        "sma_50": sma_50,
        "sma_200": sma_200,
        "vs_sma50_pct": ((current / sma_50) - 1) * 100 if sma_50 else None,
        "vs_sma200_pct": ((current / sma_200) - 1) * 100 if sma_200 else None,
        "hi_52w": hi_52,
        "lo_52w": lo_52,
        "pct_from_52w_hi": ((current / hi_52) - 1) * 100 if hi_52 else None,
        "rsi_14": rsi,
    }


def get_fundamental_metrics(info: dict) -> dict:
    """Extract fundamental metrics from yfinance info dict."""
    return {
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "pb_ratio": info.get("priceToBook"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "profit_margin": info.get("profitMargins"),
        "roe": info.get("returnOnEquity"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "dividend_yield": info.get("dividendYield"),
        "market_cap": info.get("marketCap"),
        "beta": info.get("beta"),
        # Analyst targets
        "analyst_target_mean": info.get("targetMeanPrice"),
        "analyst_target_high": info.get("targetHighPrice"),
        "analyst_target_low": info.get("targetLowPrice"),
        "analyst_target_median": info.get("targetMedianPrice"),
        "num_analysts": info.get("numberOfAnalystOpinions"),
        "recommendation": info.get("recommendationKey"),
        # EPS for DCF model
        "trailing_eps": info.get("trailingEps"),
        "forward_eps": info.get("forwardEps"),
    }
