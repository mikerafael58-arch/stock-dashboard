from __future__ import annotations
import json
import os
from datetime import datetime

_BASE = os.path.dirname(os.path.abspath(__file__))
_HISTORY_FILE = os.path.join(_BASE, "score_history.json")
_ALERTS_FILE = os.path.join(_BASE, "price_alerts.json")
_WATCHLISTS_FILE = os.path.join(_BASE, "saved_watchlists.json")

PRESET_WATCHLISTS = {
    "🏆 Default (Top 20)": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM",
        "JNJ", "V", "UNH", "XOM", "PG", "MA", "HD", "TSLA", "MRK", "ABBV", "LLY", "AVGO",
    ],
    "💻 Tech Giants": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
        "AVGO", "CRM", "ORCL", "ADBE", "QCOM", "TXN", "INTC",
    ],
    "💰 Dividend Kings": [
        "JNJ", "PG", "KO", "PEP", "T", "VZ", "XOM", "CVX",
        "IBM", "MO", "WMT", "MCD", "CL", "MMM", "ED",
    ],
    "🏦 Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "V", "MA",
        "AXP", "C", "USB", "COF", "SCHW", "TFC", "PNC",
    ],
    "🏥 Healthcare": [
        "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT",
        "BMY", "AMGN", "MDT", "ISRG", "CVS", "CI", "HUM",
    ],
    "⚡ Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO",
        "OXY", "HAL", "DVN", "HES", "BKR", "APA", "FANG",
    ],
}


def _read(path: str) -> list | dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return [] if path == _HISTORY_FILE else {}


def _write(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Score history ──────────────────────────────────────────────────────────────

def save_snapshot(ranked: list):
    history = _read(_HISTORY_FILE)
    if not isinstance(history, list):
        history = []
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "scores": {
            s["ticker"]: {
                "composite": s.get("composite"),
                "momentum": s.get("momentum", {}).get("score"),
                "value": s.get("value", {}).get("score"),
                "quality": s.get("quality", {}).get("score"),
            }
            for s in ranked
        },
    }
    history.append(snapshot)
    history = history[-90:]  # keep last 90 snapshots
    _write(_HISTORY_FILE, history)


def load_history() -> list:
    data = _read(_HISTORY_FILE)
    return data if isinstance(data, list) else []


# ── Price alerts ───────────────────────────────────────────────────────────────

def load_alerts() -> dict:
    data = _read(_ALERTS_FILE)
    return data if isinstance(data, dict) else {}


def save_alert(ticker: str, target: float, condition: str):
    alerts = load_alerts()
    alerts[ticker] = {"target": target, "condition": condition}
    _write(_ALERTS_FILE, alerts)


def delete_alert(ticker: str):
    alerts = load_alerts()
    alerts.pop(ticker, None)
    _write(_ALERTS_FILE, alerts)


def check_alerts(ranked: list, alerts: dict) -> list:
    triggered = []
    price_map = {s["ticker"]: s["price_metrics"].get("current_price") for s in ranked}
    for ticker, alert in alerts.items():
        price = price_map.get(ticker)
        if price is None:
            continue
        target = alert["target"]
        condition = alert["condition"]
        if condition == "above" and price >= target:
            triggered.append({"ticker": ticker, "price": price, "target": target, "condition": condition})
        elif condition == "below" and price <= target:
            triggered.append({"ticker": ticker, "price": price, "target": target, "condition": condition})
    return triggered


# ── Custom watchlists ──────────────────────────────────────────────────────────

def load_saved_watchlists() -> dict:
    data = _read(_WATCHLISTS_FILE)
    return data if isinstance(data, dict) else {}


def save_watchlist(name: str, tickers: list):
    wl = load_saved_watchlists()
    wl[name] = tickers
    _write(_WATCHLISTS_FILE, wl)


def delete_watchlist(name: str):
    wl = load_saved_watchlists()
    wl.pop(name, None)
    _write(_WATCHLISTS_FILE, wl)
