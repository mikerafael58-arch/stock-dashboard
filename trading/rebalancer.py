from __future__ import annotations
import json
import os
from datetime import datetime

from trading.alpaca_client import get_account, get_positions, place_market_order

_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trade_log.json")


def load_trade_log() -> list[dict]:
    try:
        with open(_LOG_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save_trade_log(log: list):
    with open(_LOG_FILE, "w") as f:
        json.dump(log[-200:], f, indent=2)  # keep last 200 entries


def generate_rebalance_plan(ranked: list[dict], top_n: int = 5, allocation_pct: float = 0.90) -> dict:
    """
    Compare current Alpaca positions to desired top-N holdings.
    Returns a plan of buys and sells without executing anything.

    allocation_pct: fraction of portfolio equity to deploy (default 90%, keep 10% as cash buffer)
    """
    account = get_account()
    equity = account["equity"]
    deploy = equity * allocation_pct
    target_per_stock = deploy / top_n

    current_positions = {p["symbol"]: p for p in get_positions()}
    top_tickers = [s["ticker"] for s in ranked[:top_n]]

    sells = []
    buys = []

    # Sell anything not in top N
    for symbol, pos in current_positions.items():
        if symbol not in top_tickers:
            sells.append({
                "symbol": symbol,
                "action": "SELL",
                "reason": "Dropped out of top rankings",
                "current_value": pos["market_value"],
                "current_score": next((s["composite"] for s in ranked if s["ticker"] == symbol), None),
            })

    # Buy anything in top N not already held (or underweight)
    for stock in ranked[:top_n]:
        ticker = stock["ticker"]
        existing = current_positions.get(ticker)
        existing_value = existing["market_value"] if existing else 0.0
        needed = target_per_stock - existing_value

        if needed > 10:  # only buy if gap is more than $10
            buys.append({
                "symbol": ticker,
                "action": "BUY",
                "reason": f"Rank #{stock['rank']} — Score {stock['composite']:.0f}",
                "target_value": round(target_per_stock, 2),
                "current_value": round(existing_value, 2),
                "buy_amount": round(needed, 2),
                "composite_score": stock["composite"],
                "signal": "🟢 Buy" if stock["composite"] >= 65 else "🟡 Hold",
            })

    return {
        "account_equity": round(equity, 2),
        "deploy_amount": round(deploy, 2),
        "target_per_stock": round(target_per_stock, 2),
        "top_tickers": top_tickers,
        "sells": sells,
        "buys": buys,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def execute_rebalance(plan: dict) -> list[dict]:
    """Execute the rebalance plan — sells first, then buys."""
    results = []
    log = load_trade_log()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Sells first (free up cash)
    for sell in plan["sells"]:
        try:
            result = place_market_order(sell["symbol"], "sell", sell.get("current_value", 0))
            entry = {
                "timestamp": timestamp,
                "symbol": sell["symbol"],
                "action": "SELL",
                "amount": sell.get("current_value"),
                "reason": sell["reason"],
                "status": "submitted",
                "order_id": result.get("id"),
            }
            results.append({**entry, "success": True})
            log.append(entry)
        except Exception as e:
            entry = {
                "timestamp": timestamp,
                "symbol": sell["symbol"],
                "action": "SELL",
                "status": "failed",
                "error": str(e),
                "reason": sell["reason"],
            }
            results.append({**entry, "success": False})
            log.append(entry)

    # Buys after sells settle
    for buy in plan["buys"]:
        try:
            result = place_market_order(buy["symbol"], "buy", buy["buy_amount"])
            entry = {
                "timestamp": timestamp,
                "symbol": buy["symbol"],
                "action": "BUY",
                "amount": buy["buy_amount"],
                "reason": buy["reason"],
                "status": "submitted",
                "order_id": result.get("id"),
            }
            results.append({**entry, "success": True})
            log.append(entry)
        except Exception as e:
            entry = {
                "timestamp": timestamp,
                "symbol": buy["symbol"],
                "action": "BUY",
                "status": "failed",
                "error": str(e),
                "reason": buy["reason"],
            }
            results.append({**entry, "success": False})
            log.append(entry)

    _save_trade_log(log)
    return results


def get_performance_summary(positions: list[dict], account: dict) -> dict:
    """Summarise current portfolio performance."""
    if not positions:
        return {}

    total_pl = sum(p["unrealized_pl"] or 0 for p in positions)
    total_value = sum(p["market_value"] or 0 for p in positions)
    winners = [p for p in positions if (p["unrealized_pl"] or 0) > 0]
    losers = [p for p in positions if (p["unrealized_pl"] or 0) < 0]

    best = max(positions, key=lambda p: p["unrealized_plpc"] or -999) if positions else None
    worst = min(positions, key=lambda p: p["unrealized_plpc"] or 999) if positions else None

    return {
        "total_unrealized_pl": round(total_pl, 2),
        "total_invested": round(total_value, 2),
        "total_return_pct": round((total_pl / (total_value - total_pl)) * 100, 2) if total_value > total_pl else 0,
        "num_winners": len(winners),
        "num_losers": len(losers),
        "best_performer": {"symbol": best["symbol"], "pct": round(float(best["unrealized_plpc"] or 0) * 100, 2)} if best else None,
        "worst_performer": {"symbol": worst["symbol"], "pct": round(float(worst["unrealized_plpc"] or 0) * 100, 2)} if worst else None,
        "cash": round(account["cash"], 2),
        "equity": round(account["equity"], 2),
    }
