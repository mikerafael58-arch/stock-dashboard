from __future__ import annotations
import json
import os
from datetime import datetime

from trading.alpaca_client import get_account, get_positions, place_market_order, close_position

_CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strategy_config.json")
_PERF_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strategy_performance.json")


DEFAULT_CONFIG = {
    "top_n": 5,
    "min_composite_score": 45,
    "allocation_method": "score_weighted",  # or "equal_weight"
    "deploy_pct": 90,                        # % of equity to invest
    "stop_loss_pct": -15,                    # sell if position drops this %
    "rebalance_on_run": True,
}


def load_config() -> dict:
    try:
        with open(_CONFIG_FILE) as f:
            saved = json.load(f)
            return {**DEFAULT_CONFIG, **saved}
    except Exception:
        return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    with open(_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def load_performance() -> list[dict]:
    try:
        with open(_PERF_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save_performance(entries: list):
    with open(_PERF_FILE, "w") as f:
        json.dump(entries[-365:], f, indent=2)  # keep 1 year of daily snapshots


def snapshot_performance(account: dict, positions: list[dict], strategy_tickers: list[str]):
    """Save a daily performance snapshot for tracking over time."""
    entries = load_performance()
    total_pl = sum(p["unrealized_pl"] or 0 for p in positions)
    entries.append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "portfolio_value": account["portfolio_value"],
        "equity": account["equity"],
        "cash": account["cash"],
        "unrealized_pl": round(total_pl, 2),
        "num_positions": len(positions),
        "holdings": strategy_tickers,
    })
    _save_performance(entries)


def generate_signals(
    ranked: list[dict],
    config: dict,
) -> dict:
    """
    Core strategy engine. Generates buy/sell signals from scoring data.

    Returns:
        dict with keys: buys, sells, holds, stops, summary
    """
    account = get_account()
    positions = {p["symbol"]: p for p in get_positions()}
    equity = account["equity"]
    deploy = equity * (config["deploy_pct"] / 100)

    top_n = config["top_n"]
    min_score = config["min_composite_score"]
    method = config["allocation_method"]
    stop_loss = config["stop_loss_pct"]

    # Filter to stocks meeting minimum score
    qualified = [s for s in ranked if (s.get("composite") or 0) >= min_score]
    top_stocks = qualified[:top_n]
    top_tickers = {s["ticker"] for s in top_stocks}

    if not top_stocks:
        return {
            "buys": [], "sells": list(positions.keys()), "holds": [], "stops": [],
            "summary": f"No stocks meet the minimum score of {min_score}. Consider lowering the threshold.",
            "account": account,
        }

    # ── Target allocations ─────────────────────────────────────────────────────
    if method == "score_weighted":
        total_score = sum(s["composite"] for s in top_stocks)
        target_alloc = {
            s["ticker"]: (s["composite"] / total_score) * deploy
            for s in top_stocks
        }
    else:  # equal_weight
        per_stock = deploy / len(top_stocks)
        target_alloc = {s["ticker"]: per_stock for s in top_stocks}

    # ── Sell signals ───────────────────────────────────────────────────────────
    sells = []
    stops = []

    for symbol, pos in positions.items():
        pl_pct = (pos["unrealized_plpc"] or 0) * 100

        if pl_pct <= stop_loss:
            stops.append({
                "symbol": symbol,
                "action": "SELL",
                "reason": f"🛑 Stop loss — position down {pl_pct:.1f}%",
                "current_value": pos["market_value"],
                "pl_pct": round(pl_pct, 2),
                "urgency": "HIGH",
            })
        elif symbol not in top_tickers:
            curr_score = next((s["composite"] for s in ranked if s["ticker"] == symbol), None)
            score_str = f"Score dropped to {curr_score:.0f}" if curr_score else "Fell out of rankings"
            sells.append({
                "symbol": symbol,
                "action": "SELL",
                "reason": f"📉 {score_str} — no longer in top {top_n}",
                "current_value": pos["market_value"],
                "pl_pct": round(pl_pct, 2),
                "urgency": "NORMAL",
            })

    # ── Buy / adjust signals ───────────────────────────────────────────────────
    buys = []
    holds = []

    for stock in top_stocks:
        ticker = stock["ticker"]
        target = target_alloc[ticker]
        current_pos = positions.get(ticker)
        current_value = current_pos["market_value"] if current_pos else 0.0
        gap = target - current_value
        pl_pct = (current_pos["unrealized_plpc"] or 0) * 100 if current_pos else 0.0

        score = stock["composite"]
        signal = "🟢 Buy" if score >= 65 else "🟡 Hold"
        rank_str = f"Rank #{stock['rank']} | Score {score:.0f}"

        if gap > 25:  # buy threshold — only act if gap > $25
            buys.append({
                "symbol": ticker,
                "action": "BUY",
                "reason": f"{signal} — {rank_str}",
                "buy_amount": round(gap, 2),
                "target_value": round(target, 2),
                "current_value": round(current_value, 2),
                "composite_score": score,
                "momentum_score": stock["momentum"].get("score"),
                "value_score": stock["value"].get("score"),
                "quality_score": stock["quality"].get("score"),
                "urgency": "HIGH" if score >= 70 else "NORMAL",
            })
        else:
            holds.append({
                "symbol": ticker,
                "reason": f"✅ On target — {rank_str}",
                "current_value": round(current_value, 2),
                "target_value": round(target, 2),
                "pl_pct": round(pl_pct, 2),
                "composite_score": score,
            })

    n_actions = len(stops) + len(sells) + len(buys)
    summary_parts = []
    if stops:
        summary_parts.append(f"{len(stops)} stop-loss sell(s)")
    if sells:
        summary_parts.append(f"{len(sells)} rotation sell(s)")
    if buys:
        summary_parts.append(f"{len(buys)} buy order(s)")
    if holds:
        summary_parts.append(f"{len(holds)} position(s) on target")

    summary = " · ".join(summary_parts) if summary_parts else "Portfolio is fully aligned. No action needed."

    return {
        "buys": buys,
        "sells": sells,
        "holds": holds,
        "stops": stops,
        "summary": summary,
        "account": account,
        "top_tickers": list(top_tickers),
        "target_allocations": {k: round(v, 2) for k, v in target_alloc.items()},
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def execute_signals(signals: dict) -> list[dict]:
    """Execute buy/sell signals. Stops and sells run first, then buys."""
    results = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Stops first (highest urgency)
    for s in signals.get("stops", []):
        try:
            result = close_position(s["symbol"])
            results.append({"timestamp": timestamp, "symbol": s["symbol"], "action": "SELL (STOP)",
                            "reason": s["reason"], "status": "submitted", "success": True})
        except Exception as e:
            results.append({"timestamp": timestamp, "symbol": s["symbol"], "action": "SELL (STOP)",
                            "reason": s["reason"], "status": "failed", "error": str(e), "success": False})

    # Regular sells
    for s in signals.get("sells", []):
        try:
            close_position(s["symbol"])
            results.append({"timestamp": timestamp, "symbol": s["symbol"], "action": "SELL",
                            "reason": s["reason"], "status": "submitted", "success": True})
        except Exception as e:
            results.append({"timestamp": timestamp, "symbol": s["symbol"], "action": "SELL",
                            "reason": s["reason"], "status": "failed", "error": str(e), "success": False})

    # Buys
    for b in signals.get("buys", []):
        try:
            place_market_order(b["symbol"], "buy", b["buy_amount"])
            results.append({"timestamp": timestamp, "symbol": b["symbol"], "action": "BUY",
                            "amount": b["buy_amount"], "reason": b["reason"],
                            "status": "submitted", "success": True})
        except Exception as e:
            results.append({"timestamp": timestamp, "symbol": b["symbol"], "action": "BUY",
                            "amount": b["buy_amount"], "reason": b["reason"],
                            "status": "failed", "error": str(e), "success": False})

    return results
