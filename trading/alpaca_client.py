from __future__ import annotations
import os
from datetime import datetime, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

_client: TradingClient | None = None


def _get_secret(key: str) -> str:
    """Read from st.secrets first (Streamlit Cloud), fall back to os.getenv (local)."""
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return str(val).strip()
    except Exception:
        pass
    return os.getenv(key, "").strip()


def get_client() -> TradingClient:
    global _client
    api_key = _get_secret("ALPACA_API_KEY")
    secret_key = _get_secret("ALPACA_SECRET_KEY")
    if not api_key or not secret_key or "your_alpaca" in api_key:
        raise ValueError("Alpaca API keys not configured. Add ALPACA_API_KEY and ALPACA_SECRET_KEY to Streamlit Secrets.")
    # Always rebuild if keys changed
    if _client is None:
        _client = TradingClient(api_key, secret_key, paper=True)
    return _client


def get_account() -> dict:
    client = get_client()
    acct = client.get_account()
    return {
        "equity": float(acct.equity),
        "cash": float(acct.cash),
        "portfolio_value": float(acct.portfolio_value),
        "buying_power": float(acct.buying_power),
        "daytrade_count": acct.daytrade_count,
        "status": acct.status,
        "currency": acct.currency,
    }


def get_positions() -> list[dict]:
    client = get_client()
    positions = client.get_all_positions()
    result = []
    for p in positions:
        result.append({
            "symbol": p.symbol,
            "qty": float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price) if p.current_price else None,
            "market_value": float(p.market_value) if p.market_value else None,
            "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl else None,
            "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc else None,
            "change_today": float(p.change_today) if p.change_today else None,
        })
    return result


def get_recent_orders(limit: int = 50) -> list[dict]:
    client = get_client()
    request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
    orders = client.get_orders(filter=request)
    result = []
    for o in orders:
        result.append({
            "id": str(o.id),
            "symbol": o.symbol,
            "side": str(o.side).replace("OrderSide.", ""),
            "qty": float(o.qty) if o.qty else None,
            "notional": float(o.notional) if o.notional else None,
            "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
            "filled_qty": float(o.filled_qty) if o.filled_qty else None,
            "status": str(o.status).replace("OrderStatus.", ""),
            "created_at": o.created_at.strftime("%Y-%m-%d %H:%M") if o.created_at else None,
            "type": str(o.type).replace("OrderType.", ""),
        })
    return result


def place_market_order(symbol: str, side: str, notional: float) -> dict:
    """Place a dollar-amount market order. side = 'buy' or 'sell'."""
    client = get_client()
    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

    if order_side == OrderSide.SELL:
        # For sells, use qty from current position
        positions = {p["symbol"]: p for p in get_positions()}
        if symbol not in positions:
            raise ValueError(f"No position in {symbol} to sell.")
        qty = positions[symbol]["qty"]
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
    else:
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(notional, 2),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )

    order = client.submit_order(req)
    return {
        "id": str(order.id),
        "symbol": order.symbol,
        "side": side,
        "notional": notional,
        "status": str(order.status),
        "created_at": order.created_at.strftime("%Y-%m-%d %H:%M") if order.created_at else None,
    }


def cancel_all_orders():
    client = get_client()
    client.cancel_orders()


def close_position(symbol: str) -> dict:
    client = get_client()
    order = client.close_position(symbol)
    return {"symbol": symbol, "status": "closed", "order_id": str(order.id)}
