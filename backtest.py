from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf


def _momentum_score(series: pd.Series, date) -> float | None:
    hist = series.loc[:date].dropna()
    if len(hist) < 63:
        return None
    current = float(hist.iloc[-1])
    score = 0.0
    count = 0
    for lookback in [21, 63, 126]:
        if len(hist) >= lookback:
            past = float(hist.iloc[-lookback])
            if past > 0:
                score += (current / past - 1) * 100
                count += 1
    return score / count if count > 0 else None


def _max_drawdown(cum_returns: pd.Series) -> float:
    roll_max = cum_returns.expanding().max()
    drawdown = (cum_returns - roll_max) / roll_max
    return float(drawdown.min() * 100)


def _sharpe(monthly_returns: list, risk_free_monthly: float = 0.05 / 12) -> float:
    r = pd.Series(monthly_returns)
    excess = r - risk_free_monthly
    std = excess.std()
    return float((excess.mean() / std) * (12 ** 0.5)) if std > 0 else 0.0


def run_backtest(tickers: list[str], top_n: int = 5, years: int = 2) -> dict:
    """
    Monthly momentum backtest: rank stocks by recent momentum each month,
    hold the top N equal-weighted, measure next-month return vs SPY.
    """
    symbols = list(set(tickers + ["SPY"]))

    prices: pd.DataFrame = yf.download(
        symbols,
        period=f"{years + 1}y",
        progress=False,
        auto_adjust=True,
    )["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna(how="all")

    # Month-end rebalance dates
    month_ends = prices.resample("ME").last().index.tolist()
    if len(month_ends) < 8:
        return {"error": "Not enough historical data for backtest."}

    port_returns: list[float] = []
    spy_returns: list[float] = []
    dates: list = []
    holdings_log: list[list[str]] = []

    for i in range(6, len(month_ends) - 1):
        date = month_ends[i]
        next_date = month_ends[i + 1]

        scores: dict[str, float] = {}
        for ticker in tickers:
            if ticker not in prices.columns:
                continue
            score = _momentum_score(prices[ticker], date)
            if score is not None:
                scores[ticker] = score

        if len(scores) < top_n:
            continue

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_tickers = [t for t, _ in top]

        # Portfolio return
        port_ret = 0.0
        valid = 0
        for ticker in top_tickers:
            if ticker not in prices.columns:
                continue
            p0 = prices[ticker].asof(date)
            p1 = prices[ticker].asof(next_date)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                port_ret += (float(p1) / float(p0) - 1)
                valid += 1
        if valid == 0:
            continue
        port_ret /= valid

        # SPY return
        spy_ret = 0.0
        if "SPY" in prices.columns:
            s0 = prices["SPY"].asof(date)
            s1 = prices["SPY"].asof(next_date)
            if pd.notna(s0) and pd.notna(s1) and s0 > 0:
                spy_ret = float(s1) / float(s0) - 1

        port_returns.append(port_ret)
        spy_returns.append(spy_ret)
        dates.append(next_date)
        holdings_log.append(top_tickers)

    if not port_returns:
        return {"error": "Backtest produced no results. Try more tickers or longer period."}

    port_cum = (1 + pd.Series(port_returns)).cumprod()
    spy_cum = (1 + pd.Series(spy_returns)).cumprod()

    port_total = (float(port_cum.iloc[-1]) - 1) * 100
    spy_total = (float(spy_cum.iloc[-1]) - 1) * 100
    port_dd = _max_drawdown(port_cum)
    spy_dd = _max_drawdown(spy_cum)
    port_sharpe = _sharpe(port_returns)
    spy_sharpe = _sharpe(spy_returns)

    winning_months = sum(p > s for p, s in zip(port_returns, spy_returns))
    win_rate = winning_months / len(port_returns) * 100

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "portfolio_cumulative": port_cum.tolist(),
        "spy_cumulative": spy_cum.tolist(),
        "portfolio_monthly_returns": port_returns,
        "spy_monthly_returns": spy_returns,
        "holdings_log": holdings_log,
        "summary": {
            "portfolio_total_return": round(port_total, 1),
            "spy_total_return": round(spy_total, 1),
            "outperformance": round(port_total - spy_total, 1),
            "portfolio_max_drawdown": round(port_dd, 1),
            "spy_max_drawdown": round(spy_dd, 1),
            "portfolio_sharpe": round(port_sharpe, 2),
            "spy_sharpe": round(spy_sharpe, 2),
            "win_rate_vs_spy": round(win_rate, 1),
            "months_tested": len(port_returns),
        },
    }
