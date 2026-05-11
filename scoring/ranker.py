from __future__ import annotations
import numpy as np


def _clamp(value, lo=0.0, hi=100.0):
    if value is None or np.isnan(float(value)):
        return None
    return max(lo, min(hi, float(value)))


def score_momentum(price_metrics: dict) -> dict:
    """
    Score momentum 0-100 based on:
    - 1m, 3m, 6m returns (trend direction)
    - Position vs SMA50 and SMA200 (structural trend)
    - RSI (not overbought/oversold extremes)
    """
    points = 0
    max_points = 0

    r1 = price_metrics.get("return_1m")
    r3 = price_metrics.get("return_3m")
    r6 = price_metrics.get("return_6m")
    vs50 = price_metrics.get("vs_sma50_pct")
    vs200 = price_metrics.get("vs_sma200_pct")
    rsi = price_metrics.get("rsi_14")

    if r1 is not None:
        max_points += 15
        points += _clamp(r1 * 1.5 + 7.5, 0, 15)

    if r3 is not None:
        max_points += 25
        points += _clamp(r3 * 1.0 + 12.5, 0, 25)

    if r6 is not None:
        max_points += 30
        points += _clamp(r6 * 0.75 + 15, 0, 30)

    if vs50 is not None:
        max_points += 15
        points += 15 if vs50 > 0 else 0

    if vs200 is not None:
        max_points += 15
        points += 15 if vs200 > 0 else 0

    # RSI: ideal range 45-70, penalize extremes
    if rsi is not None:
        max_points += 10
        if 45 <= rsi <= 70:
            points += 10
        elif 35 <= rsi < 45 or 70 < rsi <= 80:
            points += 5
        else:
            points += 0

    if max_points == 0:
        return {"score": None, "breakdown": {}}

    score = (points / max_points) * 100
    return {
        "score": round(_clamp(score), 1),
        "breakdown": {
            "1m_return": round(r1, 1) if r1 is not None else None,
            "3m_return": round(r3, 1) if r3 is not None else None,
            "6m_return": round(r6, 1) if r6 is not None else None,
            "above_sma50": vs50 > 0 if vs50 is not None else None,
            "above_sma200": vs200 > 0 if vs200 is not None else None,
            "rsi": round(rsi, 1) if rsi is not None else None,
        },
    }


def score_value(fund_metrics: dict) -> dict:
    """
    Score value 0-100. Lower P/E, P/B, EV/EBITDA = higher score.
    Uses sector-agnostic thresholds as a baseline.
    """
    points = 0
    max_points = 0

    pe = fund_metrics.get("pe_ratio")
    fpe = fund_metrics.get("forward_pe")
    pb = fund_metrics.get("pb_ratio")
    ev_ebitda = fund_metrics.get("ev_ebitda")

    # Trailing P/E
    if pe is not None and pe > 0:
        max_points += 30
        if pe < 15:
            points += 30
        elif pe < 20:
            points += 22
        elif pe < 25:
            points += 15
        elif pe < 35:
            points += 8
        else:
            points += 0

    # Forward P/E (forward-looking, weighted more)
    if fpe is not None and fpe > 0:
        max_points += 35
        if fpe < 15:
            points += 35
        elif fpe < 20:
            points += 26
        elif fpe < 25:
            points += 18
        elif fpe < 35:
            points += 9
        else:
            points += 0

    # P/B ratio
    if pb is not None and pb > 0:
        max_points += 15
        if pb < 2:
            points += 15
        elif pb < 4:
            points += 10
        elif pb < 8:
            points += 5
        else:
            points += 0

    # EV/EBITDA
    if ev_ebitda is not None and ev_ebitda > 0:
        max_points += 20
        if ev_ebitda < 10:
            points += 20
        elif ev_ebitda < 15:
            points += 14
        elif ev_ebitda < 20:
            points += 8
        else:
            points += 0

    if max_points == 0:
        return {"score": None, "breakdown": {}}

    score = (points / max_points) * 100
    return {
        "score": round(_clamp(score), 1),
        "breakdown": {
            "trailing_pe": round(pe, 1) if pe else None,
            "forward_pe": round(fpe, 1) if fpe else None,
            "pb_ratio": round(pb, 2) if pb else None,
            "ev_ebitda": round(ev_ebitda, 1) if ev_ebitda else None,
        },
    }


def score_quality(fund_metrics: dict) -> dict:
    """
    Score quality 0-100 based on:
    - Revenue and earnings growth
    - Profit margin
    - Return on equity
    - Debt-to-equity (lower is safer)
    """
    points = 0
    max_points = 0

    rev_growth = fund_metrics.get("revenue_growth")
    earn_growth = fund_metrics.get("earnings_growth")
    margin = fund_metrics.get("profit_margin")
    roe = fund_metrics.get("roe")
    de = fund_metrics.get("debt_to_equity")

    if rev_growth is not None:
        max_points += 25
        pct = rev_growth * 100
        if pct >= 20:
            points += 25
        elif pct >= 10:
            points += 18
        elif pct >= 5:
            points += 10
        elif pct >= 0:
            points += 4
        else:
            points += 0

    if earn_growth is not None:
        max_points += 25
        pct = earn_growth * 100
        if pct >= 20:
            points += 25
        elif pct >= 10:
            points += 18
        elif pct >= 5:
            points += 10
        elif pct >= 0:
            points += 4
        else:
            points += 0

    if margin is not None:
        max_points += 25
        pct = margin * 100
        if pct >= 25:
            points += 25
        elif pct >= 15:
            points += 18
        elif pct >= 8:
            points += 10
        elif pct >= 2:
            points += 4
        else:
            points += 0

    if roe is not None:
        max_points += 15
        pct = roe * 100
        if pct >= 20:
            points += 15
        elif pct >= 12:
            points += 10
        elif pct >= 5:
            points += 5
        else:
            points += 0

    if de is not None:
        max_points += 10
        if de < 30:
            points += 10
        elif de < 80:
            points += 7
        elif de < 150:
            points += 4
        else:
            points += 0

    if max_points == 0:
        return {"score": None, "breakdown": {}}

    score = (points / max_points) * 100
    return {
        "score": round(_clamp(score), 1),
        "breakdown": {
            "revenue_growth_pct": round(rev_growth * 100, 1) if rev_growth is not None else None,
            "earnings_growth_pct": round(earn_growth * 100, 1) if earn_growth is not None else None,
            "profit_margin_pct": round(margin * 100, 1) if margin is not None else None,
            "roe_pct": round(roe * 100, 1) if roe is not None else None,
            "debt_to_equity": round(de, 1) if de is not None else None,
        },
    }


def composite_score(momentum: dict, value: dict, quality: dict, weights=(0.35, 0.30, 0.35)) -> float | None:
    """Weighted average of the three factor scores."""
    scores = [momentum.get("score"), value.get("score"), quality.get("score")]
    w = list(weights)

    # If any score is missing, redistribute its weight evenly
    valid = [(s, wt) for s, wt in zip(scores, w) if s is not None]
    if not valid:
        return None

    total_weight = sum(wt for _, wt in valid)
    return round(sum(s * wt for s, wt in valid) / total_weight, 1)


def rank_stocks(stock_scores: list[dict]) -> list[dict]:
    """Sort stocks by composite score descending and add rank."""
    sorted_stocks = sorted(
        [s for s in stock_scores if s.get("composite") is not None],
        key=lambda x: x["composite"],
        reverse=True,
    )
    for i, stock in enumerate(sorted_stocks, 1):
        stock["rank"] = i
    return sorted_stocks
