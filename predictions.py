from __future__ import annotations


def calculate_predictions(current_price: float, fund_metrics: dict) -> dict:
    """
    Generate return predictions across multiple time horizons.

    1-year  → analyst consensus price target (Wall Street forecasts via Yahoo Finance)
    3-year  → earnings growth model with dampened growth + mild PE compression
    5-year  → same model with heavier dampening (growth rarely sustains at full rate)

    Returns dict keyed by horizon ("1yr", "3yr", "5yr"), each containing:
      target_price, return_pct, annualized_return, confidence, method, notes
    """
    results = {}

    if not current_price or current_price <= 0:
        return results

    # ── 1-Year: Analyst Consensus ──────────────────────────────────────────────
    mean_target = fund_metrics.get("analyst_target_mean")
    high_target = fund_metrics.get("analyst_target_high")
    low_target = fund_metrics.get("analyst_target_low")
    num_analysts = fund_metrics.get("num_analysts")
    recommendation = fund_metrics.get("recommendation", "")

    if mean_target and mean_target > 0:
        upside = (mean_target / current_price - 1) * 100
        # Confidence based on analyst count
        if num_analysts and num_analysts >= 10:
            confidence = "High"
        elif num_analysts and num_analysts >= 5:
            confidence = "Medium"
        else:
            confidence = "Low"

        results["1yr"] = {
            "label": "1-Year",
            "target_price": round(mean_target, 2),
            "return_pct": round(upside, 1),
            "annualized_return": round(upside, 1),
            "high_target": round(high_target, 2) if high_target else None,
            "low_target": round(low_target, 2) if low_target else None,
            "num_analysts": num_analysts,
            "recommendation": recommendation.replace("_", " ").title() if recommendation else None,
            "confidence": confidence,
            "method": "Wall Street Analyst Consensus",
            "note": f"Based on {num_analysts or '?'} analyst price targets",
        }

    # ── 3-Year & 5-Year: Earnings Growth Model ─────────────────────────────────
    forward_pe = fund_metrics.get("forward_pe")
    earnings_growth = fund_metrics.get("earnings_growth")
    forward_eps = fund_metrics.get("forward_eps")
    trailing_eps = fund_metrics.get("trailing_eps")

    # Use forward EPS if available, otherwise derive from forward PE
    base_eps = None
    if forward_eps and forward_eps > 0:
        base_eps = forward_eps
    elif forward_pe and forward_pe > 0:
        base_eps = current_price / forward_pe

    if base_eps and base_eps > 0 and earnings_growth is not None:
        raw_growth = max(-0.30, min(0.60, earnings_growth))

        for years, dampen, pe_compress, label in [
            (3, 0.75, 0.90, "3yr"),
            (5, 0.55, 0.82, "5yr"),
        ]:
            projected_growth = raw_growth * dampen

            # Project EPS forward
            projected_eps = base_eps * ((1 + projected_growth) ** years)

            # PE: compress toward long-run mean (~20x) over time
            if forward_pe and forward_pe > 0:
                target_pe = forward_pe * pe_compress
                target_pe = max(10.0, min(target_pe, 35.0))
            else:
                target_pe = 18.0

            projected_price = projected_eps * target_pe
            if projected_price <= 0:
                continue

            total_return = (projected_price / current_price - 1) * 100
            annualized = ((projected_price / current_price) ** (1 / years) - 1) * 100

            # Confidence lower for longer horizon
            confidence = "Medium" if years == 3 else "Low"
            growth_pct = round(projected_growth * 100, 1)

            results[label] = {
                "label": f"{years}-Year",
                "target_price": round(projected_price, 2),
                "return_pct": round(total_return, 1),
                "annualized_return": round(annualized, 1),
                "high_target": None,
                "low_target": None,
                "num_analysts": None,
                "recommendation": None,
                "confidence": confidence,
                "method": "Earnings Growth Model",
                "note": f"Assumes {growth_pct}% annual EPS growth, {round(target_pe, 1)}x PE",
            }

    return results


def prediction_color(return_pct: float) -> str:
    if return_pct >= 20:
        return "#00c853"
    if return_pct >= 5:
        return "#64dd17"
    if return_pct >= -5:
        return "#ffab00"
    if return_pct >= -20:
        return "#ff6d00"
    return "#ff1744"


def confidence_badge(confidence: str) -> str:
    colors = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
    return colors.get(confidence, "⚪")
