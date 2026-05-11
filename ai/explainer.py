import anthropic
import json

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


_SYSTEM_PROMPT = """You are a concise, data-driven stock analyst. When given a stock's quantitative scores and metrics, you:
1. Explain in plain English why it ranks the way it does
2. Highlight the strongest signal (positive or negative)
3. Flag any red flags or risks
4. Give a one-sentence forward outlook based purely on the data provided

Keep your entire response under 200 words. Be direct. Avoid generic disclaimers."""


def explain_stock(ticker: str, name: str, sector: str, scores: dict, price_metrics: dict, fund_metrics: dict) -> str:
    """Use Claude to generate a plain-English explanation of a stock's scores."""
    client = _get_client()

    data_summary = {
        "ticker": ticker,
        "company": name,
        "sector": sector,
        "composite_score": scores.get("composite"),
        "momentum_score": scores.get("momentum", {}).get("score"),
        "value_score": scores.get("value", {}).get("score"),
        "quality_score": scores.get("quality", {}).get("score"),
        "momentum_details": scores.get("momentum", {}).get("breakdown", {}),
        "value_details": scores.get("value", {}).get("breakdown", {}),
        "quality_details": scores.get("quality", {}).get("breakdown", {}),
        "current_price": price_metrics.get("current_price"),
        "52w_high": price_metrics.get("hi_52w"),
        "52w_low": price_metrics.get("lo_52w"),
        "beta": fund_metrics.get("beta"),
        "market_cap": fund_metrics.get("market_cap"),
    }

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"Analyze this stock:\n\n{json.dumps(data_summary, indent=2)}",
            }
        ],
    )

    return message.content[0].text


def generate_market_summary(ranked_stocks: list[dict]) -> str:
    """Use Claude to generate a brief overview of the overall watchlist."""
    client = _get_client()

    top5 = ranked_stocks[:5]
    bottom5 = ranked_stocks[-5:] if len(ranked_stocks) > 5 else []

    summary_data = {
        "total_stocks_analyzed": len(ranked_stocks),
        "top_5_ranked": [
            {"ticker": s["ticker"], "composite": s["composite"], "sector": s.get("sector")}
            for s in top5
        ],
        "bottom_5_ranked": [
            {"ticker": s["ticker"], "composite": s["composite"], "sector": s.get("sector")}
            for s in bottom5
        ],
        "avg_momentum": round(
            sum(s.get("momentum", {}).get("score", 0) or 0 for s in ranked_stocks) / len(ranked_stocks), 1
        ) if ranked_stocks else None,
        "avg_quality": round(
            sum(s.get("quality", {}).get("score", 0) or 0 for s in ranked_stocks) / len(ranked_stocks), 1
        ) if ranked_stocks else None,
    }

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=[
            {
                "type": "text",
                "text": "You are a concise market analyst. Summarize the state of a watchlist in 3-4 sentences. Focus on what the data says about overall market momentum, which sectors are leading, and any notable patterns. Under 150 words.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"Summarize this watchlist:\n\n{json.dumps(summary_data, indent=2)}",
            }
        ],
    )

    return message.content[0].text
