import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

from data_fetcher import (
    fetch_watchlist,
    get_price_metrics,
    get_fundamental_metrics,
    DEFAULT_WATCHLIST,
)
from scoring.ranker import score_momentum, score_value, score_quality, composite_score, rank_stocks
from ai.explainer import explain_stock, generate_market_summary

st.set_page_config(
    page_title="Stock Ranking Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .score-high { color: #00c853; font-weight: bold; font-size: 1.1rem; }
    .score-mid  { color: #ffab00; font-weight: bold; font-size: 1.1rem; }
    .score-low  { color: #ff1744; font-weight: bold; font-size: 1.1rem; }
    .metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .rank-badge { font-size: 1.4rem; font-weight: bold; color: #1976d2; }
</style>
""", unsafe_allow_html=True)


def score_color(score):
    if score is None:
        return "score-mid"
    if score >= 65:
        return "score-high"
    if score >= 40:
        return "score-mid"
    return "score-low"


def fmt_score(score):
    if score is None:
        return "N/A"
    return f"{score:.0f}"


def fmt_pct(val):
    if val is None:
        return "—"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}%"


def fmt_val(val, decimals=1, prefix="", suffix=""):
    if val is None:
        return "—"
    return f"{prefix}{val:.{decimals}f}{suffix}"


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Stock Dashboard")
    st.markdown("---")

    st.subheader("Watchlist")
    watchlist_input = st.text_area(
        "Tickers (one per line or comma-separated)",
        value="\n".join(DEFAULT_WATCHLIST),
        height=300,
    )

    raw = watchlist_input.replace(",", "\n").split("\n")
    tickers = [t.strip().upper() for t in raw if t.strip()]

    st.markdown("---")
    st.subheader("Factor Weights")
    w_momentum = st.slider("Momentum", 0, 100, 35)
    w_value = st.slider("Value", 0, 100, 30)
    w_quality = st.slider("Quality", 0, 100, 35)
    total_w = w_momentum + w_value + w_quality
    if total_w == 0:
        st.error("Weights must sum to > 0")
    else:
        weights = (w_momentum / total_w, w_value / total_w, w_quality / total_w)

    st.markdown("---")
    run = st.button("🔄 Refresh Data", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption("Data via Yahoo Finance · AI via Claude · Not financial advice")


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def load_and_score(tickers_tuple, weights_tuple):
    tickers = list(tickers_tuple)
    weights = weights_tuple

    raw_data = fetch_watchlist(tickers)
    results = []

    for ticker, data in raw_data.items():
        price_m = get_price_metrics(data["history"])
        fund_m = get_fundamental_metrics(data["info"])

        mom = score_momentum(price_m)
        val = score_value(fund_m)
        qual = score_quality(fund_m)
        comp = composite_score(mom, val, qual, weights)

        results.append({
            "ticker": ticker,
            "name": data["name"],
            "sector": data["sector"],
            "industry": data["industry"],
            "momentum": mom,
            "value": val,
            "quality": qual,
            "composite": comp,
            "price_metrics": price_m,
            "fund_metrics": fund_m,
        })

    return rank_stocks(results), raw_data


if run:
    st.cache_data.clear()

with st.spinner("Fetching market data..."):
    try:
        ranked, raw_data = load_and_score(tuple(tickers), weights)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

if not ranked:
    st.warning("No data returned. Check your tickers and try again.")
    st.stop()


# ── Main tabs ──────────────────────────────────────────────────────────────────

tab_overview, tab_detail, tab_charts, tab_ai = st.tabs(
    ["📊 Rankings", "🔍 Stock Detail", "📉 Charts", "🤖 AI Analysis"]
)


# ── Tab 1: Rankings ────────────────────────────────────────────────────────────

with tab_overview:
    st.header("Multi-Factor Stock Rankings")

    with st.expander("🔎 Filter Stocks", expanded=False):
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        with fc1:
            sector_options = ["All"] + sorted(set(s["sector"] for s in ranked if s["sector"] != "Unknown"))
            sector_filter = st.selectbox("Sector", sector_options)
        with fc2:
            min_composite = st.slider("Min Composite", 0, 100, 0)
        with fc3:
            min_momentum = st.slider("Min Momentum", 0, 100, 0)
        with fc4:
            min_value = st.slider("Min Value", 0, 100, 0)
        with fc5:
            min_quality = st.slider("Min Quality", 0, 100, 0)

        st.caption(f"Showing stocks where Composite ≥ {min_composite}, Momentum ≥ {min_momentum}, Value ≥ {min_value}, Quality ≥ {min_quality}")

    def passes_filters(s):
        if sector_filter != "All" and s["sector"] != sector_filter:
            return False
        if (s["composite"] or 0) < min_composite:
            return False
        if (s["momentum"].get("score") or 0) < min_momentum:
            return False
        if (s["value"].get("score") or 0) < min_value:
            return False
        if (s["quality"].get("score") or 0) < min_quality:
            return False
        return True

    filtered = [s for s in ranked if passes_filters(s)]
    st.caption(f"{len(filtered)} of {len(ranked)} stocks match your filters")

    rows = []
    for s in filtered:
        rows.append({
            "Rank": s.get("rank", "—"),
            "Ticker": s["ticker"],
            "Name": s["name"][:30],
            "Sector": s["sector"],
            "Price": fmt_val(s["price_metrics"].get("current_price"), 2, "$"),
            "Composite": fmt_score(s["composite"]),
            "Momentum": fmt_score(s["momentum"].get("score")),
            "Value": fmt_score(s["value"].get("score")),
            "Quality": fmt_score(s["quality"].get("score")),
            "1M Ret": fmt_pct(s["price_metrics"].get("return_1m")),
            "3M Ret": fmt_pct(s["price_metrics"].get("return_3m")),
            "6M Ret": fmt_pct(s["price_metrics"].get("return_6m")),
            "RSI": fmt_val(s["price_metrics"].get("rsi_14"), 1),
            "Fwd P/E": fmt_val(s["fund_metrics"].get("forward_pe"), 1),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, height=600)

    st.markdown("---")
    st.subheader("Score Distribution")
    fig_scatter = px.scatter(
        pd.DataFrame([{
            "ticker": s["ticker"],
            "momentum": s["momentum"].get("score"),
            "value": s["value"].get("score"),
            "quality": s["quality"].get("score"),
            "composite": s["composite"],
            "sector": s["sector"],
        } for s in ranked if s["composite"] is not None]),
        x="momentum",
        y="quality",
        size="composite",
        color="sector",
        hover_name="ticker",
        labels={"momentum": "Momentum Score", "quality": "Quality Score"},
        title="Momentum vs Quality (bubble size = composite score)",
        height=450,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ── Tab 2: Stock Detail ────────────────────────────────────────────────────────

with tab_detail:
    st.header("Stock Detail View")

    ticker_options = [s["ticker"] for s in ranked]
    selected = st.selectbox("Select a stock", ticker_options, format_func=lambda t: f"{t} — {next(s['name'] for s in ranked if s['ticker'] == t)}")
    stock = next(s for s in ranked if s["ticker"] == selected)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rank", f"#{stock['rank']}")
    col_b.metric("Composite Score", fmt_score(stock["composite"]))
    col_c.metric("Price", fmt_val(stock["price_metrics"].get("current_price"), 2, "$"))
    col_d.metric("Sector", stock["sector"])

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Momentum")
        st.markdown(f"<span class='{score_color(stock['momentum'].get('score'))}'>Score: {fmt_score(stock['momentum'].get('score'))}/100</span>", unsafe_allow_html=True)
        bd = stock["momentum"].get("breakdown", {})
        st.markdown(f"**1M Return:** {fmt_pct(bd.get('1m_return'))}")
        st.markdown(f"**3M Return:** {fmt_pct(bd.get('3m_return'))}")
        st.markdown(f"**6M Return:** {fmt_pct(bd.get('6m_return'))}")
        above50 = bd.get("above_sma50")
        above200 = bd.get("above_sma200")
        st.markdown(f"**Above SMA50:** {'✅' if above50 else '❌' if above50 is not None else '—'}")
        st.markdown(f"**Above SMA200:** {'✅' if above200 else '❌' if above200 is not None else '—'}")
        st.markdown(f"**RSI (14):** {fmt_val(bd.get('rsi'), 1)}")

    with col2:
        st.subheader("Value")
        st.markdown(f"<span class='{score_color(stock['value'].get('score'))}'>Score: {fmt_score(stock['value'].get('score'))}/100</span>", unsafe_allow_html=True)
        bd = stock["value"].get("breakdown", {})
        st.markdown(f"**Trailing P/E:** {fmt_val(bd.get('trailing_pe'), 1)}")
        st.markdown(f"**Forward P/E:** {fmt_val(bd.get('forward_pe'), 1)}")
        st.markdown(f"**P/B Ratio:** {fmt_val(bd.get('pb_ratio'), 2)}")
        st.markdown(f"**EV/EBITDA:** {fmt_val(bd.get('ev_ebitda'), 1)}")

    with col3:
        st.subheader("Quality")
        st.markdown(f"<span class='{score_color(stock['quality'].get('score'))}'>Score: {fmt_score(stock['quality'].get('score'))}/100</span>", unsafe_allow_html=True)
        bd = stock["quality"].get("breakdown", {})
        st.markdown(f"**Revenue Growth:** {fmt_pct(bd.get('revenue_growth_pct'))}")
        st.markdown(f"**Earnings Growth:** {fmt_pct(bd.get('earnings_growth_pct'))}")
        st.markdown(f"**Profit Margin:** {fmt_pct(bd.get('profit_margin_pct'))}")
        st.markdown(f"**ROE:** {fmt_pct(bd.get('roe_pct'))}")
        st.markdown(f"**Debt/Equity:** {fmt_val(bd.get('debt_to_equity'), 1)}")

    st.markdown("---")
    # Radar chart
    categories = ["Momentum", "Value", "Quality"]
    values = [
        stock["momentum"].get("score") or 0,
        stock["value"].get("score") or 0,
        stock["quality"].get("score") or 0,
    ]

    fig_radar = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        line_color="#1976d2",
        fillcolor="rgba(25, 118, 210, 0.2)",
        name=selected,
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"{selected} Factor Radar",
        height=400,
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ── Tab 3: Charts ──────────────────────────────────────────────────────────────

with tab_charts:
    st.header("Price Chart")

    chart_ticker = st.selectbox(
        "Select ticker",
        [s["ticker"] for s in ranked],
        key="chart_ticker",
    )
    chart_stock = next(s for s in ranked if s["ticker"] == chart_ticker)
    hist = raw_data[chart_ticker]["history"]

    if not hist.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name=chart_ticker,
        ))

        if len(hist) >= 50:
            sma50 = hist["Close"].rolling(50).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=sma50, name="SMA 50", line=dict(color="orange", width=1.5)))

        if len(hist) >= 200:
            sma200 = hist["Close"].rolling(200).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=sma200, name="SMA 200", line=dict(color="red", width=1.5)))

        fig.update_layout(
            title=f"{chart_ticker} — 1 Year Price History",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume
        fig_vol = go.Figure(go.Bar(
            x=hist.index,
            y=hist["Volume"],
            marker_color="steelblue",
            name="Volume",
        ))
        fig_vol.update_layout(title="Volume", height=200, margin=dict(t=30))
        st.plotly_chart(fig_vol, use_container_width=True)

    # Sector comparison bar chart
    st.markdown("---")
    st.subheader("Composite Score by Stock")
    bar_df = pd.DataFrame([{
        "ticker": s["ticker"],
        "composite": s["composite"],
        "sector": s["sector"],
    } for s in ranked if s["composite"] is not None])

    fig_bar = px.bar(
        bar_df.sort_values("composite", ascending=True),
        x="composite",
        y="ticker",
        color="sector",
        orientation="h",
        title="All Stocks Ranked by Composite Score",
        height=max(400, len(ranked) * 25),
        labels={"composite": "Composite Score", "ticker": ""},
    )
    fig_bar.update_layout(yaxis=dict(tickfont=dict(size=11)))
    st.plotly_chart(fig_bar, use_container_width=True)


# ── Tab 4: AI Analysis ─────────────────────────────────────────────────────────

with tab_ai:
    st.header("AI Analysis (Claude)")
    st.caption("Uses Claude to explain scores in plain English. Each call uses the API — click only when needed.")

    col_ai1, col_ai2 = st.columns([2, 1])
    with col_ai1:
        ai_ticker = st.selectbox(
            "Select a stock for AI analysis",
            [s["ticker"] for s in ranked],
            key="ai_ticker",
        )
    with col_ai2:
        analyze_btn = st.button("Analyze Stock", type="primary", use_container_width=True)

    if analyze_btn:
        stock = next(s for s in ranked if s["ticker"] == ai_ticker)
        with st.spinner(f"Analyzing {ai_ticker} with Claude..."):
            try:
                explanation = explain_stock(
                    ticker=stock["ticker"],
                    name=stock["name"],
                    sector=stock["sector"],
                    scores=stock,
                    price_metrics=stock["price_metrics"],
                    fund_metrics=stock["fund_metrics"],
                )
                st.markdown(f"### {ai_ticker} — AI Analysis")
                st.info(explanation)
            except Exception as e:
                st.error(f"AI analysis failed: {e}. Check your ANTHROPIC_API_KEY in .env")

    st.markdown("---")
    st.subheader("Watchlist Market Summary")
    summary_btn = st.button("Generate Market Summary", use_container_width=True)

    if summary_btn:
        with st.spinner("Generating summary with Claude..."):
            try:
                summary = generate_market_summary(ranked)
                st.success(summary)
            except Exception as e:
                st.error(f"Summary failed: {e}. Check your ANTHROPIC_API_KEY in .env")

    st.markdown("---")
    st.subheader("Top & Bottom Picks")
    col_top, col_bot = st.columns(2)

    with col_top:
        st.markdown("**Top 5 Ranked**")
        for s in ranked[:5]:
            st.markdown(f"**#{s['rank']} {s['ticker']}** — {fmt_score(s['composite'])} · {s['sector']}")

    with col_bot:
        st.markdown("**Bottom 5 Ranked**")
        for s in ranked[-5:]:
            st.markdown(f"**#{s['rank']} {s['ticker']}** — {fmt_score(s['composite'])} · {s['sector']}")
