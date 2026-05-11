import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

from data_fetcher import fetch_watchlist, get_price_metrics, get_fundamental_metrics, DEFAULT_WATCHLIST
from scoring.ranker import score_momentum, score_value, score_quality, composite_score, rank_stocks
from ai.explainer import explain_stock, generate_market_summary
from storage import (
    PRESET_WATCHLISTS,
    save_snapshot, load_history,
    load_alerts, save_alert, delete_alert, check_alerts,
    load_saved_watchlists, save_watchlist, delete_watchlist,
)
from backtest import run_backtest
from predictions import calculate_predictions, prediction_color, confidence_badge

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
    .alert-box  { background: #1a237e22; border-left: 4px solid #3f51b5; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def signal_label(score):
    if score is None:
        return "—"
    if score >= 65:
        return "🟢 Buy"
    if score >= 40:
        return "🟡 Hold"
    return "🔴 Avoid"


def score_color_class(score):
    if score is None:
        return "score-mid"
    if score >= 65:
        return "score-high"
    if score >= 40:
        return "score-mid"
    return "score-low"


def fmt_score(val):
    return f"{val:.0f}" if val is not None else "N/A"


def fmt_pct(val):
    if val is None:
        return "—"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}%"


def fmt_val(val, decimals=1, prefix="", suffix=""):
    if val is None:
        return "—"
    return f"{prefix}{val:.{decimals}f}{suffix}"


def _color_score(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v >= 65:
        return "background-color: rgba(0,200,83,0.12); color: #00c853; font-weight:600"
    if v >= 40:
        return "background-color: rgba(255,171,0,0.12); color: #ffab00; font-weight:600"
    return "background-color: rgba(255,23,68,0.12); color: #ff1744; font-weight:600"


# ── Session state init ─────────────────────────────────────────────────────────

if "watchlist_text" not in st.session_state:
    st.session_state.watchlist_text = "\n".join(DEFAULT_WATCHLIST)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Stock Dashboard")
    st.markdown("---")

    # Watchlist presets
    st.subheader("Watchlist Presets")
    all_presets = {**PRESET_WATCHLISTS, **load_saved_watchlists()}
    preset_names = list(all_presets.keys())
    chosen_preset = st.selectbox("Load a preset", ["— pick one —"] + preset_names)
    if chosen_preset != "— pick one —":
        if st.button("Load Preset", use_container_width=True):
            st.session_state.watchlist_text = "\n".join(all_presets[chosen_preset])
            st.rerun()

    st.markdown("---")

    # Save custom watchlist
    with st.expander("💾 Save Current List as Preset"):
        new_name = st.text_input("Preset name")
        if st.button("Save", use_container_width=True) and new_name.strip():
            tickers_to_save = [
                t.strip().upper()
                for t in st.session_state.watchlist_text.replace(",", "\n").split("\n")
                if t.strip()
            ]
            save_watchlist(new_name.strip(), tickers_to_save)
            st.success(f"Saved '{new_name}'")

        saved = load_saved_watchlists()
        if saved:
            del_name = st.selectbox("Delete a saved preset", ["—"] + list(saved.keys()))
            if st.button("Delete", use_container_width=True) and del_name != "—":
                delete_watchlist(del_name)
                st.rerun()

    st.markdown("---")

    st.subheader("Watchlist")
    watchlist_text = st.text_area(
        "Tickers (one per line)",
        value=st.session_state.watchlist_text,
        height=220,
        key="watchlist_text",
    )

    raw = st.session_state.watchlist_text.replace(",", "\n").split("\n")
    tickers = [t.strip().upper() for t in raw if t.strip()]

    st.markdown("---")
    st.subheader("Factor Weights")
    w_momentum = st.slider("Momentum", 0, 100, 35)
    w_value = st.slider("Value", 0, 100, 30)
    w_quality = st.slider("Quality", 0, 100, 35)
    total_w = w_momentum + w_value + w_quality
    weights = (w_momentum / total_w, w_value / total_w, w_quality / total_w) if total_w > 0 else (0.35, 0.30, 0.35)

    st.markdown("---")
    run = st.button("🔄 Refresh Data", use_container_width=True, type="primary")
    st.caption("Data via Yahoo Finance · AI via Claude · Not financial advice")


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def load_and_score(tickers_tuple, weights_tuple):
    raw_data = fetch_watchlist(list(tickers_tuple))
    results = []
    for ticker, data in raw_data.items():
        price_m = get_price_metrics(data["history"])
        fund_m = get_fundamental_metrics(data["info"])
        mom = score_momentum(price_m)
        val = score_value(fund_m)
        qual = score_quality(fund_m)
        comp = composite_score(mom, val, qual, weights_tuple)
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

# Auto-save snapshot when data loads
try:
    save_snapshot(ranked)
except Exception:
    pass

# Check price alerts
alerts = load_alerts()
triggered = check_alerts(ranked, alerts)
if triggered:
    for alert in triggered:
        cond = "above" if alert["condition"] == "above" else "below"
        st.warning(f"🔔 **{alert['ticker']}** is {cond} your target — Current: ${alert['price']:.2f} | Target: ${alert['target']:.2f}")


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_overview, tab_detail, tab_charts, tab_predict, tab_history, tab_backtest, tab_ai = st.tabs([
    "📊 Rankings", "🔍 Stock Detail", "📉 Charts", "🔮 Predictions", "📅 History", "🧪 Backtest", "🤖 AI Analysis"
])


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
            "Name": s["name"][:28],
            "Sector": s["sector"],
            "Signal": signal_label(s["composite"]),
            "Price": s["price_metrics"].get("current_price"),
            "Composite": s["composite"],
            "Momentum": s["momentum"].get("score"),
            "Value": s["value"].get("score"),
            "Quality": s["quality"].get("score"),
            "1M %": s["price_metrics"].get("return_1m"),
            "3M %": s["price_metrics"].get("return_3m"),
            "6M %": s["price_metrics"].get("return_6m"),
            "RSI": s["price_metrics"].get("rsi_14"),
            "Fwd P/E": s["fund_metrics"].get("forward_pe"),
        })

    df = pd.DataFrame(rows)

    _styler = df.style
    _cell_style = getattr(_styler, "map", getattr(_styler, "applymap", None))
    styled = _cell_style(
        _color_score, subset=["Composite", "Momentum", "Value", "Quality"]
    ).format({
        "Price": lambda v: f"${v:.2f}" if v else "—",
        "Composite": lambda v: f"{v:.0f}" if v else "—",
        "Momentum": lambda v: f"{v:.0f}" if v else "—",
        "Value": lambda v: f"{v:.0f}" if v else "—",
        "Quality": lambda v: f"{v:.0f}" if v else "—",
        "1M %": lambda v: (("+" if v >= 0 else "") + f"{v:.1f}%") if v else "—",
        "3M %": lambda v: (("+" if v >= 0 else "") + f"{v:.1f}%") if v else "—",
        "6M %": lambda v: (("+" if v >= 0 else "") + f"{v:.1f}%") if v else "—",
        "RSI": lambda v: f"{v:.1f}" if v else "—",
        "Fwd P/E": lambda v: f"{v:.1f}" if v else "—",
    })

    st.dataframe(styled, use_container_width=True, hide_index=True, height=580)

    st.markdown("---")

    # Price Alerts
    st.subheader("🔔 Price Alerts")
    col_al1, col_al2, col_al3, col_al4 = st.columns([2, 2, 1, 1])
    with col_al1:
        alert_ticker = st.selectbox("Stock", [s["ticker"] for s in ranked], key="alert_ticker")
    with col_al2:
        alert_price = st.number_input("Target Price ($)", min_value=0.01, value=100.0, step=1.0)
    with col_al3:
        alert_cond = st.selectbox("Condition", ["above", "below"])
    with col_al4:
        st.write("")
        st.write("")
        if st.button("Set Alert", use_container_width=True):
            save_alert(alert_ticker, alert_price, alert_cond)
            st.success(f"Alert set for {alert_ticker}")

    current_alerts = load_alerts()
    if current_alerts:
        st.markdown("**Active Alerts:**")
        price_map = {s["ticker"]: s["price_metrics"].get("current_price") for s in ranked}
        for ticker, al in current_alerts.items():
            current = price_map.get(ticker)
            current_str = f"${current:.2f}" if current else "N/A"
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"**{ticker}** — Alert when {al['condition']} **${al['target']:.2f}** · Current: {current_str}")
            with col_b:
                if st.button("Remove", key=f"del_{ticker}"):
                    delete_alert(ticker)
                    st.rerun()

    st.markdown("---")

    # Scatter
    st.subheader("Score Distribution")
    fig_scatter = px.scatter(
        pd.DataFrame([{
            "ticker": s["ticker"],
            "momentum": s["momentum"].get("score"),
            "value": s["value"].get("score"),
            "quality": s["quality"].get("score"),
            "composite": s["composite"],
            "sector": s["sector"],
            "signal": signal_label(s["composite"]),
        } for s in ranked if s["composite"] is not None]),
        x="momentum", y="quality",
        size="composite", color="sector",
        hover_name="ticker",
        hover_data={"signal": True},
        labels={"momentum": "Momentum Score", "quality": "Quality Score"},
        title="Momentum vs Quality (bubble size = composite score)",
        height=420,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ── Tab 2: Stock Detail ────────────────────────────────────────────────────────

with tab_detail:
    st.header("Stock Detail View")

    ticker_options = [s["ticker"] for s in ranked]
    selected = st.selectbox(
        "Select a stock", ticker_options,
        format_func=lambda t: f"{t} — {next(s['name'] for s in ranked if s['ticker'] == t)}"
    )
    stock = next(s for s in ranked if s["ticker"] == selected)

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Rank", f"#{stock['rank']}")
    col_b.metric("Signal", signal_label(stock["composite"]))
    col_c.metric("Composite", fmt_score(stock["composite"]))
    col_d.metric("Price", fmt_val(stock["price_metrics"].get("current_price"), 2, "$"))
    col_e.metric("Sector", stock["sector"])

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Momentum")
        st.markdown(f"<span class='{score_color_class(stock['momentum'].get('score'))}'>Score: {fmt_score(stock['momentum'].get('score'))}/100</span>", unsafe_allow_html=True)
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
        st.markdown(f"<span class='{score_color_class(stock['value'].get('score'))}'>Score: {fmt_score(stock['value'].get('score'))}/100</span>", unsafe_allow_html=True)
        bd = stock["value"].get("breakdown", {})
        st.markdown(f"**Trailing P/E:** {fmt_val(bd.get('trailing_pe'), 1)}")
        st.markdown(f"**Forward P/E:** {fmt_val(bd.get('forward_pe'), 1)}")
        st.markdown(f"**P/B Ratio:** {fmt_val(bd.get('pb_ratio'), 2)}")
        st.markdown(f"**EV/EBITDA:** {fmt_val(bd.get('ev_ebitda'), 1)}")

    with col3:
        st.subheader("Quality")
        st.markdown(f"<span class='{score_color_class(stock['quality'].get('score'))}'>Score: {fmt_score(stock['quality'].get('score'))}/100</span>", unsafe_allow_html=True)
        bd = stock["quality"].get("breakdown", {})
        st.markdown(f"**Revenue Growth:** {fmt_pct(bd.get('revenue_growth_pct'))}")
        st.markdown(f"**Earnings Growth:** {fmt_pct(bd.get('earnings_growth_pct'))}")
        st.markdown(f"**Profit Margin:** {fmt_pct(bd.get('profit_margin_pct'))}")
        st.markdown(f"**ROE:** {fmt_pct(bd.get('roe_pct'))}")
        st.markdown(f"**Debt/Equity:** {fmt_val(bd.get('debt_to_equity'), 1)}")

    st.markdown("---")
    categories = ["Momentum", "Value", "Quality"]
    values = [stock["momentum"].get("score") or 0, stock["value"].get("score") or 0, stock["quality"].get("score") or 0]

    fig_radar = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        line_color="#1976d2",
        fillcolor="rgba(25,118,210,0.2)",
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

    chart_ticker = st.selectbox("Select ticker", [s["ticker"] for s in ranked], key="chart_ticker")
    hist = raw_data[chart_ticker]["history"]

    if not hist.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"], name=chart_ticker,
        ))
        if len(hist) >= 50:
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(50).mean(), name="SMA 50", line=dict(color="orange", width=1.5)))
        if len(hist) >= 200:
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(200).mean(), name="SMA 200", line=dict(color="red", width=1.5)))
        fig.update_layout(title=f"{chart_ticker} — 1 Year", xaxis_title="Date", yaxis_title="Price (USD)", height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        fig_vol = go.Figure(go.Bar(x=hist.index, y=hist["Volume"], marker_color="steelblue", name="Volume"))
        fig_vol.update_layout(title="Volume", height=200, margin=dict(t=30))
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")
    st.subheader("Composite Score by Stock")
    bar_df = pd.DataFrame([{"ticker": s["ticker"], "composite": s["composite"], "sector": s["sector"], "signal": signal_label(s["composite"])} for s in ranked if s["composite"] is not None])
    fig_bar = px.bar(
        bar_df.sort_values("composite", ascending=True),
        x="composite", y="ticker", color="sector", orientation="h",
        title="All Stocks Ranked by Composite Score",
        height=max(400, len(ranked) * 25),
        labels={"composite": "Composite Score", "ticker": ""},
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ── Tab 4: Predictions ────────────────────────────────────────────────────────

with tab_predict:
    st.header("Predicted Returns")
    st.markdown("""
    | Horizon | Source | Reliability |
    |---------|--------|-------------|
    | **1-Year** | Wall Street analyst consensus price targets | Higher — real forecasts from professional analysts |
    | **3-Year** | Earnings growth model (projected EPS × PE) | Medium — directional, assumes growth continues |
    | **5-Year** | Same model with heavier growth dampening | Lower — treat as a rough range, not a target |
    """)
    st.caption("⚠️ All projections are estimates. Markets are unpredictable. Not financial advice.")

    st.markdown("---")

    # Full watchlist predictions table
    st.subheader("Watchlist Predictions Overview")

    pred_rows = []
    for s in ranked:
        preds = calculate_predictions(
            s["price_metrics"].get("current_price"),
            s["fund_metrics"],
        )
        row = {
            "Ticker": s["ticker"],
            "Name": s["name"][:25],
            "Price": s["price_metrics"].get("current_price"),
            "Signal": signal_label(s["composite"]),
            "Analyst Rec": s["fund_metrics"].get("recommendation", "—"),
        }
        for key, label in [("1yr", "1Y Target"), ("3yr", "3Y Return"), ("5yr", "5Y Return")]:
            if key in preds:
                row[label] = preds[key]["return_pct"]
                if key == "1yr":
                    row["1Y Price"] = preds[key]["target_price"]
                    row["# Analysts"] = preds[key]["num_analysts"]
            else:
                row[label] = None
                if key == "1yr":
                    row["1Y Price"] = None
                    row["# Analysts"] = None
        pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)

    def _color_return(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if v >= 20:
            return "background-color: rgba(0,200,83,0.15); color: #00c853; font-weight:600"
        if v >= 5:
            return "background-color: rgba(100,221,23,0.12); color: #76d275; font-weight:600"
        if v >= -5:
            return "background-color: rgba(255,171,0,0.12); color: #ffab00; font-weight:600"
        if v >= -20:
            return "background-color: rgba(255,109,0,0.12); color: #ff6d00; font-weight:600"
        return "background-color: rgba(255,23,68,0.12); color: #ff1744; font-weight:600"

    _pred_styler = pred_df.style
    _pred_cell_style = getattr(_pred_styler, "map", getattr(_pred_styler, "applymap", None))
    styled_pred = _pred_cell_style(
        _color_return, subset=["1Y Target", "3Y Return", "5Y Return"]
    ).format({
        "Price": lambda v: f"${v:.2f}" if v else "—",
        "1Y Price": lambda v: f"${v:.2f}" if v else "—",
        "1Y Target": lambda v: f"{v:+.1f}%" if v is not None else "—",
        "3Y Return": lambda v: f"{v:+.1f}%" if v is not None else "—",
        "5Y Return": lambda v: f"{v:+.1f}%" if v is not None else "—",
        "# Analysts": lambda v: str(int(v)) if v else "—",
        "Analyst Rec": lambda v: v.replace("_", " ").title() if isinstance(v, str) and v != "—" else (v or "—"),
    })

    st.dataframe(styled_pred, use_container_width=True, hide_index=True, height=500)

    st.markdown("---")

    # Deep-dive on a single stock
    st.subheader("Stock Deep-Dive")
    pred_ticker = st.selectbox("Select stock", [s["ticker"] for s in ranked], key="pred_ticker")
    pred_stock = next(s for s in ranked if s["ticker"] == pred_ticker)
    pred_price = pred_stock["price_metrics"].get("current_price")
    preds = calculate_predictions(pred_price, pred_stock["fund_metrics"])

    if not preds:
        st.warning(f"No prediction data available for {pred_ticker}. Yahoo Finance may not have analyst coverage for this stock.")
    else:
        cols = st.columns(len(preds))
        for col, (key, p) in zip(cols, preds.items()):
            ret = p["return_pct"]
            color = prediction_color(ret)
            badge = confidence_badge(p["confidence"])
            sign = "+" if ret >= 0 else ""
            with col:
                st.markdown(f"### {p['label']}")
                st.markdown(f"<h2 style='color:{color}; margin:0'>{sign}{ret:.1f}%</h2>", unsafe_allow_html=True)
                st.markdown(f"**Target Price:** ${p['target_price']:.2f}")
                st.markdown(f"**Annualized:** {p['annualized_return']:+.1f}%/yr")
                st.markdown(f"**Confidence:** {badge} {p['confidence']}")
                st.markdown(f"**Method:** {p['method']}")
                st.caption(p["note"])
                if p.get("high_target") and p.get("low_target"):
                    st.markdown(f"**Range:** ${p['low_target']:.2f} — ${p['high_target']:.2f}")
                if p.get("num_analysts"):
                    st.markdown(f"**Analysts:** {p['num_analysts']}")
                if p.get("recommendation"):
                    st.markdown(f"**Consensus:** {p['recommendation']}")

        st.markdown("---")

        # Visual: price target range chart
        if "1yr" in preds:
            p1 = preds["1yr"]
            fig_target = go.Figure()

            # Current price line
            fig_target.add_hline(y=pred_price, line_dash="solid", line_color="white", line_width=2, annotation_text=f"Current ${pred_price:.2f}", annotation_position="left")

            # 1yr range bar
            if p1.get("low_target") and p1.get("high_target"):
                fig_target.add_trace(go.Bar(
                    x=["1-Year Analyst Range"],
                    y=[p1["high_target"] - p1["low_target"]],
                    base=[p1["low_target"]],
                    marker_color="rgba(25,118,210,0.4)",
                    name="Analyst Range",
                    width=0.3,
                ))

            # Mean targets as markers
            for key, label, color in [("1yr", "1Y Mean", "#1976d2"), ("3yr", "3Y Model", "#00c853"), ("5yr", "5Y Model", "#ffab00")]:
                if key in preds:
                    fig_target.add_trace(go.Scatter(
                        x=[label],
                        y=[preds[key]["target_price"]],
                        mode="markers+text",
                        marker=dict(size=16, color=color, symbol="diamond"),
                        text=[f"${preds[key]['target_price']:.0f}"],
                        textposition="top center",
                        name=label,
                    ))

            fig_target.update_layout(
                title=f"{pred_ticker} — Price Targets",
                yaxis_title="Price (USD)",
                height=380,
                showlegend=True,
            )
            st.plotly_chart(fig_target, use_container_width=True)


# ── Tab 5: History ─────────────────────────────────────────────────────────────

with tab_history:
    st.header("Score History")
    st.caption("Scores are saved each time you load the dashboard. History builds up over time.")

    history = load_history()

    if len(history) < 2:
        st.info("Not enough history yet. Come back after a few sessions and you'll see trends here.")
    else:
        all_tickers_hist = sorted(set(
            ticker for snap in history for ticker in snap.get("scores", {}).keys()
        ))

        hist_tickers = st.multiselect(
            "Select stocks to compare",
            all_tickers_hist,
            default=all_tickers_hist[:5] if len(all_tickers_hist) >= 5 else all_tickers_hist,
        )

        score_type = st.radio("Score type", ["Composite", "Momentum", "Value", "Quality"], horizontal=True)
        key_map = {"Composite": "composite", "Momentum": "momentum", "Value": "value", "Quality": "quality"}
        score_key = key_map[score_type]

        plot_data = []
        for snap in history:
            ts = snap.get("timestamp", "")[:16]
            for ticker in hist_tickers:
                val = snap.get("scores", {}).get(ticker, {}).get(score_key)
                if val is not None:
                    plot_data.append({"date": ts, "ticker": ticker, "score": val})

        if plot_data:
            hist_df = pd.DataFrame(plot_data)
            fig_hist = px.line(
                hist_df, x="date", y="score", color="ticker",
                title=f"{score_type} Score Over Time",
                labels={"score": f"{score_type} Score", "date": ""},
                height=420,
            )
            fig_hist.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Latest vs Previous Snapshot")
            if len(history) >= 2:
                latest = history[-1]["scores"]
                prev = history[-2]["scores"]
                delta_rows = []
                for ticker in all_tickers_hist:
                    curr_val = latest.get(ticker, {}).get(score_key)
                    prev_val = prev.get(ticker, {}).get(score_key)
                    if curr_val is not None and prev_val is not None:
                        delta = curr_val - prev_val
                        delta_rows.append({
                            "Ticker": ticker,
                            f"Current {score_type}": round(curr_val, 1),
                            f"Previous {score_type}": round(prev_val, 1),
                            "Change": round(delta, 1),
                            "Direction": "🟢 Up" if delta > 0.5 else ("🔴 Down" if delta < -0.5 else "➡️ Flat"),
                        })
                if delta_rows:
                    delta_df = pd.DataFrame(delta_rows).sort_values("Change", ascending=False)
                    st.dataframe(delta_df, use_container_width=True, hide_index=True)
        else:
            st.info("Select at least one stock above.")


# ── Tab 5: Backtest ────────────────────────────────────────────────────────────

with tab_backtest:
    st.header("Momentum Backtest")
    st.markdown("""
    Tests whether picking the **highest momentum stocks** from your watchlist each month would have beaten the S&P 500 (SPY).
    At each month-end, it ranks all stocks by their recent price momentum, holds the top N equal-weighted for one month, then repeats.
    """)
    st.caption("⚠️ This is a price-momentum-only backtest and doesn't account for trading costs, taxes, or slippage. Past performance does not guarantee future results.")

    col_bt1, col_bt2, col_bt3 = st.columns(3)
    with col_bt1:
        bt_top_n = st.slider("Top N stocks to hold", 1, 10, 5)
    with col_bt2:
        bt_years = st.slider("Years of history", 1, 5, 2)
    with col_bt3:
        st.write("")
        st.write("")
        run_bt = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    if run_bt:
        with st.spinner(f"Running backtest over {bt_years} years..."):
            result = run_backtest(tickers, top_n=bt_top_n, years=bt_years)

        if "error" in result:
            st.error(result["error"])
        else:
            summary = result["summary"]

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Portfolio Return", f"{summary['portfolio_total_return']:+.1f}%", delta=f"{summary['outperformance']:+.1f}% vs SPY")
            m2.metric("SPY Return", f"{summary['spy_total_return']:+.1f}%")
            m3.metric("Portfolio Sharpe", f"{summary['portfolio_sharpe']:.2f}", delta=f"{summary['portfolio_sharpe'] - summary['spy_sharpe']:+.2f} vs SPY")
            m4.metric("Max Drawdown", f"{summary['portfolio_max_drawdown']:.1f}%", delta=f"{summary['portfolio_max_drawdown'] - summary['spy_max_drawdown']:.1f}% vs SPY", delta_color="inverse")
            m5.metric("Win Rate vs SPY", f"{summary['win_rate_vs_spy']:.0f}%", f"{summary['months_tested']} months tested")

            st.markdown("---")

            # Cumulative return chart
            bt_df = pd.DataFrame({
                "Date": result["dates"],
                f"Top {bt_top_n} Momentum": [(v - 1) * 100 for v in result["portfolio_cumulative"]],
                "SPY": [(v - 1) * 100 for v in result["spy_cumulative"]],
            })

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df[f"Top {bt_top_n} Momentum"], name=f"Top {bt_top_n} Strategy", line=dict(color="#00c853", width=2)))
            fig_bt.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["SPY"], name="SPY Benchmark", line=dict(color="#1976d2", width=2, dash="dot")))
            fig_bt.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_bt.update_layout(
                title=f"Cumulative Return: Top {bt_top_n} Momentum vs SPY",
                yaxis_title="Return (%)",
                xaxis_title="",
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            # Holdings log
            st.subheader("Monthly Holdings")
            holdings_df = pd.DataFrame({
                "Month": result["dates"],
                "Holdings": [", ".join(h) for h in result["holdings_log"]],
                "Portfolio Return": [f"{r*100:+.1f}%" for r in result["portfolio_monthly_returns"]],
                "SPY Return": [f"{r*100:+.1f}%" for r in result["spy_monthly_returns"]],
            })
            st.dataframe(holdings_df.iloc[::-1], use_container_width=True, hide_index=True, height=300)


# ── Tab 6: AI Analysis ─────────────────────────────────────────────────────────

with tab_ai:
    st.header("AI Analysis (Claude)")
    st.caption("Uses Claude to explain scores in plain English. Each call uses the API — click only when needed.")

    col_ai1, col_ai2 = st.columns([2, 1])
    with col_ai1:
        ai_ticker = st.selectbox("Select a stock for AI analysis", [s["ticker"] for s in ranked], key="ai_ticker")
    with col_ai2:
        st.write("")
        analyze_btn = st.button("Analyze Stock", type="primary", use_container_width=True)

    if analyze_btn:
        stock = next(s for s in ranked if s["ticker"] == ai_ticker)
        with st.spinner(f"Analyzing {ai_ticker} with Claude..."):
            try:
                explanation = explain_stock(
                    ticker=stock["ticker"], name=stock["name"], sector=stock["sector"],
                    scores=stock, price_metrics=stock["price_metrics"], fund_metrics=stock["fund_metrics"],
                )
                st.markdown(f"### {ai_ticker} — {signal_label(stock['composite'])}")
                st.info(explanation)
            except Exception as e:
                st.error(f"AI analysis failed: {e}. Check your ANTHROPIC_API_KEY in .env")

    st.markdown("---")
    st.subheader("Watchlist Market Summary")
    if st.button("Generate Market Summary", use_container_width=True):
        with st.spinner("Generating summary with Claude..."):
            try:
                summary = generate_market_summary(ranked)
                st.success(summary)
            except Exception as e:
                st.error(f"Summary failed: {e}. Check your ANTHROPIC_API_KEY in .env")

    st.markdown("---")
    col_top, col_bot = st.columns(2)
    with col_top:
        st.subheader("Top 5 Ranked")
        for s in ranked[:5]:
            st.markdown(f"**#{s['rank']} {s['ticker']}** {signal_label(s['composite'])} — Score: {fmt_score(s['composite'])} · {s['sector']}")
    with col_bot:
        st.subheader("Bottom 5 Ranked")
        for s in ranked[-5:]:
            st.markdown(f"**#{s['rank']} {s['ticker']}** {signal_label(s['composite'])} — Score: {fmt_score(s['composite'])} · {s['sector']}")
