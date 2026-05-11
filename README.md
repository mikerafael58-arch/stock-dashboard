# Stock Ranking Dashboard

A multi-factor stock ranking dashboard powered by Yahoo Finance data and Claude AI.

## Setup

### 1. Install dependencies
```bash
cd stock-dashboard
pip install -r requirements.txt
```

### 2. Add your Anthropic API key
```bash
cp .env.example .env
# Edit .env and add your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

Get your API key at: https://console.anthropic.com/

### 3. Run the dashboard
```bash
streamlit run app.py
```

The dashboard opens automatically in your browser at http://localhost:8501

## How it works

Stocks are scored across three factors (0–100 each):

| Factor | What it measures |
|--------|-----------------|
| **Momentum** | Price trend — returns over 1/3/6 months, position vs SMA50/200, RSI |
| **Value** | How cheap the stock is — P/E, Forward P/E, P/B, EV/EBITDA |
| **Quality** | Business health — revenue/earnings growth, profit margin, ROE, debt |

A **composite score** is a weighted average of all three. You can adjust the weights in the sidebar.

## Features

- **Rankings tab** — full sortable table with scores, returns, and fundamentals
- **Stock Detail tab** — deep-dive with radar chart for any stock
- **Charts tab** — candlestick chart with SMA overlays + volume
- **AI Analysis tab** — Claude explains why a stock ranks where it does in plain English

## Notes

- Data refreshes every 15 minutes (cached). Hit "Refresh Data" to force a reload.
- AI analysis uses your Anthropic API key and costs a small amount per call (~$0.001).
- This is for informational purposes only. Not financial advice.
