# MarketPulse AI

An explainable market research dashboard for stocks, crypto, and spot gold (XAU/USD). It combines online OHLC data, multi-timeframe structure, price-action zones, candle patterns, a rule-based confluence score, visible trade-plan levels, and position-sizing estimates.

> **Research software, not financial advice.** Signals are rule-based candidates, not forecasts or guaranteed profitable trades. No order execution, trained probability model, or verified strategy win rate is included.

## What it includes

- Online candles: Yahoo Finance for stocks, a Yahoo gold-futures proxy (`GC=F`), Binance through CCXT for crypto, and an optional Twelve Data spot-metal adapter (`XAU/USD`).
- Intraday charts at 30m, 15m, and 5m, plus 1h, 4h, daily, and weekly context when the provider supplies enough data.
- Confirmed swings, close-based BOS/CHoCH, FVG, order block, supply/demand, engulfing candles, and long-wick rejection patterns.
- A confluence score with its component evidence. The model/sentiment portion from the planning document is excluded until a model can be trained and evaluated honestly.
- A trade-plan candidate with entry area, invalidation/stop, targets, and risk-reward levels overlaid on the chart.
- A waiting checklist that names missing setup conditions and estimates a broad time window from recent directional candle movement when a confirmed swing level is available. It withholds timing when momentum is weak or points away from the level.
- Confirmed BOS/CHoCH levels and rising/falling swing-high and swing-low trendlines, each with a chart-layer toggle.
- Layer controls, chart freshness information, risk sizing, and a no-setup/wait state.
- Still-forming intraday bars are removed before analysis. Demo CSV data is not substituted when online data fails.
- The chart uses equal-width candle categories, so closed-market periods such as weekends do not create giant blank spaces. It never fills missing candles with invented prices.
- The main chart uses the full dashboard width and renders every candle returned for the selected timeframe. Use the Plotly zoom and pan controls to inspect dense history; the amount of history depends on the selected feed and start date.
- An optional 15/30/60-second polling refresh updates the displayed provider candle when the feed returns one. Signal calculations use completed bars, while the forming candle is display-only.

## Setup on Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
streamlit run app.py
```

If activation is blocked, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the current PowerShell window, then activate the environment again.

## Online feeds

### Yahoo Finance

Yahoo Finance is the default no-key option for equities. Its `GC=F` symbol is a COMEX gold futures contract proxy, not spot XAU/USD. The app labels it accordingly. Choose Twelve Data for the spot-metal pair `XAU/USD`. Yahoo intraday history is limited to roughly the latest 60 days. Availability and quote delay depend on symbol and market; the app reports the latest completed candle and does not promise a real-time feed. `yfinance` accesses Yahoo's public API, is not affiliated with Yahoo, and its documentation says the tool is intended for research and educational use. Check Yahoo's data terms before using it beyond personal research.

### Twelve Data

For a Twelve Data feed, set an API key in the environment and restart Streamlit:

```powershell
$env:TWELVEDATA_API_KEY = "your-key"
streamlit run app.py
```

The dashboard then offers Twelve Data for non-crypto instruments. Use `XAU/USD` for gold. Availability, real-time entitlement, intraday history depth, call limits, and commercial rights depend on the account and plan; the provider's supported intervals include 5m, 15m, and 30m. The app surfaces API errors and never fabricates missing candles.

### Crypto

Crypto candles come from Binance through CCXT. Example symbols are `BTC/USDT` and `ETH/USDT`. Exchange availability and history depth can vary.

## Analysis and interpretation

1. Weekly and daily structure set the broad bias when those frames are available.
2. The 4h/1h context locates price relative to unmitigated SMC zones.
3. The selected chart timeframe checks recent confirmed structure and price-action evidence.
4. The score combines measurable bias, zone, and entry-confirmation scores. The old fixed 15-point model placeholder has been removed; the displayed result is not a statistical probability.
5. A plan is shown only when a nearby zone, score, direction, and entry status pass the configured rules. Otherwise, the dashboard says to wait.

SMC terms are not standardized. Zone and candle rules are visible in the code and should be treated as hypotheses. Use the chart-layer controls to reduce clutter or inspect individual evidence in the expandable tables.

## Tests

```powershell
pytest -q
```

Tests cover data normalization, timeframe aggregation, the base SMC rules, and the explanatory candle-pattern detector. Backtest with fees, spread, slippage, market sessions, and walk-forward/out-of-sample data before treating any rule as useful. Historical strategy performance and ML output are not implemented yet.

## Data and product limitations

- Providers may delay, omit, revise, or rate-limit data. Check the latest candle timestamp before relying on a chart.
- Auto-refresh polls the selected provider; it is not a direct exchange trade stream. Yahoo may delay quotes, while API-key feeds and public exchange endpoints have plan/rate limits.
- Yahoo's XAUUSD ticker and Twelve Data's XAU/USD feed may represent different quote conventions. Confirm the selected contract, quote currency, and provider symbol.
- FX/spot-metal volume is unavailable or non-centralized on some feeds; zero volume is not treated as traded volume by the app.
- Position sizing assumes account and quote currencies align and does not model contract multipliers, margin, conversion, fees, or broker rules.
- No backtested accuracy, alerts, journal, scanner, order routing, or trained ML model is claimed.
- The momentum ETA is not a statistical average or a promise; setup waits can take much longer or never occur.
- Provider terms, retention, and data-redistribution rights must be checked before deployment.
