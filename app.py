"""MarketPulse AI: explainable online market analysis dashboard."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pandas as pd
import streamlit as st

from marketpulse.chart.builder import build_chart_state
from marketpulse.chart.renderer import create_plotly_figure
from marketpulse.data.normalize import normalize_ohlcv
from marketpulse.data.providers.yfinance_provider import YahooFinanceProvider
from marketpulse.data.timeframes import build_timeframes
from marketpulse.risk.calculator import calculate_position_size
from marketpulse.smc.confluence import calculate_confluence
from marketpulse.smc.patterns import detect_candlestick_patterns
from marketpulse.smc.structure import confirmed_swings, detect_structure_breaks
from marketpulse.smc.trade_plan import generate_trade_plan
from marketpulse.smc.waiting import assess_waiting_state
from marketpulse.smc.zones import (
    detect_fair_value_gaps,
    detect_order_blocks,
    enrich_supply_demand_zones,
    get_active_zones,
)

st.set_page_config(page_title="MarketPulse AI", page_icon="📈", layout="wide")
st.markdown("""
<style>
.stApp { background: #0b1220; }
.block-container { padding-top: 1.25rem; max-width: 1600px; }
[data-testid="stMetric"] { background: #111c2e; border: 1px solid #22324a; border-radius: 12px; padding: 12px; }
h1, h2, h3 { letter-spacing: -0.02em; }
</style>
""", unsafe_allow_html=True)


def _secret(name: str) -> str | None:
    return os.getenv(name)


def _strip_open_candles(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Exclude still-forming intraday candles from analysis and chart signals."""
    duration = {
        "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h",
        "4h": "4h", "1d": "1D", "1w": "7D",
    }
    now = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=3)
    cleaned = {}
    for timeframe, frame in frames.items():
        if timeframe in duration and not frame.empty:
            closed = frame.index + pd.Timedelta(duration[timeframe]) <= now
            frame = frame.loc[closed]
        if not frame.empty:
            cleaned[timeframe] = frame
    return cleaned


@st.cache_data(ttl=10, show_spinner="Loading online market data...")
def load_market_data(
    asset_class: str, symbol: str, start_date: str, provider_name: str
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], str]:
    if asset_class == "crypto":
        from marketpulse.data.providers.ccxt_provider import CCXTCryptoProvider
        raw = CCXTCryptoProvider(exchange_id="binance").fetch(symbol, start_date, None)
        provider_label = "Binance via CCXT"
    elif provider_name.startswith("Twelve Data"):
        from marketpulse.data.providers.twelvedata_provider import TwelveDataProvider
        raw = TwelveDataProvider(_secret("TWELVEDATA_API_KEY")).fetch(symbol, start_date, None)
        provider_label = "Twelve Data (API key)"
    else:
        raw = YahooFinanceProvider().fetch(symbol, start_date, None)
        provider_label = "Yahoo Finance (may be delayed)"

    normalized = {tf: normalize_ohlcv(df) for tf, df in raw.items() if df is not None and not df.empty}
    frames = build_timeframes(normalized, asset_class=asset_class)
    return frames, _strip_open_candles(frames), provider_label


with st.sidebar:
    st.title("MarketPulse AI")
    st.caption("Online data · Explainable price action")
    instrument = st.selectbox("Market", ["Stocks", "Crypto", "Gold · XAU/USD"])
    asset_class = {"Stocks": "stock", "Crypto": "crypto", "Gold · XAU/USD": "gold"}[instrument]
    api_key = _secret("TWELVEDATA_API_KEY")
    if asset_class == "crypto":
        provider_name = "Binance"
        symbol = st.text_input("Symbol", "BTC/USDT", help="Examples: BTC/USDT, ETH/USDT")
        st.caption("Crypto candles are fetched online from Binance through CCXT.")
    elif asset_class == "gold":
        gold_feeds = ["Yahoo gold futures proxy (GC=F)"]
        if api_key:
            gold_feeds.append("Twelve Data spot metal (XAU/USD)")
        provider_name = st.selectbox("Gold data feed", gold_feeds, index=len(gold_feeds) - 1)
        if provider_name.startswith("Twelve Data"):
            symbol = "XAU/USD"
            st.caption("Spot gold data. Plan availability and delay depend on your Twelve Data account.")
        else:
            symbol = "GC=F"
            st.caption("Yahoo futures proxy, not spot XAU/USD. Add TWELVEDATA_API_KEY to select the spot-metal feed.")
    else:
        providers = ["Yahoo Finance"] + (["Twelve Data"] if api_key else [])
        provider_name = st.selectbox("Online data feed", providers)
        symbol = st.text_input("Symbol", "NVDA", help="Examples: NVDA, AAPL, RELIANCE.NS")
        if not api_key:
            st.caption("Add TWELVEDATA_API_KEY to use Twelve Data. Yahoo intraday history is limited and can be delayed.")

    start_default = (datetime.now(UTC) - timedelta(days=55)).date().isoformat()
    start_date = st.text_input("History starts", start_default, help="Provider availability limits the usable history for intraday charts.")
    mode = st.radio("Trading mode", ["Swing", "Intraday"], horizontal=True)
    timeframe_choices = ["1w", "1d", "4h", "1h"] if mode == "Swing" else ["30m", "15m", "5m"]
    chart_tf = st.selectbox("Chart timeframe", timeframe_choices, index=0)
    st.caption("The chart shows all candles returned for this timeframe. Available history depends on the selected feed.")
    st.subheader("Chart layers")
    show_fvg = st.checkbox("Fair value gaps", True)
    show_ob = st.checkbox("Order blocks", True)
    show_sd = st.checkbox("Supply / demand", True)
    show_patterns = st.checkbox("Candle-pattern markers", True)
    show_structure = st.checkbox("BOS / CHoCH level lines", True)
    show_trendlines = st.checkbox("Swing trendlines", True)
    st.subheader("Analysis")
    min_confluence = st.slider("Minimum confluence", 45, 80, 55)
    swing_left = st.number_input("Swing left bars", 1, 8, 3)
    swing_right = st.number_input("Swing confirmation bars", 1, 8, 3)
    st.subheader("Risk planning")
    account_size = st.number_input("Account size", min_value=100.0, value=25000.0, step=1000.0)
    risk_profile = st.selectbox("Risk profile", ["conservative", "moderate", "aggressive"], index=1)
    custom_risk = st.checkbox("Set risk % manually")
    risk_pct = st.number_input("Risk %", 0.1, 3.0, 1.0, 0.1) if custom_risk else None
    st.subheader("Live chart")
    auto_refresh = st.checkbox("Auto-refresh data", value=True)
    refresh_interval = st.selectbox(
        "Refresh interval", [15, 30, 60], index=2,
        format_func=lambda seconds: f"Every {seconds} seconds",
    )
    st.caption("Polling follows provider delays and limits; it is not tick-by-tick trade data.")
    run_btn = st.button("Start / refresh analysis", type="primary", use_container_width=True)


if run_btn:
    st.session_state["analysis_started"] = True
if not st.session_state.get("analysis_started", False):
    st.title("MarketPulse AI")
    st.caption("Top-down context · Intraday confirmation · Visible strategy evidence")
    st.info("Choose an instrument and click **Start / refresh analysis** to load online candles.")
    st.stop()

fragment_interval = refresh_interval if auto_refresh else None

@st.fragment(run_every=fragment_interval)
def render_dashboard():
    st.title("MarketPulse AI")
    st.caption("Top-down context · Intraday confirmation · Visible strategy evidence")
    provider_symbol = symbol.strip()
    try:
        frames, closed_frames, provider_label = load_market_data(
            asset_class, provider_symbol, start_date, provider_name
        )
    except Exception as exc:  # noqa: BLE001 - display varied provider/network errors in the UI.
        st.error(f"Online data request failed: {exc}")
        st.caption("Check the symbol, network, provider rate limits, and any required API key. Synthetic sample data is never substituted.")
        st.stop()

    if not closed_frames:
        st.error("The feed returned no fully closed candles for these settings.")
        st.stop()
    chart_frame = closed_frames.get(chart_tf)
    if chart_frame is None or chart_frame.empty:
        available = ", ".join(closed_frames)
        st.error(f"The provider did not return {chart_tf} candles. Available intervals: {available}")
        st.stop()
    display_frame = frames.get(chart_tf, chart_frame)
    if display_frame is None or display_frame.empty:
        display_frame = chart_frame

    current_price = float(display_frame["close"].iloc[-1])
    confluence = calculate_confluence(
        symbol=symbol,
        timeframes=closed_frames,
        current_price=current_price,
        swing_left=int(swing_left),
        swing_right=int(swing_right),
        entry_timeframe=chart_tf,
    )
    swings = confirmed_swings(chart_frame, left=int(swing_left), right=int(swing_right))
    structure = detect_structure_breaks(chart_frame, swings)
    fvgs = detect_fair_value_gaps(chart_frame)
    obs = detect_order_blocks(chart_frame, structure)
    sd = enrich_supply_demand_zones(chart_frame, swings)
    zone_parts = []
    if show_fvg and not fvgs.empty:
        zone_parts.append(fvgs)
    if show_ob and not obs.empty:
        zone_parts.append(obs)
    if show_sd and not sd.empty:
        zone_parts.append(sd)
    chart_zone_timeframe = {"4h": "4H", "1h": "1H"}.get(chart_tf)
    used_higher_zones = []
    for zone in confluence.active_zones_nearby:
        zone_type = str(zone.get("zone_type", ""))
        layer_enabled = (
            (zone_type == "FVG" and show_fvg)
            or (zone_type == "ORDER_BLOCK" and show_ob)
            or (zone_type in ("SUPPLY", "DEMAND") and show_sd)
        )
        if layer_enabled and zone.get("source_timeframe") != chart_zone_timeframe:
            used_higher_zones.append({
                **zone,
                "mid": (float(zone["lower"]) + float(zone["upper"])) / 2,
                "formed_at": None,
                "is_mitigated": False,
            })
    if used_higher_zones:
        zone_parts.append(pd.DataFrame(used_higher_zones))
    zones_df = pd.concat(zone_parts, ignore_index=True) if zone_parts else pd.DataFrame()

    patterns = detect_candlestick_patterns(chart_frame)
    marker_parts = []
    if show_patterns and not patterns.empty:
        marker_parts.append(patterns.assign(label=patterns["pattern"], kind="pattern"))
    if show_structure and not structure.empty:
        marker_parts.append(pd.DataFrame({
            "time": structure["time"], "price": structure["level"],
            "label": structure["event"] + " " + structure["direction"],
            "direction": structure["direction"], "kind": "structure",
        }))
    markers_df = pd.concat(marker_parts, ignore_index=True) if marker_parts else pd.DataFrame()

    true_range = pd.concat([
        chart_frame["high"] - chart_frame["low"],
        (chart_frame["high"] - chart_frame["close"].shift()).abs(),
        (chart_frame["low"] - chart_frame["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = float(true_range.rolling(14, min_periods=1).mean().iloc[-1])
    plan = generate_trade_plan(confluence, current_price, atr=atr, min_confluence=float(min_confluence))
    state = build_chart_state(
        symbol=symbol, timeframe=chart_tf, ohlcv=display_frame, zones_df=zones_df,
        confluence=confluence, trade_plan=plan, max_bars=len(display_frame), markers_df=markers_df,
        swing_points=swings if show_trendlines else None,
        structure_events=structure if show_structure else None,
        draw_trendlines=show_trendlines,
    )
    waiting = assess_waiting_state(
        frame=chart_frame,
        timeframe=chart_tf,
        bias=confluence.bias,
        confluence_score=confluence.confluence_score,
        minimum_confluence=float(min_confluence),
        zone_score=float(confluence.zone_details.get("zone_score", 0)),
        confirmation_found=bool(confluence.entry_details.get("has_confirmation")),
        setup_ready=plan.is_valid,
        swings=swings,
        htf_details=confluence.htf_details,
    )

    age = pd.Timestamp.now(tz="UTC") - chart_frame.index[-1]
    age_text = f"Latest completed bar: {chart_frame.index[-1].strftime('%Y-%m-%d %H:%M UTC')} · {age.total_seconds()/60:.0f} min ago"
    live_bar = display_frame.index[-1] > chart_frame.index[-1]
    update_text = (
        f"Current candle updating ({display_frame.index[-1].strftime('%Y-%m-%d %H:%M UTC')})"
        if live_bar else "Provider has not returned a forming candle"
    )
    st.caption(f"{provider_label} · {age_text} · {update_text} · Candle times are UTC. Confirm feed timing with your provider.")
    st.subheader("What the analysis is waiting for")
    if waiting["ready"]:
        st.success("The current setup rules are met. Treat this as a candidate and verify the chart before acting.")
    else:
        st.info("No complete setup yet. The checks below show what is missing.")
    wait_left, wait_right = st.columns([1.6, 1])
    with wait_left:
        for check in waiting["checks"]:
            icon = "✅" if check["met"] else "◻️"
            st.markdown(f"{icon} {check['text']}")
    with wait_right:
        st.markdown(f"**Current momentum:** {waiting['momentum']}")
        st.markdown(f"**Timing reference:** {waiting['reference']}")
        st.markdown(f"**Momentum-based rough ETA:** {waiting['timing']}")
        st.caption(str(waiting["timing_method"]))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Last price", f"{current_price:,.4f}")
    m2.metric("Top-down bias", confluence.bias.upper())
    m3.metric("Confluence", f"{confluence.confluence_score:.1f}%")
    m4.metric("Confidence", confluence.confidence)
    m5.metric("Trade plan", "SETUP" if plan.is_valid else "WAIT")

    st.subheader(f"{symbol} · {chart_tf} · {len(state.candles):,} candles")
    st.plotly_chart(
        create_plotly_figure(state, height=850),
        use_container_width=True,
        theme=None,
        config={"scrollZoom": True, "displaylogo": False},
    )
    st.caption("Time gaps without provider candles are compressed; every displayed candle comes from the feed. Key: bullish/bearish BOS = green ▲/red ▼ · bullish/bearish CHoCH = cyan/orange circles · bullish/bearish candle patterns = purple/magenta diamonds · rising/falling swing lines = teal/orange. Zones: FVG green/red fills, order blocks blue/orange fills, dotted borders for supply/demand. Hover a mark or line for details.")

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("Trade thesis")
        if plan.is_valid:
            st.success(f"{plan.direction.upper()} setup candidate")
            st.write(f"**Entry zone:** {plan.entry_low:,.4f} – {plan.entry_high:,.4f} · {plan.entry_zone_type}")
            st.write(f"**Stop:** {plan.stop_loss:,.4f}")
            st.write(f"**Targets:** {plan.take_profit_1:,.4f} ({plan.risk_reward_1}R) · {plan.take_profit_2:,.4f} ({plan.risk_reward_2}R)")
            st.write(f"**Invalidation:** {plan.invalidation}")
        else:
            st.warning("No qualified setup · wait for better evidence")
        st.caption(plan.reasoning)

        if plan.is_valid and plan.entry_mid and plan.stop_loss:
            st.markdown("#### Position sizing illustration")
            try:
                pos = calculate_position_size(
                    symbol=symbol, direction=plan.direction, account_size=account_size,
                    entry_price=plan.entry_mid, stop_loss=plan.stop_loss,
                    risk_percent=risk_pct, confidence=plan.confidence,
                    confluence_score=plan.confluence_score, profile=risk_profile,
                )
                st.write(f"Risk amount: **{pos.risk_amount:,.2f}** · units: **{pos.position_size:,.4f}**")
                st.caption("Position value assumes the account currency matches the instrument quote currency; check contract sizing and FX conversion.")
            except ValueError as exc:
                st.warning(f"Position size unavailable: {exc}")

    with right:
        st.markdown("#### Why this setup scores")
        if not structure.empty:
            latest_structure = structure.iloc[-1]
            st.write(f"- Latest {chart_tf} structure: **{latest_structure['event']} {latest_structure['direction']}** at {float(latest_structure['level']):,.5f}")
        else:
            st.write(f"- Latest {chart_tf} structure: no confirmed BOS/CHoCH in this lookback")
        htf = confluence.htf_details
        st.write(f"- Weekly structure: **{htf.get('weekly_trend', 'n/a')}**")
        st.write(f"- Daily structure: **{htf.get('daily_trend', 'n/a')}**")
        zone = confluence.zone_details
        if zone.get("best_zone_type"):
            st.write(f"- Nearby aligned zone: **{zone['best_zone_type']}** · strength {zone.get('best_zone_strength', 0):.0f}/100 · {zone.get('distance_pct')}% away")
        else:
            st.write("- No nearby zone aligned with the current bias")
        entry = confluence.entry_details
        if entry.get("has_confirmation"):
            st.write(f"- {chart_tf} confirmation: **{entry.get('confirmation_type')}** · {entry.get('details')}")
        else:
            st.write(f"- {chart_tf} confirmation: not detected")
        if not patterns.empty:
            recent_patterns = patterns.tail(3)
            st.write("- Recent candle patterns: " + ", ".join(recent_patterns["pattern"].astype(str)))
        st.caption("Confluence uses measurable rules only. No trained probability model or verified win-rate claim is included.")

    with st.expander("Strategy evidence on this chart"):
        st.write("Enabled layers: " + ", ".join(name for enabled, name in [
            (show_fvg, "Fair Value Gaps"), (show_ob, "Order Blocks"),
            (show_sd, "Supply / Demand"), (show_patterns, "Candle Patterns"),
            (show_structure, "BOS/CHoCH levels"), (show_trendlines, "Swing trendlines"),
        ] if enabled))
        if not zones_df.empty:
            st.dataframe(get_active_zones(zones_df, min_strength=0).head(50), use_container_width=True, hide_index=True)
        else:
            st.info("No enabled active zones detected in the selected timeframe.")
        if (show_patterns or show_structure) and not markers_df.empty:
            st.dataframe(markers_df.tail(30), use_container_width=True, hide_index=True)
    with st.expander("Full confluence breakdown"):
        st.json(confluence.to_dict())

    st.caption("Educational research software. Not financial advice. Signals can fail; verify fees, spreads, contract size, and provider data before acting.")


render_dashboard()
