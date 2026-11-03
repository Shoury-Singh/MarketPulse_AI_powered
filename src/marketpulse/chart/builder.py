from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from marketpulse.chart.models import (
    ChartBiasBadge,
    ChartCandle,
    ChartLabel,
    ChartLine,
    ChartMarker,
    ChartTrendLine,
    ChartState,
    ChartZone,
)
from marketpulse.smc.confluence import ConfluenceResult
from marketpulse.smc.trade_plan import TradePlan
from marketpulse.smc.zones import get_active_zones


def _ts(value: Any) -> str:
    """Normalize timestamps to ISO strings."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)


def _zone_color(direction: str, zone_type: str, is_entry: bool = False) -> tuple[str, str]:
    """Return (fill, border) colors."""
    if is_entry:
        if direction == "bullish" or direction == "long":
            return "rgba(34, 197, 94, 0.18)", "#16a34a"
        return "rgba(239, 68, 68, 0.18)", "#dc2626"

    if zone_type in ("DEMAND", "FVG") and direction == "bullish":
        return "rgba(34, 197, 94, 0.12)", "#22c55e"
    if zone_type in ("SUPPLY", "FVG") and direction == "bearish":
        return "rgba(239, 68, 68, 0.12)", "#ef4444"
    if zone_type == "ORDER_BLOCK":
        if direction == "bullish":
            return "rgba(59, 130, 246, 0.12)", "#3b82f6"
        return "rgba(249, 115, 22, 0.12)", "#f97316"

    return "rgba(148, 163, 184, 0.10)", "#94a3b8"


def build_candles(frame: pd.DataFrame, max_bars: int = 300) -> list[ChartCandle]:
    """Convert OHLCV DataFrame into chart candles (most recent first limited)."""
    if frame is None or frame.empty:
        return []

    df = frame.tail(max_bars).copy()
    candles: list[ChartCandle] = []

    for ts, row in df.iterrows():
        candles.append(
            ChartCandle(
                time=_ts(ts),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0)),
            )
        )
    return candles


def build_zones_from_dataframe(
    zones_df: pd.DataFrame,
    only_active: bool = True,
    min_strength: float = 35.0,
) -> list[ChartZone]:
    """Turn zone detection output into chart zones."""
    if zones_df is None or zones_df.empty:
        return []

    if only_active:
        zones_df = get_active_zones(zones_df, min_strength=min_strength)

    result: list[ChartZone] = []
    for i, row in zones_df.iterrows():
        direction = str(row.get("direction", "bullish"))
        zone_type = str(row.get("zone_type", "ZONE"))
        source_timeframe = str(row.get("source_timeframe", "")).strip()
        fill, border = _zone_color(direction, zone_type)

        result.append(
            ChartZone(
                id=f"zone_{i}_{zone_type}",
                zone_type=zone_type,
                direction=direction,
                lower=float(row["lower"]),
                upper=float(row["upper"]),
                start_time=_ts(row.get("formed_at")),
                end_time=None,  # still active → extends to the right
                strength=float(row.get("strength", 50)),
                is_mitigated=bool(row.get("is_mitigated", False)),
                label=f"{source_timeframe + ' ' if source_timeframe else ''}{zone_type} ({direction})",
                color=fill,
                border_color=border,
            )
        )
    return result


def build_chart_state(
    symbol: str,
    timeframe: str,
    ohlcv: pd.DataFrame,
    zones_df: pd.DataFrame | None = None,
    confluence: ConfluenceResult | None = None,
    trade_plan: TradePlan | None = None,
    max_bars: int = 300,
    markers_df: pd.DataFrame | None = None,
    swing_points: pd.DataFrame | None = None,
    structure_events: pd.DataFrame | None = None,
    draw_trendlines: bool = True,
) -> ChartState:
    """
    Main builder.

    Takes raw analysis outputs and produces a complete ChartState
    that any frontend (Streamlit + Plotly, Lightweight Charts, etc.)
    can render without further business logic.
    """

    candles = build_candles(ohlcv, max_bars=max_bars)
    zones: list[ChartZone] = []
    lines: list[ChartLine] = []
    labels: list[ChartLabel] = []
    bias_badge = None
    trade_plan_active = False
    notes: list[str] = []
    markers: list[ChartMarker] = []
    trendlines: list[ChartTrendLine] = []
    visible_start = pd.Timestamp(candles[0].time) if candles else None

    if structure_events is not None and not structure_events.empty:
        visible_events = structure_events
        if visible_start is not None:
            visible_events = visible_events[visible_events["time"] >= visible_start]
        for event in visible_events.tail(3).to_dict("records"):
            direction = str(event["direction"])
            level = float(event["level"])
            current_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else level
            if current_price and abs(level - current_price) / current_price > 0.035:
                continue
            up = direction == "bullish"
            event_name = str(event["event"])
            lines.append(ChartLine(
                id=f"structure_{event_name}_{event['time']}",
                price=level,
                label=f"{event_name} {'↑' if up else '↓'} · {timeframe}",
                line_type="dotted",
                color="#34d399" if up else "#fb7185",
                width=1,
                start_time=_ts(event["time"]),
                end_time=_ts(ohlcv.index[-1]) if not ohlcv.empty else None,
            ))

    if draw_trendlines and swing_points is not None and not swing_points.empty and not ohlcv.empty:
        visible_swings = swing_points
        if visible_start is not None:
            visible_swings = visible_swings[visible_swings["pivot_time"] >= visible_start]
        for swing_type, label_prefix in (("swing_low", "Swing-low support"), ("swing_high", "Swing-high resistance")):
            points = visible_swings[visible_swings["type"] == swing_type].tail(2)
            if len(points) < 2:
                continue
            first, second = points.iloc[0], points.iloc[1]
            bar_distance = int(second["pivot_index"]) - int(first["pivot_index"])
            if bar_distance <= 0:
                continue
            slope = (float(second["price"]) - float(first["price"])) / bar_distance
            projection = int(second["pivot_index"])
            end_price = float(second["price"])
            rising = slope > 0
            trendlines.append(ChartTrendLine(
                start_time=_ts(first["pivot_time"]),
                start_price=float(first["price"]),
                end_time=_ts(ohlcv.index[projection]),
                end_price=end_price,
                label=f"{label_prefix} {'↗' if rising else '↘'} · 2 pivots",
                color="#2dd4bf" if rising else "#fb923c",
            ))
    if markers_df is not None and not markers_df.empty:
        for row in markers_df.to_dict("records"):
            marker_time = pd.Timestamp(row["time"])
            if visible_start is not None and marker_time < visible_start:
                continue
            markers.append(ChartMarker(
                time=_ts(row["time"]), price=float(row["price"]),
                label=str(row.get("label", row.get("pattern", row.get("event", "Signal")))),
                direction=str(row.get("direction", "bullish")),
                kind=str(row.get("kind", "pattern")),
            ))

    # 1. SMC Zones
    if zones_df is not None and not zones_df.empty:
        zones = build_zones_from_dataframe(zones_df, only_active=True)

    # 2. Confluence → Bias Badge
    if confluence is not None:
        bias_color = {
            "bullish": "#16a34a",
            "bearish": "#dc2626",
            "neutral": "#64748b",
        }.get(confluence.bias, "#64748b")

        bias_badge = ChartBiasBadge(
            bias=confluence.bias,
            score=confluence.confluence_score,
            confidence=confluence.confidence,
            suggestion=confluence.suggestion,
            color=bias_color,
        )

    # 3. Trade Plan → Entry zone + SL + TPs as special layers
    if trade_plan is not None and trade_plan.is_valid:
        trade_plan_active = True

        # Entry zone
        fill, border = _zone_color(
            trade_plan.direction,
            trade_plan.entry_zone_type or "ENTRY",
            is_entry=True,
        )
        zones.append(
            ChartZone(
                id="entry_zone",
                zone_type="ENTRY",
                direction=trade_plan.direction,
                lower=trade_plan.entry_low or 0,
                upper=trade_plan.entry_high or 0,
                start_time=None,
                end_time=None,
                strength=90,
                is_mitigated=False,
                label=f"ENTRY ({trade_plan.entry_zone_type})",
                color=fill,
                border_color=border,
            )
        )

        # Stop Loss line
        if trade_plan.stop_loss is not None:
            lines.append(
                ChartLine(
                    id="stop_loss",
                    price=trade_plan.stop_loss,
                    label="SL",
                    line_type="dashed",
                    color="#ef4444",
                    width=2,
                )
            )
            labels.append(
                ChartLabel(
                    id="sl_label",
                    text=f"SL {trade_plan.stop_loss:.2f}",
                    price=trade_plan.stop_loss,
                    position="right",
                    color="#fecaca",
                    background="rgba(127, 29, 29, 0.7)",
                )
            )

        # Take Profits
        if trade_plan.take_profit_1 is not None:
            lines.append(
                ChartLine(
                    id="tp1",
                    price=trade_plan.take_profit_1,
                    label=f"TP1 ({trade_plan.risk_reward_1}R)",
                    line_type="dashed",
                    color="#3b82f6",
                    width=1,
                )
            )
        if trade_plan.take_profit_2 is not None:
            lines.append(
                ChartLine(
                    id="tp2",
                    price=trade_plan.take_profit_2,
                    label=f"TP2 ({trade_plan.risk_reward_2}R)",
                    line_type="dotted",
                    color="#60a5fa",
                    width=1,
                )
            )

        notes.append(trade_plan.suggestion)

    state = ChartState(
        symbol=symbol,
        timeframe=timeframe,
        candles=candles,
        zones=zones,
        lines=lines,
        labels=labels,
        markers=markers,
        trendlines=trendlines,
        bias_badge=bias_badge,
        trade_plan_active=trade_plan_active,
        last_updated=datetime.now(UTC).isoformat(),
        notes=notes,
    )
    return state
