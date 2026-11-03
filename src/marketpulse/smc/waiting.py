from __future__ import annotations

import math

import pandas as pd

TIMEFRAME_SECONDS = {
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}


def assess_waiting_state(
    frame: pd.DataFrame,
    timeframe: str,
    bias: str,
    confluence_score: float,
    minimum_confluence: float,
    zone_score: float,
    confirmation_found: bool,
    setup_ready: bool,
    swings: pd.DataFrame | None = None,
    htf_details: dict[str, object] | None = None,
) -> dict[str, object]:
    """Describe the remaining rule checks and a cautious momentum-based time range."""
    checks: list[dict[str, object]] = []
    weekly = str((htf_details or {}).get("weekly_trend", "neutral"))
    daily = str((htf_details or {}).get("daily_trend", "neutral"))
    conflicting_htf = weekly != "neutral" and daily != "neutral" and weekly != daily
    bias_clear = bias != "neutral" and not conflicting_htf
    checks.append({
        "met": bias_clear,
        "text": (
            f"Higher-timeframe direction is {bias}."
            if bias_clear
            else (
                f"Weekly ({weekly}) and daily ({daily}) structure conflict; wait for alignment."
                if conflicting_htf
                else "Wait for weekly and daily structure to establish a direction."
            )
        ),
    })
    checks.append({
        "met": zone_score >= 40,
        "text": (
            f"Aligned zone quality is {zone_score:.0f}/100."
            if zone_score >= 40
            else "Wait for price to reach a fresh demand/supply zone, order block, or fair value gap aligned with the bias."
        ),
    })
    checks.append({
        "met": confirmation_found,
        "text": (
            f"{timeframe} entry confirmation is present."
            if confirmation_found
            else f"Wait for a completed {timeframe} candle to confirm with a bias-aligned BOS/CHoCH or fresh FVG."
        ),
    })
    checks.append({
        "met": confluence_score >= minimum_confluence,
        "text": (
            f"Confluence is {confluence_score:.1f}, above the {minimum_confluence:.0f} threshold."
            if confluence_score >= minimum_confluence
            else f"Confluence needs to reach {minimum_confluence:.0f}; it is currently {confluence_score:.1f}."
        ),
    })

    if setup_ready:
        timing = "Setup rules are currently met; wait for your own confirmation before acting."
        reference = "Current setup candidate"
        momentum = _momentum_label(frame, bias)
    else:
        estimate_bias = bias if bias_clear else "neutral"
        reference, target = _next_structure_target(frame, estimate_bias, swings)
        momentum, timing = _estimate_time_to_level(frame, timeframe, estimate_bias, reference, target)

    return {
        "ready": setup_ready,
        "checks": checks,
        "momentum": momentum,
        "reference": reference,
        "timing": timing,
        "timing_method": "Uses net close movement over the last 10 completed candles. It is a rough scenario range, not a historical average or promise.",
    }


def _momentum_label(frame: pd.DataFrame, bias: str) -> str:
    if frame is None or len(frame) < 11:
        return "Not enough candles"
    change = float(frame["close"].iloc[-1] - frame["close"].iloc[-11])
    if abs(change) < 1e-12:
        return "Sideways"
    direction = "upward" if change > 0 else "downward"
    if bias != "neutral" and ((change > 0) == (bias == "bullish")):
        return f"{direction}, aligned with {bias} bias"
    return f"{direction}, against or separate from {bias} bias"


def _next_structure_target(
    frame: pd.DataFrame,
    bias: str,
    swings: pd.DataFrame | None,
) -> tuple[str, float | None]:
    if frame is None or frame.empty or bias == "neutral" or swings is None or swings.empty:
        return "No confirmed swing level available", None

    price = float(frame["close"].iloc[-1])
    if bias == "bullish":
        levels = swings.loc[(swings["type"] == "swing_high") & (swings["price"] > price), "price"]
        if not levels.empty:
            target = float(levels.min())
            return f"Next confirmed swing high at {target:,.5f}", target
    else:
        levels = swings.loc[(swings["type"] == "swing_low") & (swings["price"] < price), "price"]
        if not levels.empty:
            target = float(levels.max())
            return f"Next confirmed swing low at {target:,.5f}", target
    return "No confirmed swing level in the bias direction", None


def _estimate_time_to_level(
    frame: pd.DataFrame,
    timeframe: str,
    bias: str,
    reference: str,
    target: float | None,
) -> tuple[str, str]:
    momentum = _momentum_label(frame, bias)
    if frame is None or len(frame) < 12 or bias == "neutral" or target is None:
        return momentum, "No reliable timing estimate: a directional trigger level or sufficient candles is missing."

    closes = frame["close"].astype(float)
    price = float(closes.iloc[-1])
    per_bar_speed = float(closes.iloc[-1] - closes.iloc[-11]) / 10
    true_range = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - frame["close"].shift(1)).abs(),
        (frame["low"] - frame["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    typical_bar_range = float(true_range.tail(14).median())
    distance = abs(target - price)
    moving_toward = per_bar_speed * (target - price) > 0

    if typical_bar_range <= 0 or not moving_toward or abs(per_bar_speed) < typical_bar_range * 0.025:
        return momentum, f"No useful ETA to {reference}: recent momentum is too weak or points away from the level."

    expected_bars = distance / abs(per_bar_speed)
    if not math.isfinite(expected_bars) or expected_bars > 200:
        return momentum, f"No short-term ETA to {reference}; the level is too far away at the current pace."

    low_bars = max(1, math.floor(expected_bars * 0.6))
    high_bars = max(low_bars, math.ceil(expected_bars * 1.6))
    seconds = TIMEFRAME_SECONDS.get(timeframe)
    if seconds is None:
        return momentum, f"About {low_bars}–{high_bars} candles to {reference}, if momentum holds."
    low_minutes, high_minutes = low_bars * seconds / 60, high_bars * seconds / 60
    window = f"{low_minutes:.0f}–{high_minutes:.0f} minutes" if high_minutes < 120 else f"{low_minutes / 60:.1f}–{high_minutes / 60:.1f} hours"
    return momentum, f"Roughly {low_bars}–{high_bars} candles ({window}) to {reference}, if recent momentum holds."
