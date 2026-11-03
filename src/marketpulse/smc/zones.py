from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

ZoneDirection = Literal["bullish", "bearish"]


def _calculate_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Simple ATR used for zone strength normalization."""
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _check_mitigation(
    frame: pd.DataFrame,
    formed_index: int,
    lower: float,
    upper: float,
    direction: ZoneDirection,
) -> tuple[bool, pd.Timestamp | None]:
    """
    Check whether a zone has been mitigated after it was formed.

    Rules (practical for live charting):
    - Bullish zone is mitigated when price trades fully through the zone to the downside
      (low goes below the lower boundary).
    - Bearish zone is mitigated when price trades fully through the zone to the upside
      (high goes above the upper boundary).
    """
    if formed_index >= len(frame) - 1:
        return False, None

    future = frame.iloc[formed_index + 1 :]

    if direction == "bullish":
        # Mitigated if price closes or wicks fully below the zone
        hits = future[future["low"] < lower]
    else:
        hits = future[future["high"] > upper]

    if hits.empty:
        return False, None

    return True, hits.index[0]


def detect_fair_value_gaps(
    frame: pd.DataFrame,
    min_gap_atr_mult: float = 0.15,
) -> pd.DataFrame:
    """
    Detect three-candle Fair Value Gaps with mitigation status and strength.

    Bullish FVG : low[i] > high[i-2]
    Bearish FVG : high[i] < low[i-2]

    Strength is based on gap size relative to ATR (larger gap = stronger imbalance).
    """
    if len(frame) < 3:
        return pd.DataFrame()

    atr = _calculate_atr(frame)
    rows: list[dict[str, object]] = []

    for i in range(2, len(frame)):
        first = frame.iloc[i - 2]
        third = frame.iloc[i]
        timestamp = frame.index[i]
        current_atr = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.0

        # Bullish FVG
        if float(third["low"]) > float(first["high"]):
            lower = float(first["high"])
            upper = float(third["low"])
            gap_size = upper - lower

            if current_atr > 0 and gap_size < current_atr * min_gap_atr_mult:
                continue  # ignore very small noise gaps

            strength = min(100.0, (gap_size / current_atr) * 40) if current_atr > 0 else 30.0
            is_mitigated, mitigated_at = _check_mitigation(frame, i, lower, upper, "bullish")

            rows.append({
                "zone_type": "FVG",
                "direction": "bullish",
                "lower": lower,
                "upper": upper,
                "mid": (lower + upper) / 2,
                "formed_at": timestamp,
                "source_index": i,
                "is_mitigated": is_mitigated,
                "mitigated_at": mitigated_at,
                "strength": round(strength, 1),
                "gap_size": round(gap_size, 6),
                "definition": "low[i] > high[i-2]",
            })

        # Bearish FVG
        if float(third["high"]) < float(first["low"]):
            lower = float(third["high"])
            upper = float(first["low"])
            gap_size = upper - lower

            if current_atr > 0 and gap_size < current_atr * min_gap_atr_mult:
                continue

            strength = min(100.0, (gap_size / current_atr) * 40) if current_atr > 0 else 30.0
            is_mitigated, mitigated_at = _check_mitigation(frame, i, lower, upper, "bearish")

            rows.append({
                "zone_type": "FVG",
                "direction": "bearish",
                "lower": lower,
                "upper": upper,
                "mid": (lower + upper) / 2,
                "formed_at": timestamp,
                "source_index": i,
                "is_mitigated": is_mitigated,
                "mitigated_at": mitigated_at,
                "strength": round(strength, 1),
                "gap_size": round(gap_size, 6),
                "definition": "high[i] < low[i-2]",
            })

    columns = [
        "zone_type", "direction", "lower", "upper", "mid",
        "formed_at", "source_index", "is_mitigated", "mitigated_at",
        "strength", "gap_size", "definition"
    ]
    return pd.DataFrame(rows, columns=columns)


def detect_order_blocks(
    frame: pd.DataFrame,
    structure_events: pd.DataFrame,
    lookback: int = 12,
    min_body_atr_mult: float = 0.25,
) -> pd.DataFrame:
    """
    Detect Order Blocks linked to structure breaks.

    Logic:
    - After a confirmed BOS/CHoCH, look backward for the last opposing candle
      (bearish candle before bullish break, bullish candle before bearish break).
    - Prefer candles with decent body size relative to ATR.
    - Track mitigation status for live chart filtering.
    """
    if structure_events.empty or len(frame) < 5:
        return pd.DataFrame(columns=[
            "zone_type", "direction", "lower", "upper", "mid", "formed_at",
            "triggered_at", "source_index", "is_mitigated", "mitigated_at",
            "strength", "body_size", "definition",
        ])

    atr = _calculate_atr(frame)
    rows: list[dict[str, object]] = []

    for event in structure_events.to_dict("records"):
        break_time = event["time"]
        try:
            break_index = int(frame.index.get_loc(break_time))
        except KeyError:
            continue

        direction = str(event["direction"])
        start = max(0, break_index - lookback)
        search = frame.iloc[start:break_index]

        if search.empty:
            continue

        if direction == "bullish":
            # Last bearish candle before the break
            candidates = search[search["close"] < search["open"]]
        else:
            # Last bullish candle before the break
            candidates = search[search["close"] > search["open"]]

        if candidates.empty:
            continue

        # Prefer the most recent opposing candle with reasonable body
        selected = None
        for ts in reversed(candidates.index):
            candle = frame.loc[ts]
            body = abs(float(candle["close"]) - float(candle["open"]))
            idx = int(frame.index.get_loc(ts))
            current_atr = float(atr.iloc[idx]) if not np.isnan(atr.iloc[idx]) else 0.0

            if current_atr > 0 and body < current_atr * min_body_atr_mult:
                continue  # too small / doji-like

            selected = (ts, idx, candle, body, current_atr)
            break

        if selected is None:
            # fallback to the last opposing candle even if small
            ts = candidates.index[-1]
            candle = frame.loc[ts]
            idx = int(frame.index.get_loc(ts))
            body = abs(float(candle["close"]) - float(candle["open"]))
            current_atr = float(atr.iloc[idx]) if not np.isnan(atr.iloc[idx]) else 0.0
            selected = (ts, idx, candle, body, current_atr)

        ts, idx, candle, body, current_atr = selected
        lower = float(candle["low"])
        upper = float(candle["high"])

        # Strength: combination of body size and how clean the later displacement was
        strength = 45.0
        if current_atr > 0:
            strength = min(95.0, 35 + (body / current_atr) * 40)

        is_mitigated, mitigated_at = _check_mitigation(frame, idx, lower, upper, direction)  # type: ignore

        rows.append({
            "zone_type": "ORDER_BLOCK",
            "direction": direction,
            "lower": lower,
            "upper": upper,
            "mid": (lower + upper) / 2,
            "formed_at": ts,
            "triggered_at": break_time,
            "source_index": idx,
            "is_mitigated": is_mitigated,
            "mitigated_at": mitigated_at,
            "strength": round(strength, 1),
            "body_size": round(body, 6),
            "definition": "last opposing candle before structure break",
        })

    columns = [
        "zone_type", "direction", "lower", "upper", "mid",
        "formed_at", "triggered_at", "source_index",
        "is_mitigated", "mitigated_at", "strength", "body_size", "definition"
    ]
    return pd.DataFrame(rows, columns=columns)


def enrich_supply_demand_zones(
    frame: pd.DataFrame,
    swings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create Supply / Demand zones from confirmed swing pivots.

    - Swing High → Supply zone
    - Swing Low  → Demand zone

    Includes mitigation status so the live chart can hide already-tested zones.
    """
    if swings.empty:
        return pd.DataFrame(columns=[
            "zone_type", "direction", "lower", "upper", "mid", "formed_at",
            "confirmed_at", "source_index", "is_mitigated", "mitigated_at",
            "strength", "definition",
        ])

    atr = _calculate_atr(frame)
    rows: list[dict[str, object]] = []

    for swing in swings.to_dict("records"):
        idx = int(swing["pivot_index"])
        if idx >= len(frame):
            continue

        candle = frame.iloc[idx]
        is_high = swing["type"] == "swing_high"
        direction: ZoneDirection = "bearish" if is_high else "bullish"
        lower = float(candle["low"])
        upper = float(candle["high"])
        zone_range = upper - lower

        current_atr = float(atr.iloc[idx]) if not np.isnan(atr.iloc[idx]) else 0.0
        strength = 40.0
        if current_atr > 0:
            # Tighter zones relative to ATR are often more respected
            strength = min(90.0, 55 - (zone_range / current_atr) * 15)
            strength = max(25.0, strength)

        is_mitigated, mitigated_at = _check_mitigation(frame, idx, lower, upper, direction)

        rows.append({
            "zone_type": "SUPPLY" if is_high else "DEMAND",
            "direction": direction,
            "lower": lower,
            "upper": upper,
            "mid": (lower + upper) / 2,
            "formed_at": swing["pivot_time"],
            "confirmed_at": swing["confirmed_at"],
            "source_index": idx,
            "is_mitigated": is_mitigated,
            "mitigated_at": mitigated_at,
            "strength": round(strength, 1),
            "definition": "swing pivot range",
        })

    columns = [
        "zone_type", "direction", "lower", "upper", "mid",
        "formed_at", "confirmed_at", "source_index",
        "is_mitigated", "mitigated_at", "strength", "definition"
    ]
    return pd.DataFrame(rows, columns=columns)


def get_active_zones(
    zones: pd.DataFrame,
    min_strength: float = 35.0,
    only_unmitigated: bool = True,
) -> pd.DataFrame:
    """
    Helper for live chart and confluence engine.

    Returns only high-quality, still-valid zones.
    """
    if zones.empty:
        return zones

    result = zones.copy()
    if only_unmitigated and "is_mitigated" in result.columns:
        result = result[result["is_mitigated"] == False]

    if "strength" in result.columns:
        result = result[result["strength"] >= min_strength]

    return result.sort_values("strength", ascending=False).reset_index(drop=True)
