from __future__ import annotations

import pandas as pd


def confirmed_swings(frame: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Find pivot highs/lows, recording when each pivot becomes knowable.

    A pivot at position i is only emitted at i + right. This avoids using a
    swing before the required right-side candles have closed.
    """
    if left < 1 or right < 1:
        raise ValueError("left and right must both be >= 1")
    rows: list[dict[str, object]] = []
    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    index = frame.index

    for pivot_i in range(left, len(frame) - right):
        left_highs = highs[pivot_i - left : pivot_i]
        right_highs = highs[pivot_i + 1 : pivot_i + right + 1]
        left_lows = lows[pivot_i - left : pivot_i]
        right_lows = lows[pivot_i + 1 : pivot_i + right + 1]

        if highs[pivot_i] > left_highs.max() and highs[pivot_i] >= right_highs.max():
            rows.append(
                {
                    "type": "swing_high",
                    "price": float(highs[pivot_i]),
                    "pivot_time": index[pivot_i],
                    "confirmed_at": index[pivot_i + right],
                    "pivot_index": pivot_i,
                    "confirmed_index": pivot_i + right,
                }
            )
        if lows[pivot_i] < left_lows.min() and lows[pivot_i] <= right_lows.min():
            rows.append(
                {
                    "type": "swing_low",
                    "price": float(lows[pivot_i]),
                    "pivot_time": index[pivot_i],
                    "confirmed_at": index[pivot_i + right],
                    "pivot_index": pivot_i,
                    "confirmed_index": pivot_i + right,
                }
            )

    return pd.DataFrame(rows, columns=[
        "type", "price", "pivot_time", "confirmed_at", "pivot_index", "confirmed_index"
    ])


def detect_structure_breaks(frame: pd.DataFrame, swings: pd.DataFrame) -> pd.DataFrame:
    """Detect close-based BOS/CHoCH events from already-confirmed swing levels."""
    if swings.empty:
        return pd.DataFrame(columns=["event", "direction", "level", "time", "swing_time", "trend_before"])

    swing_map: dict[int, list[dict[str, object]]] = {}
    for swing in swings.to_dict("records"):
        swing_map.setdefault(int(swing["confirmed_index"]), []).append(swing)

    last_high: dict[str, object] | None = None
    last_low: dict[str, object] | None = None
    broken_high_time: pd.Timestamp | None = None
    broken_low_time: pd.Timestamp | None = None
    trend: str | None = None
    events: list[dict[str, object]] = []

    for i, (timestamp, candle) in enumerate(frame.iterrows()):
        for swing in swing_map.get(i, []):
            if swing["type"] == "swing_high":
                last_high = swing
                broken_high_time = None
            else:
                last_low = swing
                broken_low_time = None

        close = float(candle["close"])
        if last_high and close > float(last_high["price"]):
            if broken_high_time != last_high["confirmed_at"]:
                event = "BOS" if trend in (None, "bullish") else "CHoCH"
                events.append({
                    "event": event,
                    "direction": "bullish",
                    "level": float(last_high["price"]),
                    "time": timestamp,
                    "swing_time": last_high["pivot_time"],
                    "trend_before": trend,
                })
                trend = "bullish"
                broken_high_time = last_high["confirmed_at"]

        if last_low and close < float(last_low["price"]):
            if broken_low_time != last_low["confirmed_at"]:
                event = "BOS" if trend in (None, "bearish") else "CHoCH"
                events.append({
                    "event": event,
                    "direction": "bearish",
                    "level": float(last_low["price"]),
                    "time": timestamp,
                    "swing_time": last_low["pivot_time"],
                    "trend_before": trend,
                })
                trend = "bearish"
                broken_low_time = last_low["confirmed_at"]

    return pd.DataFrame(events, columns=["event", "direction", "level", "time", "swing_time", "trend_before"])
