from __future__ import annotations

import pandas as pd


def detect_candlestick_patterns(frame: pd.DataFrame, lookback: int = 120) -> pd.DataFrame:
    """Find simple, explicit engulfing and rejection-wick patterns on closed bars."""
    columns = ["time", "price", "pattern", "direction", "reason"]
    if frame is None or len(frame) < 2:
        return pd.DataFrame(columns=columns)

    df = frame.tail(lookback)
    rows: list[dict[str, object]] = []
    for i in range(1, len(df)):
        previous, candle = df.iloc[i - 1], df.iloc[i]
        o, h, l, c = (float(candle[k]) for k in ("open", "high", "low", "close"))
        po, pc = float(previous["open"]), float(previous["close"])
        body = max(abs(c - o), 1e-12)
        upper_wick, lower_wick = h - max(o, c), min(o, c) - l
        pattern = direction = reason = None

        if pc < po and c > o and o <= pc and c >= po:
            pattern, direction = "Bullish engulfing", "bullish"
            reason = "Current bullish body engulfs the prior bearish body."
        elif pc > po and c < o and o >= pc and c <= po:
            pattern, direction = "Bearish engulfing", "bearish"
            reason = "Current bearish body engulfs the prior bullish body."
        elif lower_wick >= body * 2.5 and upper_wick <= body:
            pattern, direction = "Lower-wick rejection", "bullish"
            reason = "Lower wick is at least 2.5 times the candle body."
        elif upper_wick >= body * 2.5 and lower_wick <= body:
            pattern, direction = "Upper-wick rejection", "bearish"
            reason = "Upper wick is at least 2.5 times the candle body."

        if pattern:
            rows.append({
                "time": df.index[i], "price": c, "pattern": pattern,
                "direction": direction, "reason": reason,
            })
    return pd.DataFrame(rows, columns=columns)
