from __future__ import annotations

import pandas as pd

# These labels are used consistently across providers and output files.
TIMEFRAME_RULES = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1w": "W-SUN",
}


def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample normalized OHLCV candles using standard candle aggregation."""
    result = frame.resample(rule, label="left", closed="left", origin="start_day").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    result = result.dropna(subset=["open", "high", "low", "close"])
    return result.astype(float)


def build_timeframes(
    source_frames: dict[str, pd.DataFrame],
    asset_class: str,
) -> dict[str, pd.DataFrame]:
    """Build available frames from provider candles without fabricating lower intervals.

    Native source candles are preferred. Larger timeframes may be resampled
    from smaller candles; smaller timeframes are never fabricated from larger
    candles.
    """
    frames = {key: value for key, value in source_frames.items() if value is not None and not value.empty}
    if not frames:
        raise ValueError("No source timeframes were supplied.")

    output: dict[str, pd.DataFrame] = {}
    for timeframe in ("5m", "15m", "30m"):
        if timeframe in frames:
            output[timeframe] = frames[timeframe]

    if "15m" not in output and "5m" in output:
        output["15m"] = resample_ohlcv(output["5m"], TIMEFRAME_RULES["15m"])
    if "30m" not in output:
        smaller = output.get("15m", output.get("5m"))
        if smaller is not None:
            rule = TIMEFRAME_RULES["30m"] if "15m" in output else "30min"
            output["30m"] = resample_ohlcv(smaller, rule)

    if "1h" in frames:
        output["1h"] = frames["1h"]
    elif "30m" in output:
        output["1h"] = resample_ohlcv(output["30m"], TIMEFRAME_RULES["1h"])
    elif "15m" in output:
        output["1h"] = resample_ohlcv(output["15m"], TIMEFRAME_RULES["1h"])
    elif "5m" in output:
        output["1h"] = resample_ohlcv(output["5m"], TIMEFRAME_RULES["1h"])

    if "4h" in frames:
        output["4h"] = frames["4h"]
    elif "1h" in output:
        output["4h"] = resample_ohlcv(output["1h"], TIMEFRAME_RULES["4h"])

    if "1d" in frames:
        output["1d"] = frames["1d"]
    elif "1h" in output:
        output["1d"] = resample_ohlcv(output["1h"], TIMEFRAME_RULES["1d"])
    elif "30m" in output:
        output["1d"] = resample_ohlcv(output["30m"], TIMEFRAME_RULES["1d"])
    elif "15m" in output:
        output["1d"] = resample_ohlcv(output["15m"], TIMEFRAME_RULES["1d"])
    elif "5m" in output:
        output["1d"] = resample_ohlcv(output["5m"], TIMEFRAME_RULES["1d"])

    if "1w" in frames:
        output["1w"] = frames["1w"]
    elif "1d" in output:
        output["1w"] = resample_ohlcv(output["1d"], TIMEFRAME_RULES["1w"])

    order = ["1w", "1d", "4h", "1h", "30m", "15m", "5m"]
    return {key: output[key] for key in order if key in output and not output[key].empty}
