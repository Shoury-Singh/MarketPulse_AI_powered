from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
COLUMN_ALIASES = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "close",
    "adj_close": "close",
    "volume": "volume",
}


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a UTC-indexed OHLCV frame with canonical lowercase columns.

    Accepted inputs can have a DatetimeIndex or a timestamp/date column.
    Additional provider columns are discarded deliberately.
    """
    if frame is None or frame.empty:
        raise ValueError("The provider returned no OHLCV rows.")

    df = frame.copy()

    # yfinance may return a MultiIndex column structure for some download modes.
    if isinstance(df.columns, pd.MultiIndex):
        flattened: list[str] = []
        for column in df.columns:
            parts = [str(part) for part in column if str(part) not in ("", "None")]
            flattened.append(parts[0] if parts else "")
        df.columns = flattened

    rename: dict[object, str] = {}
    for column in df.columns:
        normalized = str(column).strip().lower().replace("_", " ")
        if normalized in COLUMN_ALIASES:
            rename[column] = COLUMN_ALIASES[normalized]
    df = df.rename(columns=rename)

    if not isinstance(df.index, pd.DatetimeIndex):
        timestamp_col = next(
            (c for c in df.columns if str(c).strip().lower() in {"timestamp", "datetime", "date", "time"}),
            None,
        )
        if timestamp_col is None:
            raise ValueError("Input needs a DatetimeIndex or a timestamp/date column.")
        df = df.set_index(timestamp_col)
        # Parse CSV/string timestamps; utc=True also handles mixed timezone offsets.
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True, errors="coerce"))
    elif df.index.tz is None:
        # CSV timestamps without an offset are interpreted as UTC by convention.
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    df = df[REQUIRED_COLUMNS]
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=REQUIRED_COLUMNS)

    if df.empty:
        raise ValueError("No valid OHLCV rows remain after normalization.")

    return df.astype(float)


def filter_date_range(
    frame: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter a normalized UTC OHLCV frame; end is exclusive."""
    result = frame
    if start is not None:
        start_ts = pd.Timestamp(start)
        start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
        result = result.loc[result.index >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
        result = result.loc[result.index < end_ts]
    return result.copy()
