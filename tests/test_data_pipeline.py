import pandas as pd
import pytest

from marketpulse.data.normalize import normalize_ohlcv
from marketpulse.data.quality import inspect_ohlcv, validate_ohlcv
from marketpulse.data.timeframes import build_timeframes, resample_ohlcv


def sample_frame(periods: int = 120, freq: str = "30min") -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq=freq, tz="UTC")
    close = pd.Series(range(100, 100 + periods), index=index, dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1000.0,
        },
        index=index,
    )


def test_normalize_standardizes_columns_and_timezone():
    raw = sample_frame()
    normalized = normalize_ohlcv(raw)
    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
    assert str(normalized.index.tz) == "UTC"
    assert len(normalized) == len(raw)


def test_normalize_deduplicates_timestamps():
    raw = sample_frame()
    raw = pd.concat([raw, raw.iloc[[-1]]])
    normalized = normalize_ohlcv(raw)
    assert normalized.index.is_unique


def test_quality_report_accepts_valid_ohlcv():
    normalized = normalize_ohlcv(sample_frame())
    report = validate_ohlcv(normalized)
    assert report.is_valid
    assert report.rows == 120


def test_quality_report_finds_invalid_ohlc():
    normalized = normalize_ohlcv(sample_frame())
    normalized.iloc[0, normalized.columns.get_loc("high")] = 1.0
    report = inspect_ohlcv(normalized)
    assert report.invalid_ohlc_rows == 1
    with pytest.raises(ValueError, match="invalid OHLC"):
        validate_ohlcv(normalized)


def test_resample_ohlcv_uses_correct_aggregation():
    normalized = normalize_ohlcv(sample_frame(periods=4))
    hourly = resample_ohlcv(normalized, "1h")
    assert len(hourly) == 2
    assert hourly.iloc[0]["open"] == normalized.iloc[0]["open"]
    assert hourly.iloc[0]["close"] == normalized.iloc[1]["close"]
    assert hourly.iloc[0]["high"] == normalized.iloc[:2]["high"].max()
    assert hourly.iloc[0]["volume"] == 2000


def test_30m_csv_can_build_all_larger_timeframes():
    normalized = normalize_ohlcv(sample_frame(periods=24 * 14))
    frames = build_timeframes({"30m": normalized}, asset_class="crypto")
    assert set(frames) == {"1w", "1d", "4h", "1h", "30m"}
    assert len(frames["1h"]) > 0
    assert len(frames["4h"]) > 0


def test_5m_data_builds_real_15m_and_30m_candles():
    normalized = normalize_ohlcv(sample_frame(periods=120, freq="5min"))
    frames = build_timeframes({"5m": normalized}, asset_class="gold")
    assert {"5m", "15m", "30m"}.issubset(frames)
    assert len(frames["15m"]) == 40
    assert len(frames["30m"]) == 20
    assert frames["15m"].iloc[0]["open"] == normalized.iloc[0]["open"]
    assert frames["15m"].iloc[0]["close"] == normalized.iloc[2]["close"]


def test_normalize_parses_timestamp_string_column():
    raw = sample_frame(periods=3).reset_index(names="timestamp")
    raw["timestamp"] = raw["timestamp"].astype(str)
    normalized = normalize_ohlcv(raw)
    assert isinstance(normalized.index, pd.DatetimeIndex)
    assert str(normalized.index.tz) == "UTC"
    assert len(normalized) == 3
