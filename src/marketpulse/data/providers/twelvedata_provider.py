from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import ClassVar

import pandas as pd

from marketpulse.data.normalize import normalize_ohlcv


class TwelveDataProvider:
    """Online OHLC provider for stocks, forex/metals and other supported symbols."""

    endpoint = "https://api.twelvedata.com/time_series"
    intervals: ClassVar[dict[str, str]] = {"1d": "1day", "5m": "5min"}

    def __init__(self, api_key: str | None = None, timeout: int = 20):
        self.api_key = api_key or os.getenv("TWELVEDATA_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("Set TWELVEDATA_API_KEY to use the Twelve Data feed.")

    def fetch(self, symbol: str, start: str, end: str | None = None) -> dict[str, pd.DataFrame]:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests is required for Twelve Data. Install project requirements.") from exc

        output: dict[str, pd.DataFrame] = {}
        for label, interval in self.intervals.items():
            params = {
                "symbol": symbol,
                "interval": interval,
                "start_date": start,
                "end_date": end or datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                "order": "ASC",
                "timezone": "UTC",
                "outputsize": 5000,
                "apikey": self.api_key,
            }
            response = requests.get(self.endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") == "error" or "values" not in payload:
                message = payload.get("message", "No values returned")
                if label == "1d" and not output:
                    raise RuntimeError(f"Twelve Data error for {symbol} ({label}): {message}")
                continue
            frame = pd.DataFrame(payload["values"])
            if frame.empty:
                continue
            # FX and spot metals do not have centralized volume. Keep chart code's
            # OHLCV schema without suggesting that zero is traded volume.
            if "volume" not in frame.columns:
                frame["volume"] = 0.0
            else:
                frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
            frame = frame.rename(columns={"datetime": "timestamp"})
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame = frame.set_index("timestamp")
            output[label] = normalize_ohlcv(frame)

        if not output:
            raise RuntimeError(f"Twelve Data returned no OHLC data for {symbol}.")
        return output
