from __future__ import annotations

import warnings
from datetime import UTC, datetime, timedelta

import pandas as pd

from marketpulse.data.normalize import normalize_ohlcv


class YahooFinanceProvider:
    """Historical equities provider backed by yfinance.

    Yahoo intraday lookback constraints are handled by clamping the requested
    start date and emitting a warning instead of silently returning no data.
    """

    def fetch(
        self,
        symbol: str,
        start: str,
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is not installed. Run: pip install -e .") from exc

        end_dt = self._parse_date(end) if end else datetime.now(UTC)
        start_dt = self._parse_date(start)
        now = datetime.now(UTC)
        end_dt = min(end_dt, now)

        # Yahoo intraday history is bounded to about 60 days. Fetch one small
        # native interval plus daily candles; larger intervals are resampled from
        # the 5m feed, reducing provider calls during dashboard refreshes.
        intraday_start = max(start_dt, end_dt - timedelta(days=59))
        requested = {
            "1d": ("1d", start_dt),
            "5m": ("5m", intraday_start),
        }
        output: dict[str, pd.DataFrame] = {}
        for label, (interval, effective_start) in requested.items():
            if effective_start > start_dt:
                warnings.warn(
                    f"Yahoo Finance limits intraday history. {symbol} {label} data starts at "
                    f"{effective_start.date()} instead of {start_dt.date()}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            try:
                raw = yf.Ticker(symbol).history(
                    start=effective_start,
                    end=end_dt,
                    interval=interval,
                    auto_adjust=True,
                    actions=False,
                    prepost=False,
                    raise_errors=True,
                )
            except Exception as exc:  # provider-specific errors should be visible to the CLI
                warnings.warn(f"Yahoo Finance failed for {symbol} {interval}: {exc}", RuntimeWarning)
                continue
            if raw is not None and not raw.empty:
                output[label] = normalize_ohlcv(raw)
        if not output:
            raise RuntimeError(
                f"Yahoo Finance returned no data for {symbol}. Check the ticker and requested dates."
            )
        return output

    @staticmethod
    def _parse_date(value: str | datetime) -> datetime:
        result = pd.Timestamp(value).to_pydatetime()
        if result.tzinfo is None:
            result = result.replace(tzinfo=UTC)
        return result.astimezone(UTC)
