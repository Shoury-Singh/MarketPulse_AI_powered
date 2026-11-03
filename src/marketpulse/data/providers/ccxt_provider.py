from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from marketpulse.data.normalize import normalize_ohlcv


class CCXTCryptoProvider:
    """Historical crypto OHLCV provider with explicit date-based pagination."""

    def __init__(self, exchange_id: str = "binance", limit: int = 1000, max_pages: int = 300):
        try:
            import ccxt
        except ImportError as exc:
            raise RuntimeError("ccxt is not installed. Run: pip install -e .") from exc
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"Unknown CCXT exchange: {exchange_id}")
        self.ccxt = ccxt
        self.exchange_id = exchange_id
        self.limit = limit
        self.max_pages = max_pages
        self.exchange = getattr(self.ccxt, exchange_id)({"enableRateLimit": True})

    def fetch(
        self,
        symbol: str,
        start: str,
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        self.exchange.load_markets()
        if symbol not in self.exchange.markets:
            raise ValueError(
                f"{symbol} is not listed on {self.exchange_id}. Check the exchange's symbol format."
            )
        if not self.exchange.has.get("fetchOHLCV"):
            raise RuntimeError(f"{self.exchange_id} does not expose fetchOHLCV through CCXT.")

        start_dt = self._parse_date(start)
        end_dt = self._parse_date(end) if end else datetime.now(timezone.utc)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        output: dict[str, pd.DataFrame] = {}
        recent_start_ms = max(start_ms, end_ms - 3 * 24 * 60 * 60 * 1000)
        for timeframe in ("1d", "30m", "15m", "5m"):
            if timeframe not in (self.exchange.timeframes or {}):
                continue
            # Preserve deeper 30m/daily context; only fetch a short recent window
            # for 5m/15m so live refreshes remain light enough for the dashboard.
            interval_start_ms = recent_start_ms if timeframe in ("5m", "15m") else start_ms
            rows = self._fetch_paginated(symbol, timeframe, interval_start_ms, end_ms)
            if rows:
                frame = pd.DataFrame(
                    rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
                frame = frame.set_index("timestamp")
                output[timeframe] = normalize_ohlcv(frame)
        if not output:
            raise RuntimeError(
                f"No OHLCV data returned for {symbol} on {self.exchange_id}. "
                "Check symbol, timeframe support, and date range."
            )
        return output

    def _fetch_paginated(
        self, symbol: str, timeframe: str, start_ms: int, end_ms: int
    ) -> list[list[float]]:
        rows: list[list[float]] = []
        cursor = start_ms
        duration_ms = self._timeframe_ms(timeframe)

        for _ in range(self.max_pages):
            if cursor >= end_ms:
                break
            batch = self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=cursor, limit=self.limit
            )
            if not batch:
                break
            in_range = [row for row in batch if start_ms <= row[0] < end_ms]
            rows.extend(in_range)
            last_timestamp = batch[-1][0]
            next_cursor = last_timestamp + duration_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < self.limit:
                break
        else:
            raise RuntimeError(
                f"Reached max_pages={self.max_pages} while downloading {symbol} {timeframe}. "
                "Increase max_pages or shorten the date range."
            )

        # Keep one candle per timestamp and preserve chronological order.
        unique = {int(row[0]): row for row in rows}
        return [unique[key] for key in sorted(unique)]

    @staticmethod
    def _timeframe_ms(timeframe: str) -> int:
        unit = timeframe[-1]
        amount = int(timeframe[:-1])
        multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
        if unit not in multipliers:
            raise ValueError(f"Unsupported timeframe unit: {timeframe}")
        return amount * multipliers[unit]

    @staticmethod
    def _parse_date(value: str | datetime) -> datetime:
        result = pd.Timestamp(value).to_pydatetime()
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result.astimezone(timezone.utc)
