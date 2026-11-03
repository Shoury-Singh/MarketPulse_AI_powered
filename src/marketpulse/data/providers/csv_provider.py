from __future__ import annotations

from pathlib import Path

import pandas as pd

from marketpulse.data.normalize import filter_date_range, normalize_ohlcv


class CSVProvider:
    """Load a historical OHLCV CSV for reproducible research and testing."""

    def fetch(self, path: str | Path, start: str | None = None, end: str | None = None) -> dict[str, pd.DataFrame]:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        normalized = normalize_ohlcv(frame)
        filtered = filter_date_range(normalized, start, end)
        if filtered.empty:
            raise ValueError("CSV contains no candles inside the requested date range.")
        return {"30m": filtered}
