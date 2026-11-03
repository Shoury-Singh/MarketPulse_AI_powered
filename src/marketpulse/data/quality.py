from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class QualityReport:
    rows: int
    duplicate_timestamps: int
    missing_values: int
    invalid_ohlc_rows: int
    negative_volume_rows: int
    non_monotonic_index: bool
    first_timestamp: str | None
    last_timestamp: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def is_valid(self) -> bool:
        return (
            self.rows > 0
            and self.missing_values == 0
            and self.invalid_ohlc_rows == 0
            and self.negative_volume_rows == 0
            and self.duplicate_timestamps == 0
            and not self.non_monotonic_index
        )


def inspect_ohlcv(frame: pd.DataFrame) -> QualityReport:
    required = ["open", "high", "low", "close", "volume"]
    missing_values = int(frame[required].isna().sum().sum()) if all(c in frame for c in required) else -1
    duplicated = int(frame.index.duplicated(keep=False).sum())
    non_monotonic = not frame.index.is_monotonic_increasing

    if all(column in frame.columns for column in required):
        invalid = (
            (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
            | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
            | (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
        )
        invalid_rows = int(invalid.sum())
        negative_volume = int((frame["volume"] < 0).sum())
    else:
        invalid_rows = -1
        negative_volume = -1

    return QualityReport(
        rows=len(frame),
        duplicate_timestamps=duplicated,
        missing_values=missing_values,
        invalid_ohlc_rows=invalid_rows,
        negative_volume_rows=negative_volume,
        non_monotonic_index=non_monotonic,
        first_timestamp=frame.index[0].isoformat() if len(frame) else None,
        last_timestamp=frame.index[-1].isoformat() if len(frame) else None,
    )


def validate_ohlcv(frame: pd.DataFrame) -> QualityReport:
    report = inspect_ohlcv(frame)
    issues = []
    if report.rows == 0:
        issues.append("no rows")
    if report.missing_values != 0:
        issues.append(f"missing values={report.missing_values}")
    if report.invalid_ohlc_rows != 0:
        issues.append(f"invalid OHLC rows={report.invalid_ohlc_rows}")
    if report.negative_volume_rows != 0:
        issues.append(f"negative volume rows={report.negative_volume_rows}")
    if report.non_monotonic_index:
        issues.append("timestamps are not sorted")
    if report.duplicate_timestamps:
        issues.append(f"duplicate timestamps={report.duplicate_timestamps}")
    if issues:
        raise ValueError("OHLCV validation failed: " + "; ".join(issues))
    return report
