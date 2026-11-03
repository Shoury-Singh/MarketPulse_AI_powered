import pandas as pd

from marketpulse.data.normalize import normalize_ohlcv
from marketpulse.smc.patterns import detect_candlestick_patterns
from marketpulse.smc.structure import confirmed_swings, detect_structure_breaks
from marketpulse.smc.zones import (
    detect_fair_value_gaps,
    detect_order_blocks,
    enrich_supply_demand_zones,
)


def make_ohlcv(rows):
    index = pd.date_range("2025-01-01", periods=len(rows), freq="1h", tz="UTC")
    return normalize_ohlcv(pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close", "volume"]))


def test_bullish_fvg_detection():
    frame = make_ohlcv([
        [10, 11, 9, 10, 100],
        [10, 12, 10, 11, 100],
        [12, 14, 12, 13, 100],
    ])
    zones = detect_fair_value_gaps(frame)
    assert len(zones) == 1
    assert zones.iloc[0]["direction"] == "bullish"
    assert zones.iloc[0]["lower"] == 11
    assert zones.iloc[0]["upper"] == 12


def test_swing_is_only_confirmed_after_right_side_bars():
    frame = make_ohlcv([
        [10, 11, 9, 10, 100],
        [10, 13, 10, 12, 100],
        [12, 12, 10, 11, 100],
        [11, 11, 9, 10, 100],
        [10, 11, 9, 10, 100],
    ])
    swings = confirmed_swings(frame, left=1, right=2)
    high = swings[swings["type"] == "swing_high"].iloc[0]
    assert high["pivot_index"] == 1
    assert high["confirmed_index"] == 3
    assert high["confirmed_at"] == frame.index[3]


def test_zone_detectors_return_expected_columns_without_events():
    frame = make_ohlcv([[10, 11, 9, 10, 100] for _ in range(8)])
    swings = confirmed_swings(frame, left=2, right=2)
    structure = detect_structure_breaks(frame, swings)
    order_blocks = detect_order_blocks(frame, structure)
    supply_demand = enrich_supply_demand_zones(frame, swings)
    assert "direction" in order_blocks.columns
    assert "zone_type" in supply_demand.columns


def test_engulfing_pattern_is_explained():
    frame = make_ohlcv([
        [12, 12.5, 9, 10, 100],
        [9.5, 13, 9, 12.5, 100],
    ])
    patterns = detect_candlestick_patterns(frame)
    assert patterns.iloc[-1]["pattern"] == "Bullish engulfing"
    assert patterns.iloc[-1]["direction"] == "bullish"
    assert "engulfs" in patterns.iloc[-1]["reason"]
