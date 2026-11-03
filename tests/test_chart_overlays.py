import pandas as pd

from marketpulse.chart.builder import build_chart_state
from marketpulse.chart.models import ChartZone
from marketpulse.chart.renderer import _filter_zones_for_display, create_plotly_figure


def test_confirmed_swings_and_structure_breaks_render_as_lines():
    index = pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC")
    close = [100, 102, 101, 103, 102, 104, 103, 105]
    frame = pd.DataFrame({
        "open": [value - 0.2 for value in close],
        "high": [value + 1 for value in close],
        "low": [value - 1 for value in close],
        "close": close,
        "volume": [100.0] * len(close),
    }, index=index)
    swings = pd.DataFrame([
        {"type": "swing_low", "price": 98.0, "pivot_time": index[1], "pivot_index": 1},
        {"type": "swing_low", "price": 99.0, "pivot_time": index[3], "pivot_index": 3},
        {"type": "swing_high", "price": 103.0, "pivot_time": index[2], "pivot_index": 2},
        {"type": "swing_high", "price": 104.0, "pivot_time": index[4], "pivot_index": 4},
    ])
    breaks = pd.DataFrame([{
        "event": "CHoCH", "direction": "bearish", "level": 102.5,
        "time": index[6], "swing_time": index[3], "trend_before": "bullish",
    }])
    markers = pd.DataFrame([{
        "time": index[6], "price": 102.5, "label": "CHoCH bearish",
        "direction": "bearish", "kind": "structure",
    }])

    state = build_chart_state(
        symbol="DEMO",
        timeframe="1h",
        ohlcv=frame,
        swing_points=swings,
        structure_events=breaks,
        markers_df=markers,
    )
    figure = create_plotly_figure(state)

    assert len(state.trendlines) == 2
    assert state.lines[0].label == "CHoCH ↓ · 1h"
    assert any("Swing-low support" in trace.name for trace in figure.data)
    assert any("Swing-high resistance" in trace.name for trace in figure.data)
    signal = next(trace for trace in figure.data if trace.name == "CHoCH ↓")
    assert signal.mode == "markers"
    assert signal.text is None
    assert signal.marker.color == "#fb923c"
    assert figure.layout.xaxis.type == "category"


def test_faraway_zones_are_omitted_to_keep_price_scale_readable():
    zones = [
        ChartZone("near", "FVG", "bullish", 98.0, 99.0, strength=60),
        ChartZone("far", "ORDER_BLOCK", "bearish", 120.0, 122.0, strength=99),
    ]

    visible = _filter_zones_for_display(zones, current_price=100.0)

    assert [zone.id for zone in visible] == ["near"]
