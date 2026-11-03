import pandas as pd

from marketpulse.smc.waiting import assess_waiting_state


def test_waiting_estimate_uses_directional_momentum_and_timeframe():
    close = [100 + index * 0.5 for index in range(30)]
    frame = pd.DataFrame({
        "open": [value - 0.1 for value in close],
        "high": [value + 0.5 for value in close],
        "low": [value - 0.5 for value in close],
        "close": close,
        "volume": [100.0] * len(close),
    }, index=pd.date_range("2026-01-01", periods=30, freq="5min", tz="UTC"))
    swings = pd.DataFrame([{"type": "swing_high", "price": 120.0}])

    result = assess_waiting_state(
        frame=frame,
        timeframe="5m",
        bias="bullish",
        confluence_score=48,
        minimum_confluence=55,
        zone_score=0,
        confirmation_found=False,
        setup_ready=False,
        swings=swings,
    )

    assert result["reference"] == "Next confirmed swing high at 120.00000"
    assert "candles" in result["timing"]
    assert "not a historical average or promise" in result["timing_method"]
    assert any(not check["met"] for check in result["checks"])


def test_waiting_eta_is_withheld_when_momentum_moves_away():
    close = [100 + index * 0.5 for index in range(20)]
    close = list(reversed(close))
    frame = pd.DataFrame({
        "open": close,
        "high": [value + 0.5 for value in close],
        "low": [value - 0.5 for value in close],
        "close": close,
        "volume": [100.0] * len(close),
    }, index=pd.date_range("2026-01-01", periods=20, freq="15min", tz="UTC"))
    swings = pd.DataFrame([{"type": "swing_high", "price": 120.0}])

    result = assess_waiting_state(
        frame=frame,
        timeframe="15m",
        bias="bullish",
        confluence_score=40,
        minimum_confluence=55,
        zone_score=20,
        confirmation_found=False,
        setup_ready=False,
        swings=swings,
    )

    assert "points away" in result["timing"]
