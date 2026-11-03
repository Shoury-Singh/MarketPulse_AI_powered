from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Color = str  # e.g. "#22c55e", "red", "rgba(...)"


@dataclass
class ChartCandle:
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class ChartZone:
    """Represents any rectangular zone that should be drawn on the chart."""
    id: str
    zone_type: str                  # FVG, ORDER_BLOCK, DEMAND, SUPPLY, ENTRY
    direction: str                  # bullish / bearish
    lower: float
    upper: float
    start_time: str | None = None
    end_time: str | None = None     # None = extends to the right (still active)
    strength: float = 50.0
    is_mitigated: bool = False
    label: str = ""
    color: Color = "rgba(34, 197, 94, 0.15)"
    border_color: Color = "#22c55e"


@dataclass
class ChartLine:
    """Horizontal or trend line (SL, TP, structure level, etc.)."""
    id: str
    price: float
    label: str
    line_type: Literal["solid", "dashed", "dotted"] = "dashed"
    color: Color = "#ef4444"
    width: int = 1
    start_time: str | None = None
    end_time: str | None = None


@dataclass
class ChartLabel:
    """Text annotation on the chart."""
    id: str
    text: str
    price: float
    time: str | None = None
    position: Literal["left", "right", "above", "below"] = "right"
    color: Color = "#ffffff"
    background: Color = "rgba(0,0,0,0.6)"


@dataclass
class ChartMarker:
    time: str
    price: float
    label: str
    direction: Literal["bullish", "bearish"]
    kind: str = "pattern"


@dataclass
class ChartTrendLine:
    start_time: str
    start_price: float
    end_time: str
    end_price: float
    label: str
    color: Color = "#38bdf8"
    line_type: Literal["solid", "dashed", "dotted"] = "dashed"


@dataclass
class ChartBiasBadge:
    """Top-level bias + confluence information."""
    bias: str
    score: float
    confidence: str
    suggestion: str
    color: Color = "#3b82f6"


@dataclass
class ChartState:
    """
    Complete state that a live chart needs to render.

    This is the single source of truth between the analysis engine
    and the frontend chart (Streamlit + Plotly / Lightweight Charts / etc.).
    """
    symbol: str
    timeframe: str
    candles: list[ChartCandle] = field(default_factory=list)

    # SMC layers
    zones: list[ChartZone] = field(default_factory=list)
    lines: list[ChartLine] = field(default_factory=list)
    labels: list[ChartLabel] = field(default_factory=list)
    markers: list[ChartMarker] = field(default_factory=list)
    trendlines: list[ChartTrendLine] = field(default_factory=list)

    # Decision layer
    bias_badge: ChartBiasBadge | None = None
    trade_plan_active: bool = False

    # Meta
    last_updated: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
