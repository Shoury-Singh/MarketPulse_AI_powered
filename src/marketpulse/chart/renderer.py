from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from marketpulse.chart.models import ChartState, ChartZone


def _filter_zones_for_display(
    zones: list[ChartZone],
    current_price: float,
    max_zones: int = 12,
    min_strength: float = 40.0,
    max_distance_pct: float = 0.035,
) -> list[ChartZone]:
    """
    Keep the chart clean:
    - Never draw mitigated zones
    - Prefer higher strength
    - Always keep ENTRY zone if present
    - Limit total rectangles so the chart does not become noisy
    """
    if not zones:
        return []

    entry_zones = [z for z in zones if z.zone_type == "ENTRY"]
    others = [
        z for z in zones
        if z.zone_type != "ENTRY"
        and not z.is_mitigated
        and z.strength >= min_strength
        and max(z.lower - current_price, current_price - z.upper, 0.0)
        / max(current_price, 1e-12) <= max_distance_pct
    ]

    # Prefer levels near current price, then strength, to preserve chart scale.
    others = sorted(
        others,
        key=lambda z: (
            min(abs(current_price - z.lower), abs(current_price - z.upper)),
            -z.strength,
        ),
    )

    # Limit non-entry zones
    remaining_slots = max(0, max_zones - len(entry_zones))
    selected = entry_zones + others[:remaining_slots]
    return selected


def create_plotly_figure(state: ChartState, height: int = 720) -> go.Figure:
    """
    Convert a ChartState into a clean, professional Plotly candlestick chart.
    Focus on readability: fewer overlapping zones, clear SL/TP, strong ENTRY highlight.
    """

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.80, 0.20],
    )

    if not state.candles:
        fig.add_annotation(
            text="No candle data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#94a3b8"),
        )
        fig.update_layout(
            template="plotly_dark",
            height=height,
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
        )
        return fig

    times = [c.time for c in state.candles]
    tick_stride = max(1, len(times) // 7)
    tick_indices = list(range(0, len(times), tick_stride))
    if tick_indices[-1] != len(times) - 1:
        tick_indices.append(len(times) - 1)
    tick_values = [times[index] for index in tick_indices]
    tick_format = "%b %d" if state.timeframe in ("1d", "1w") else "%b %d<br>%H:%M"
    tick_labels = [pd.Timestamp(value).strftime(tick_format) for value in tick_values]
    opens = [c.open for c in state.candles]
    highs = [c.high for c in state.candles]
    lows = [c.low for c in state.candles]
    closes = [c.close for c in state.candles]
    volumes = [c.volume for c in state.candles]

    last_price = closes[-1]

    # ---------- Candlesticks ----------
    fig.add_trace(
        go.Candlestick(
            x=times,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="Price",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
            increasing_fillcolor="#16a34a",
            decreasing_fillcolor="#dc2626",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ---------- Volume ----------
    vol_colors = ["#22c55e" if c >= o else "#ef4444" for o, c in zip(opens, closes)]
    fig.add_trace(
        go.Bar(
            x=times,
            y=volumes,
            marker_color=vol_colors,
            name="Volume",
            opacity=0.45,
            showlegend=False,
        ),
        row=2, col=1,
    )

    # ---------- Filtered Zones ----------
    display_zones = _filter_zones_for_display(
        state.zones, last_price, max_zones=8, min_strength=42.0
    )

    for zone in display_zones:
        is_entry = zone.zone_type == "ENTRY"

        fig.add_hrect(
            y0=zone.lower,
            y1=zone.upper,
            line_width=1.5 if is_entry else 1,
            line_color=zone.border_color,
            fillcolor=zone.color,
            opacity=0.95 if is_entry else 0.75,
            line_dash={"FVG": "dash", "SUPPLY": "dot", "DEMAND": "dot"}.get(
                zone.zone_type, "solid"
            ),
            row=1, col=1,
        )

        # Keep compact labels at the right edge; the color key identifies the layer.
        if is_entry or (zone.strength >= 65 and zone.zone_type in ("FVG", "ORDER_BLOCK")):
            mid = (zone.lower + zone.upper) / 2
            short_label = {
                "ORDER_BLOCK": "OB",
                "SUPPLY": "SUP",
                "DEMAND": "DEM",
            }.get(zone.zone_type, zone.zone_type)
            fig.add_annotation(
                x=times[-1],
                y=mid,
                text=short_label,
                showarrow=False,
                font=dict(size=9, color=zone.border_color),
                bgcolor="rgba(15, 23, 42, 0.88)",
                borderpad=2,
                xanchor="right",
                row=1, col=1,
            )

    # ---------- SL / TP lines (clean, right-side labels only) ----------
    for line in state.lines:
        dash_style = {"solid": "solid", "dashed": "dash", "dotted": "dot"}.get(
            line.line_type, "dash"
        )
        if line.id.startswith("structure_"):
            fig.add_trace(go.Scatter(
                x=[line.start_time or times[0], line.end_time or times[-1]],
                y=[line.price, line.price],
                mode="lines",
                line={"color": line.color, "dash": dash_style, "width": 1.5},
                name=line.label,
                hovertemplate=f"{line.label}<br>Level: %{{y:.5f}}<extra></extra>",
                showlegend=False,
            ), row=1, col=1)
            continue
        fig.add_hline(
            y=line.price,
            line_dash=dash_style,
            line_color=line.color,
            line_width=line.width + 0.5,
            row=1, col=1,
        )
        # Single clean label on the right
        fig.add_annotation(
            x=times[-1],
            y=line.price,
            text=f" {line.label} ",
            showarrow=False,
            xanchor="left",
            font=dict(size=11, color=line.color),
            bgcolor="rgba(15, 23, 42, 0.85)",
            borderpad=4,
            row=1, col=1,
        )

    for trendline in state.trendlines:
        dash_style = {"solid": "solid", "dashed": "dash", "dotted": "dot"}.get(
            trendline.line_type, "dash"
        )
        fig.add_trace(go.Scatter(
            x=[trendline.start_time, trendline.end_time],
            y=[trendline.start_price, trendline.end_price],
            mode="lines",
            line={"color": trendline.color, "dash": dash_style, "width": 2},
            name=trendline.label,
            hovertemplate=f"{trendline.label}<br>%{{y:.5f}}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

    # Signal names appear on hover; shape and color carry the category on the chart.
    marker_styles = {
        ("BOS", "bullish"): ("triangle-up", "#4ade80", "BOS ↑"),
        ("BOS", "bearish"): ("triangle-down", "#f87171", "BOS ↓"),
        ("CHOCH", "bullish"): ("circle", "#22d3ee", "CHoCH ↑"),
        ("CHOCH", "bearish"): ("circle", "#fb923c", "CHoCH ↓"),
        ("PATTERN", "bullish"): ("diamond", "#c084fc", "Pattern ↑"),
        ("PATTERN", "bearish"): ("diamond", "#e879f9", "Pattern ↓"),
    }
    for category, (symbol, color, display_name) in marker_styles.items():
        category_type, direction = category
        selected = []
        for marker in state.markers:
            if marker.direction != direction:
                continue
            marker_type = marker.label.split()[0].upper() if marker.kind == "structure" else "PATTERN"
            if marker_type == category_type:
                selected.append(marker)
        selected = sorted(selected, key=lambda marker: marker.time)[-5:]
        if selected:
            fig.add_trace(go.Scatter(
                x=[marker.time for marker in selected],
                y=[marker.price for marker in selected],
                mode="markers",
                marker={"symbol": symbol, "size": 8, "color": color,
                        "line": {"width": 1, "color": "#0f172a"}},
                name=display_name,
                hovertemplate="%{customdata}<br>%{y:.5f}<extra></extra>",
                customdata=[marker.label for marker in selected],
                showlegend=False,
            ), row=1, col=1)

    # ---------- Title ----------
    title = f"{state.symbol}   ·   {state.timeframe}"
    if state.bias_badge:
        title += f"     |     Bias: {state.bias_badge.bias.upper()}   ({state.bias_badge.score:.0f})"

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=17, color="#e2e8f0"),
            x=0.02,
            xanchor="left",
        ),
        template="plotly_dark",
        height=height,
        margin=dict(l=50, r=90, t=55, b=25),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#cbd5e1", size=12),
        showlegend=False,
        uirevision=f"{state.symbol}:{state.timeframe}",
    )

    # Clean grid
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=times,
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_labels,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.08)",
        zeroline=False,
        row=1, col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.08)",
        zeroline=False,
        side="right",
        row=1, col=1,
    )
    fig.update_xaxes(
        type="category", categoryorder="array", categoryarray=times,
        tickmode="array", tickvals=tick_values, ticktext=tick_labels,
        showgrid=False, row=2, col=1,
    )
    fig.update_yaxes(showgrid=False, side="right", row=2, col=1)

    # Current price line (subtle)
    fig.add_hline(
        y=last_price,
        line_dash="dot",
        line_color="rgba(148, 163, 184, 0.55)",
        line_width=1,
        row=1, col=1,
    )

    return fig


def bias_badge_markdown(state: ChartState) -> str:
    """Clean HTML card for the decision panel."""
    if not state.bias_badge:
        return "_No confluence data available._"

    b = state.bias_badge
    color = {
        "bullish": "#16a34a",
        "bearish": "#dc2626",
        "neutral": "#64748b",
    }.get(b.bias, "#64748b")

    return f"""
<div style="background:#1e293b; border-left:5px solid {color}; padding:16px 18px; border-radius:8px; margin-bottom:14px;">
  <div style="font-size:12px; color:#94a3b8; letter-spacing:0.5px; margin-bottom:4px;">HIGHER TIMEFRAME BIAS</div>
  <div style="font-size:22px; font-weight:700; color:{color}; text-transform:uppercase;">{b.bias}</div>
  <div style="margin-top:8px; font-size:14px; color:#e2e8f0;">
    Score: <b>{b.score:.1f}</b> &nbsp;·&nbsp; Confidence: <b>{b.confidence}</b>
  </div>
  <div style="margin-top:10px; font-size:13.5px; color:#cbd5e1; line-height:1.45;">{b.suggestion}</div>
</div>
"""
