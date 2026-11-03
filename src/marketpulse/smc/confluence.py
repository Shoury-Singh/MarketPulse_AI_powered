from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

from marketpulse.smc.structure import confirmed_swings, detect_structure_breaks
from marketpulse.smc.zones import (
    detect_fair_value_gaps,
    detect_order_blocks,
    enrich_supply_demand_zones,
    get_active_zones,
)

Bias = Literal["bullish", "bearish", "neutral"]


@dataclass
class ConfluenceResult:
    symbol: str
    bias: Bias
    confluence_score: float
    confidence: Literal["Low", "Medium", "High"]
    suggestion: str
    reason: str
    htf_details: dict
    zone_details: dict
    entry_details: dict
    active_zones_nearby: list

    def to_dict(self) -> dict:
        return asdict(self)


def _get_latest_trend(structure: pd.DataFrame) -> Bias:
    """Return the most recent structure trend from BOS/CHoCH events."""
    if structure.empty:
        return "neutral"
    last = structure.iloc[-1]
    return str(last["direction"])  # type: ignore


def get_htf_bias(
    weekly_frame: pd.DataFrame | None,
    daily_frame: pd.DataFrame | None,
    swing_left: int = 3,
    swing_right: int = 3,
) -> dict:
    """
    Determine higher-timeframe bias from Weekly + Daily structure.

    Rules:
    - Both bullish → strong bullish bias
    - Both bearish → strong bearish bias
    - Mixed or missing → neutral
    """
    result = {
        "weekly_trend": "neutral",
        "daily_trend": "neutral",
        "bias": "neutral",
        "bias_strength": 0.0,
        "aligned": False,
    }

    if weekly_frame is not None and len(weekly_frame) > 10:
        swings_w = confirmed_swings(weekly_frame, left=swing_left, right=swing_right)
        structure_w = detect_structure_breaks(weekly_frame, swings_w)
        result["weekly_trend"] = _get_latest_trend(structure_w)

    if daily_frame is not None and len(daily_frame) > 20:
        swings_d = confirmed_swings(daily_frame, left=swing_left, right=swing_right)
        structure_d = detect_structure_breaks(daily_frame, swings_d)
        result["daily_trend"] = _get_latest_trend(structure_d)

    w = result["weekly_trend"]
    d = result["daily_trend"]

    if w == d and w != "neutral":
        result["bias"] = w
        result["bias_strength"] = 85.0
        result["aligned"] = True
    elif w != "neutral" and d == "neutral":
        result["bias"] = w
        result["bias_strength"] = 55.0
    elif d != "neutral" and w == "neutral":
        result["bias"] = d
        result["bias_strength"] = 60.0
    elif w != d and w != "neutral" and d != "neutral":
        # Conflict → respect the more recent (Daily) but lower confidence
        result["bias"] = d
        result["bias_strength"] = 35.0
    else:
        result["bias"] = "neutral"
        result["bias_strength"] = 0.0

    return result


def score_zone_quality(
    price: float,
    active_zones: pd.DataFrame,
    bias: Bias,
    max_distance_pct: float = 0.025,
) -> dict:
    """
    Score how well current price is interacting with high-quality zones
    that align with the higher-timeframe bias.
    """
    result = {
        "best_zone_type": None,
        "best_zone_strength": 0.0,
        "distance_pct": None,
        "zone_score": 0.0,
        "aligned_zones": 0,
        "nearby_zones": [],
    }

    if active_zones.empty or bias == "neutral":
        return result

    # Only keep zones that match the bias direction
    if bias == "bullish":
        candidates = active_zones[active_zones["direction"] == "bullish"]
    else:
        candidates = active_zones[active_zones["direction"] == "bearish"]

    if candidates.empty:
        return result

    scored = []
    for _, zone in candidates.iterrows():
        lower = float(zone["lower"])
        upper = float(zone["upper"])
        mid = float(zone.get("mid", (lower + upper) / 2))
        strength = float(zone.get("strength", 40))

        # Distance from current price to the zone
        if lower <= price <= upper:
            distance_pct = 0.0  # price is inside the zone
        else:
            distance = min(abs(price - lower), abs(price - upper), abs(price - mid))
            distance_pct = distance / price

        if distance_pct > max_distance_pct:
            continue

        # Score: closer + stronger = better
        proximity_score = max(0.0, 100 - (distance_pct * 100 * 12))
        zone_score = (strength * 0.65) + (proximity_score * 0.35)

        scored.append({
            "zone_type": zone["zone_type"],
            "direction": zone["direction"],
            "lower": lower,
            "upper": upper,
            "strength": strength,
            "distance_pct": round(distance_pct * 100, 3),
            "zone_score": round(zone_score, 1),
        })

    if not scored:
        return result

    scored = sorted(scored, key=lambda x: x["zone_score"], reverse=True)
    best = scored[0]

    result["best_zone_type"] = best["zone_type"]
    result["best_zone_strength"] = best["strength"]
    result["distance_pct"] = best["distance_pct"]
    result["zone_score"] = best["zone_score"]
    result["aligned_zones"] = len(scored)
    result["nearby_zones"] = scored[:5]  # top 5 for display

    return result


def check_entry_confirmation(
    lower_tf_frame: pd.DataFrame,
    bias: Bias,
    lookback: int = 8,
) -> dict:
    """
    Look for recent entry confirmation on the lower timeframe
    (structure break or fresh FVG in the bias direction).
    """
    result = {
        "has_confirmation": False,
        "confirmation_type": None,
        "confirmation_score": 0.0,
        "details": None,
    }

    if lower_tf_frame is None or len(lower_tf_frame) < 15 or bias == "neutral":
        return result

    recent = lower_tf_frame.iloc[-lookback:]
    recent_start = recent.index[0]

    # 1. Recent structure break in bias direction
    swings = confirmed_swings(lower_tf_frame, left=2, right=2)
    structure = detect_structure_breaks(lower_tf_frame, swings)

    if not structure.empty:
        structure = structure[structure["time"] >= recent_start]
        last_events = structure.iloc[-3:]  # last few events
        matching = last_events[last_events["direction"] == bias]
        if not matching.empty:
            result["has_confirmation"] = True
            result["confirmation_type"] = str(matching.iloc[-1]["event"])
            result["confirmation_score"] = 75.0
            result["details"] = f"Recent {matching.iloc[-1]['event']} in {bias} direction"
            return result

    # 2. Fresh unmitigated FVG in bias direction near current price
    fvgs = detect_fair_value_gaps(lower_tf_frame)
    if not fvgs.empty:
        active_fvgs = get_active_zones(fvgs, min_strength=30.0)
        if not active_fvgs.empty:
            matching = active_fvgs[active_fvgs["direction"] == bias]
            if not matching.empty:
                # Prefer the most recent one
                latest = matching.sort_values("formed_at").iloc[-1]
                price = float(lower_tf_frame["close"].iloc[-1])
                if not float(latest["lower"]) * 0.985 <= price <= float(latest["upper"]) * 1.015:
                    return result
                result["has_confirmation"] = True
                result["confirmation_type"] = "FVG"
                result["confirmation_score"] = min(70.0, float(latest.get("strength", 40)))
                result["details"] = f"Fresh {bias} FVG (strength {latest.get('strength', 0)})"
                return result

    return result


def calculate_confluence(
    symbol: str,
    timeframes: dict[str, pd.DataFrame],
    current_price: float | None = None,
    swing_left: int = 3,
    swing_right: int = 3,
    entry_timeframe: str | None = None,
) -> ConfluenceResult:
    """
    Main confluence engine.

    Weights (from original plan):
    - Bias aligned          : 40%
    - Zone quality          : 25%
    - Entry confirmation    : 20%
    - Scores are based only on measurable rule evidence. The planned 15% model
      allocation is omitted until a trained and validated model exists.
    """

    weekly = timeframes.get("1w")
    daily = timeframes.get("1d")
    h4 = timeframes.get("4h")
    h1 = timeframes.get("1h")

    # ---------- 1. Higher Timeframe Bias ----------
    htf = get_htf_bias(weekly, daily, swing_left=swing_left, swing_right=swing_right)
    bias: Bias = htf["bias"]  # type: ignore
    bias_score = float(htf["bias_strength"])

    # ---------- 2. Current Price ----------
    if current_price is None:
        for tf in ["30m", "1h", "4h", "1d"]:
            frame = timeframes.get(tf)
            if frame is not None and not frame.empty:
                current_price = float(frame["close"].iloc[-1])
                break
    if current_price is None:
        current_price = 0.0

    # ---------- 3. Zone Quality (prefer 4H then 1H) ----------
    zone_frame = h4 if h4 is not None and not h4.empty else h1
    zone_timeframe = "4H" if zone_frame is h4 and h4 is not None else "1H"
    active_zones = pd.DataFrame()

    if zone_frame is not None and not zone_frame.empty:
        swings = confirmed_swings(zone_frame, left=swing_left, right=swing_right)
        structure = detect_structure_breaks(zone_frame, swings)
        fvgs = detect_fair_value_gaps(zone_frame)
        obs = detect_order_blocks(zone_frame, structure)
        sd = enrich_supply_demand_zones(zone_frame, swings)

        parts = [p for p in (fvgs, obs, sd) if not p.empty]
        all_zones = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        active_zones = get_active_zones(all_zones, min_strength=35.0)

    zone_info = score_zone_quality(current_price, active_zones, bias)
    for nearby_zone in zone_info.get("nearby_zones", []):
        nearby_zone["source_timeframe"] = zone_timeframe
    zone_score = float(zone_info["zone_score"])

    # ---------- 4. Entry Confirmation ----------
    entry_frame = timeframes.get(entry_timeframe) if entry_timeframe else None
    if entry_frame is None or entry_frame.empty:
        entry_frame = next(
            (timeframes.get(tf) for tf in ("5m", "15m", "30m", "1h")
             if timeframes.get(tf) is not None and not timeframes[tf].empty),
            None,
        )
    entry_info = check_entry_confirmation(entry_frame, bias)
    entry_score = float(entry_info["confirmation_score"])

    # ---------- 5. Final Confluence Score ----------
    # Weights from original plan
    raw_score = (
        (bias_score * 0.40)
        + (zone_score * 0.25)
        + (entry_score * 0.20)
    )
    final_score = max(0.0, min(100.0, raw_score / 0.85))

    # Confidence label
    if final_score >= 72:
        confidence = "High"
    elif final_score >= 52:
        confidence = "Medium"
    else:
        confidence = "Low"

    # ---------- Suggestion Engine ----------
    if bias == "neutral" or final_score < 45:
        suggestion = "No clear setup — wait"
        reason = "Higher timeframes are not aligned or confluence is too low."
    elif bias == "bullish":
        if final_score >= 70 and zone_info["zone_score"] > 50:
            suggestion = "Look for long entries near demand / bullish OB / FVG"
            reason = (
                f"Bullish HTF bias ({htf['weekly_trend']}/{htf['daily_trend']}), "
                f"price near quality {zone_info['best_zone_type']} zone "
                f"(strength {zone_info['best_zone_strength']})."
            )
        else:
            suggestion = "Bullish bias but wait for better location or confirmation"
            reason = "Bias is bullish but zone quality or entry confirmation is still weak."
    else:  # bearish
        if final_score >= 70 and zone_info["zone_score"] > 50:
            suggestion = "Look for short entries near supply / bearish OB / FVG"
            reason = (
                f"Bearish HTF bias ({htf['weekly_trend']}/{htf['daily_trend']}), "
                f"price near quality {zone_info['best_zone_type']} zone "
                f"(strength {zone_info['best_zone_strength']})."
            )
        else:
            suggestion = "Bearish bias but wait for better location or confirmation"
            reason = "Bias is bearish but zone quality or entry confirmation is still weak."

    return ConfluenceResult(
        symbol=symbol,
        bias=bias,
        confluence_score=round(final_score, 1),
        confidence=confidence,  # type: ignore
        suggestion=suggestion,
        reason=reason,
        htf_details=htf,
        zone_details=zone_info,
        entry_details=entry_info,
        active_zones_nearby=zone_info.get("nearby_zones", []),
    )
