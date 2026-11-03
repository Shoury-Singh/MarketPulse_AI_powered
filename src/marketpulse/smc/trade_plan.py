from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import pandas as pd

from marketpulse.smc.confluence import ConfluenceResult


Direction = Literal["long", "short", "none"]
ZoneStatus = Literal["inside", "approaching", "extended", "invalid"]


@dataclass
class TradePlan:
    symbol: str
    direction: Direction
    bias: str
    confluence_score: float
    confidence: str

    # Core levels
    entry_low: float | None
    entry_high: float | None
    entry_mid: float | None
    stop_loss: float | None
    take_profit_1: float | None
    take_profit_2: float | None

    # Metrics
    risk_reward_1: float | None
    risk_reward_2: float | None
    risk_pct: float | None

    # Context
    entry_zone_type: str | None
    zone_status: ZoneStatus | None
    invalidation: str
    suggestion: str
    reasoning: str

    # For live chart labelling
    labels: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_valid(self) -> bool:
        return (
            self.direction != "none"
            and self.entry_mid is not None
            and self.stop_loss is not None
            and self.zone_status in ("inside", "approaching")
        )


def _classify_zone_status(
    direction: Direction,
    price: float,
    lower: float,
    upper: float,
    max_extension_pct: float = 0.012,
) -> ZoneStatus:
    """
    Decide whether the zone is still actionable.

    - inside      → price is within the zone (best)
    - approaching → price is close and moving toward the zone
    - extended    → price has already left the zone meaningfully
    - invalid     → zone is on the wrong side for the direction
    """
    mid = (lower + upper) / 2
    zone_height = upper - lower
    buffer = max(zone_height * 0.25, price * 0.004)  # small tolerance

    if direction == "long":
        # Ideal: price is at or slightly above the demand zone
        if lower - buffer <= price <= upper + buffer:
            return "inside"
        if price < lower - buffer:
            # Price still below zone → approaching from below
            distance = (lower - price) / price
            return "approaching" if distance <= max_extension_pct else "extended"
        # Price already well above the zone
        return "extended"

    else:  # short
        if lower - buffer <= price <= upper + buffer:
            return "inside"
        if price > upper + buffer:
            # Price still above zone → approaching from above
            distance = (price - upper) / price
            return "approaching" if distance <= max_extension_pct else "extended"
        return "extended"


def _find_opposing_target(
    direction: Direction,
    entry_mid: float,
    active_zones: list[dict],
) -> tuple[float | None, float | None]:
    """Find logical Take Profit levels from opposing zones."""
    tp1 = None
    tp2 = None

    if direction == "long":
        candidates = [
            z for z in active_zones
            if z.get("direction") == "bearish" and float(z["lower"]) > entry_mid * 1.002
        ]
        candidates = sorted(candidates, key=lambda z: float(z["lower"]))
        if candidates:
            tp1 = float(candidates[0]["lower"])
            if len(candidates) > 1:
                tp2 = float(candidates[1]["lower"])
    else:
        candidates = [
            z for z in active_zones
            if z.get("direction") == "bullish" and float(z["upper"]) < entry_mid * 0.998
        ]
        candidates = sorted(candidates, key=lambda z: float(z["upper"]), reverse=True)
        if candidates:
            tp1 = float(candidates[0]["upper"])
            if len(candidates) > 1:
                tp2 = float(candidates[1]["upper"])

    return tp1, tp2


def generate_trade_plan(
    confluence: ConfluenceResult,
    current_price: float,
    atr: float | None = None,
    min_confluence: float = 55.0,
    min_rr: float = 1.5,
    max_distance_pct: float = 0.018,
) -> TradePlan:
    """
    Generate a realistic, timing-aware trade plan.

    Key improvements:
    - Only accepts zones that are still actionable (inside or approaching)
    - Rejects zones that price has already left
    - Uses ATR buffer for Stop Loss when available
    - Produces clean labels for future live chart
    """

    def empty_plan(suggestion: str, reasoning: str) -> TradePlan:
        return TradePlan(
            symbol=confluence.symbol,
            direction="none",
            bias=confluence.bias,
            confluence_score=confluence.confluence_score,
            confidence=confluence.confidence,
            entry_low=None,
            entry_high=None,
            entry_mid=None,
            stop_loss=None,
            take_profit_1=None,
            take_profit_2=None,
            risk_reward_1=None,
            risk_reward_2=None,
            risk_pct=None,
            entry_zone_type=None,
            zone_status=None,
            invalidation="No valid setup",
            suggestion=suggestion,
            reasoning=reasoning,
            labels=[],
        )

    # 1. Basic filters
    if confluence.confluence_score < min_confluence or confluence.bias == "neutral":
        return empty_plan(
            "No clear setup — wait",
            "Confluence too low or no clear higher-timeframe bias.",
        )

    zone_info = confluence.zone_details
    if not zone_info or not zone_info.get("best_zone_type") or zone_info.get("zone_score", 0) < 40:
        return empty_plan(
            f"{confluence.bias.capitalize()} bias but no high-quality zone nearby",
            "Bias exists but price is not at a strong decision point.",
        )

    nearby = zone_info.get("nearby_zones") or []
    if not nearby:
        return empty_plan(
            "No actionable zone found near current price",
            "No high-quality zone is close enough to current price.",
        )

    direction: Direction = "long" if confluence.bias == "bullish" else "short"

    # 2. Select the best still-actionable zone
    best = None
    best_status: ZoneStatus = "invalid"

    for candidate in nearby:
        lower = float(candidate["lower"])
        upper = float(candidate["upper"])
        status = _classify_zone_status(direction, current_price, lower, upper)

        if status in ("inside", "approaching"):
            # Also check overall distance is reasonable
            mid = (lower + upper) / 2
            dist = abs(current_price - mid) / current_price
            if dist <= max_distance_pct:
                best = candidate
                best_status = status
                break

    if best is None:
        return empty_plan(
            f"{confluence.bias.capitalize()} bias exists but price has already left the best zones",
            "The high-quality zones aligned with bias are no longer near current price. Wait for a fresh location.",
        )

    entry_low = float(best["lower"])
    entry_high = float(best["upper"])
    entry_mid = (entry_low + entry_high) / 2
    entry_zone_type = best["zone_type"]

    # 3. Stop Loss with optional ATR buffer
    buffer = 0.0
    if atr is not None and atr > 0:
        buffer = atr * 0.12
    else:
        # Small percentage buffer if no ATR
        buffer = current_price * 0.0015

    if direction == "long":
        stop_loss = entry_low - buffer
        risk = entry_mid - stop_loss
    else:
        stop_loss = entry_high + buffer
        risk = stop_loss - entry_mid

    if risk <= 0:
        return empty_plan("Invalid risk calculation", "Stop loss placement produced zero or negative risk.")

    risk_pct = (risk / current_price) * 100

    # 4. Take Profit levels
    all_zones = confluence.active_zones_nearby or nearby
    tp1, tp2 = _find_opposing_target(direction, entry_mid, all_zones)

    # Fallback to fixed R-multiples
    if tp1 is None:
        tp1 = entry_mid + (risk * 1.8) if direction == "long" else entry_mid - (risk * 1.8)
    if tp2 is None:
        tp2 = entry_mid + (risk * 3.0) if direction == "long" else entry_mid - (risk * 3.0)

    rr1 = abs(tp1 - entry_mid) / risk
    rr2 = abs(tp2 - entry_mid) / risk

    # 5. Build messaging
    status_text = {
        "inside": "Price is currently inside the zone",
        "approaching": "Price is approaching the zone",
    }[best_status]

    if direction == "long":
        invalidation = f"4H/Daily close below {stop_loss:.4f}"
        suggestion = (
            f"Look for LONG entries in {entry_low:.2f} – {entry_high:.2f} "
            f"({entry_zone_type}). {status_text}."
        )
    else:
        invalidation = f"4H/Daily close above {stop_loss:.4f}"
        suggestion = (
            f"Look for SHORT entries in {entry_low:.2f} – {entry_high:.2f} "
            f"({entry_zone_type}). {status_text}."
        )

    reasoning = (
        f"{confluence.bias.capitalize()} HTF bias (score {confluence.confluence_score}). "
        f"{status_text} of a {entry_zone_type} (strength {best.get('strength', 0)})."
    )

    # 6. Labels for live chart
    labels = [
        {
            "type": "entry_zone",
            "direction": direction,
            "low": entry_low,
            "high": entry_high,
            "label": f"ENTRY ({entry_zone_type})",
            "color": "green" if direction == "long" else "red",
            "status": best_status,
        },
        {
            "type": "stop_loss",
            "price": stop_loss,
            "label": "SL",
            "color": "red",
        },
        {
            "type": "take_profit",
            "price": tp1,
            "label": f"TP1 ({rr1:.1f}R)",
            "color": "blue",
        },
        {
            "type": "take_profit",
            "price": tp2,
            "label": f"TP2 ({rr2:.1f}R)",
            "color": "blue",
        },
    ]

    return TradePlan(
        symbol=confluence.symbol,
        direction=direction,
        bias=confluence.bias,
        confluence_score=confluence.confluence_score,
        confidence=confluence.confidence,
        entry_low=round(entry_low, 4),
        entry_high=round(entry_high, 4),
        entry_mid=round(entry_mid, 4),
        stop_loss=round(stop_loss, 4),
        take_profit_1=round(tp1, 4),
        take_profit_2=round(tp2, 4),
        risk_reward_1=round(rr1, 2),
        risk_reward_2=round(rr2, 2),
        risk_pct=round(risk_pct, 3),
        entry_zone_type=entry_zone_type,
        zone_status=best_status,
        invalidation=invalidation,
        suggestion=suggestion,
        reasoning=reasoning,
        labels=labels,
    )


def print_trade_plan(plan: TradePlan) -> None:
    """Clean console printer for the trade plan."""
    print("=" * 62)
    print("MARKETPULSE AI — TRADE PLAN")
    print("=" * 62)
    print(f"Symbol            : {plan.symbol}")
    print(f"Direction         : {plan.direction.upper()}")
    print(f"Bias              : {plan.bias.upper()}")
    print(f"Confluence Score  : {plan.confluence_score} | Confidence: {plan.confidence}")
    print()

    if plan.is_valid:
        print(f"Zone Status       : {plan.zone_status.upper()}")
        print(f"Entry Zone        : {plan.entry_low}  →  {plan.entry_high}  ({plan.entry_zone_type})")
        print(f"Stop Loss         : {plan.stop_loss}")
        print(f"Take Profit 1     : {plan.take_profit_1}   (R:R = {plan.risk_reward_1})")
        print(f"Take Profit 2     : {plan.take_profit_2}   (R:R = {plan.risk_reward_2})")
        print(f"Risk % of price   : {plan.risk_pct}%")
        print()
        print(f"Invalidation      : {plan.invalidation}")
        print()
        print(f"Suggestion        : {plan.suggestion}")
        print(f"Reasoning         : {plan.reasoning}")
    else:
        print("No valid trade plan at the moment.")
        print(f"Suggestion        : {plan.suggestion}")
        print(f"Reasoning         : {plan.reasoning}")
    print("=" * 62)
