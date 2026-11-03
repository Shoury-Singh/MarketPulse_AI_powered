from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal


RiskProfile = Literal["conservative", "moderate", "aggressive"]


@dataclass
class PositionSizeResult:
    symbol: str
    direction: str
    account_size: float
    risk_percent: float
    risk_amount: float

    entry_price: float
    stop_loss: float
    risk_per_unit: float

    position_size: float          # in units (shares / coins)
    position_value: float         # in account currency
    max_loss: float

    # Optional helpful fields
    recommended_risk_percent: float
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def recommend_risk_percent(
    confidence: str,
    confluence_score: float,
    profile: RiskProfile = "moderate",
) -> float:
    """
    Suggest a sensible risk % based on setup quality and trader profile.

    This is a guideline, not a hard rule.
    """
    base = {
        "conservative": 0.5,
        "moderate": 1.0,
        "aggressive": 1.5,
    }[profile]

    if confluence_score >= 75 and confidence == "High":
        return round(base * 1.25, 2)
    if confluence_score >= 60 and confidence in ("High", "Medium"):
        return round(base, 2)
    if confluence_score >= 50:
        return round(base * 0.7, 2)

    return round(base * 0.4, 2)  # very small risk on weak setups


def calculate_position_size(
    symbol: str,
    direction: str,
    account_size: float,
    entry_price: float,
    stop_loss: float,
    risk_percent: float | None = None,
    confidence: str = "Medium",
    confluence_score: float = 60.0,
    profile: RiskProfile = "moderate",
    max_position_pct: float = 25.0,
) -> PositionSizeResult:
    """
    Professional position size calculator.

    Formula:
        Risk Amount = Account Size × (Risk % / 100)
        Risk per Unit = |Entry - Stop Loss|
        Position Size = Risk Amount / Risk per Unit

    Safety limits:
        - Position value cannot exceed max_position_pct of account
        - Risk % is auto-recommended if not provided
    """

    notes: list[str] = []

    if account_size <= 0:
        raise ValueError("Account size must be positive")
    if entry_price <= 0 or stop_loss <= 0:
        raise ValueError("Entry and Stop Loss must be positive")

    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit == 0:
        raise ValueError("Entry and Stop Loss cannot be the same")

    # Auto recommend risk % if not given
    recommended = recommend_risk_percent(confidence, confluence_score, profile)
    if risk_percent is None:
        risk_percent = recommended
        notes.append(f"Risk % auto-selected based on setup quality: {risk_percent}%")
    else:
        notes.append(f"Using user-defined risk: {risk_percent}% (recommended was {recommended}%)")

    risk_amount = account_size * (risk_percent / 100.0)
    position_size = risk_amount / risk_per_unit
    position_value = position_size * entry_price

    # Safety: cap position size so it doesn't become too large
    max_value = account_size * (max_position_pct / 100.0)
    if position_value > max_value:
        old_size = position_size
        position_size = max_value / entry_price
        position_value = max_value
        risk_amount = position_size * risk_per_unit
        notes.append(
            f"Position size reduced from {old_size:.4f} to {position_size:.4f} "
            f"to respect max {max_position_pct}% account exposure."
        )

    max_loss = position_size * risk_per_unit

    return PositionSizeResult(
        symbol=symbol,
        direction=direction,
        account_size=round(account_size, 2),
        risk_percent=round(risk_percent, 3),
        risk_amount=round(risk_amount, 2),
        entry_price=round(entry_price, 4),
        stop_loss=round(stop_loss, 4),
        risk_per_unit=round(risk_per_unit, 4),
        position_size=round(position_size, 6),
        position_value=round(position_value, 2),
        max_loss=round(max_loss, 2),
        recommended_risk_percent=recommended,
        notes=notes,
    )


def print_position_size(result: PositionSizeResult) -> None:
    print("=" * 60)
    print("MARKETPULSE AI — POSITION SIZE / RISK CALCULATOR")
    print("=" * 60)
    print(f"Symbol              : {result.symbol}")
    print(f"Direction           : {result.direction.upper()}")
    print(f"Account Size        : {result.account_size:,.2f}")
    print()
    print(f"Entry Price         : {result.entry_price}")
    print(f"Stop Loss           : {result.stop_loss}")
    print(f"Risk per Unit       : {result.risk_per_unit}")
    print()
    print(f"Risk %              : {result.risk_percent}%")
    print(f"Risk Amount         : {result.risk_amount:,.2f}")
    print(f"Position Size       : {result.position_size:,.4f} units")
    print(f"Position Value      : {result.position_value:,.2f}")
    print(f"Maximum Loss        : {result.max_loss:,.2f}")
    print()
    if result.notes:
        print("Notes:")
        for note in result.notes:
            print(f"  • {note}")
    print("=" * 60)
