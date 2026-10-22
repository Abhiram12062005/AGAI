"""Discounts, promo codes and tax."""

from datetime import date
from decimal import Decimal

from pyshop.money import percent_of, to_money

TAX_RATES = {"ca": Decimal("7.25"), "ny": Decimal("4.00"), "tx": Decimal("6.25"), "or": Decimal("0")}

PROMOS = {
    "SPRING10": {"pct": 10, "start": date(2026, 3, 1), "end": date(2026, 6, 1)},
    "WELCOME5": {"pct": 5, "start": date(2020, 1, 1), "end": date(2100, 1, 1)},
}


def normalize_region(region: str) -> str:
    return region.strip().lower()


def apply_discount(amount, pct) -> Decimal:
    """Reduce `amount` by `pct` percent."""
    if not 0 <= pct <= 100:
        raise ValueError("pct must be within 0..100")
    return to_money(Decimal(str(amount)) - percent_of(amount, pct))


def bulk_discount_pct(quantity: int) -> int:
    """Tiered bulk discount: 10+ units -> 5%, 50+ -> 10%, 100+ -> 15%."""
    if quantity >= 100:
        return 15
    if quantity >= 50:
        return 10
    if quantity >= 10:
        return 5
    return 0


def promo_pct(code: str | None, today: date) -> int:
    """Promo codes are valid from `start` (inclusive) to `end` (exclusive). Codes are case-insensitive."""
    if not code:
        return 0
    promo = PROMOS.get(code.strip().upper())
    if promo is None:
        return 0
    if promo["start"] <= today < promo["end"]:
        return promo["pct"]
    return 0


def tax_for(amount, region: str) -> Decimal:
    rate = TAX_RATES[normalize_region(region)]
    return percent_of(amount, rate)
