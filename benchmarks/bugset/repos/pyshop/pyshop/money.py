"""Money helpers. All amounts are Decimal and rounded to cents, half-up."""

from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def to_money(value) -> Decimal:
    """Convert int/float/str to a Decimal rounded to cents (half-up)."""
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def percent_of(amount: Decimal, pct) -> Decimal:
    """Return `pct` percent of `amount`, rounded to cents."""
    return to_money(Decimal(str(amount)) * Decimal(str(pct)) / Decimal(100))


def split_evenly(amount: Decimal, parts: int) -> list[Decimal]:
    """Split an amount into `parts` shares that sum exactly to the amount.

    The first shares absorb the leftover cents.
    """
    if parts <= 0:
        raise ValueError("parts must be positive")
    total_cents = int(to_money(amount) * 100)
    base, extra = divmod(total_cents, parts)
    return [Decimal(base + (1 if i < extra else 0)) / 100 for i in range(parts)]
