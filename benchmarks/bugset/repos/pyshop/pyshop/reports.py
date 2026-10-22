"""Sales reporting helpers."""

from collections import Counter
from decimal import Decimal


def top_sellers(orders: list[dict], n: int = 3) -> list[tuple[str, int]]:
    """Most sold SKUs by quantity, ties broken alphabetically by sku."""
    counts: Counter = Counter()
    for order in orders:
        for sku, qty in order["lines"]:
            counts[sku] += qty
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def revenue_by_day(orders: list[dict]) -> dict[str, Decimal]:
    out: dict[str, Decimal] = {}
    for order in orders:
        out[order["day"]] = out.get(order["day"], Decimal("0")) + Decimal(str(order["total"]))
    return dict(sorted(out.items()))


def tag_orders(orders: list[dict], tag: str, existing: list | None = None) -> list[str]:
    """Return ids of orders tagged with `tag`, appended to `existing` if given (a new list otherwise)."""
    result = existing if existing is not None else []
    for order in orders:
        if tag in order.get("tags", ()):
            result.append(order["id"])
    return result


def parse_quantity(text: str) -> int:
    """Parse a human-entered quantity like ' 1,200 ' or '3'."""
    cleaned = text.strip().replace(",", "").replace("_", "")
    if not cleaned.isdigit():
        raise ValueError(f"not a quantity: {text!r}")
    return int(cleaned)
