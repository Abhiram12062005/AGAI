from decimal import Decimal

import pytest

from pyshop.reports import parse_quantity, revenue_by_day, tag_orders, top_sellers

ORDERS = [
    {"id": "o1", "day": "2026-04-02", "total": "10.00", "lines": [("A", 2), ("B", 1)], "tags": ["gift"]},
    {"id": "o2", "day": "2026-04-01", "total": "5.50", "lines": [("B", 2), ("C", 3)]},
    {"id": "o3", "day": "2026-04-02", "total": "1.25", "lines": [("A", 1)], "tags": ["gift", "rush"]},
]


def test_top_sellers_ties_alphabetical():
    assert top_sellers(ORDERS) == [("A", 3), ("B", 3), ("C", 3)]
    assert top_sellers(ORDERS, n=1) == [("A", 3)]


def test_revenue_by_day_sorted():
    assert revenue_by_day(ORDERS) == {"2026-04-01": Decimal("5.50"), "2026-04-02": Decimal("11.25")}


def test_tag_orders_fresh_list_each_call():
    assert tag_orders(ORDERS, "gift") == ["o1", "o3"]
    assert tag_orders(ORDERS, "rush") == ["o3"]


def test_tag_orders_appends_to_existing():
    existing = ["o0"]
    assert tag_orders(ORDERS, "rush", existing) == ["o0", "o3"]


@pytest.mark.parametrize("text,qty", [("3", 3), (" 1,200 ", 1200), ("1_000", 1000)])
def test_parse_quantity(text, qty):
    assert parse_quantity(text) == qty


def test_parse_quantity_rejects_garbage():
    with pytest.raises(ValueError):
        parse_quantity("12a")
