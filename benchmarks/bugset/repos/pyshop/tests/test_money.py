from decimal import Decimal

import pytest

from pyshop.money import percent_of, split_evenly, to_money


def test_to_money_rounds_half_up():
    assert to_money("2.345") == Decimal("2.35")
    assert to_money(2.344) == Decimal("2.34")
    assert to_money(10) == Decimal("10.00")


def test_percent_of():
    assert percent_of(Decimal("80.00"), 15) == Decimal("12.00")
    assert percent_of(Decimal("19.99"), Decimal("7.25")) == Decimal("1.45")


def test_split_evenly_sums_exactly():
    shares = split_evenly(Decimal("10.00"), 3)
    assert shares == [Decimal("3.34"), Decimal("3.33"), Decimal("3.33")]
    assert sum(shares) == Decimal("10.00")


def test_split_evenly_rejects_zero_parts():
    with pytest.raises(ValueError):
        split_evenly(Decimal("1"), 0)
