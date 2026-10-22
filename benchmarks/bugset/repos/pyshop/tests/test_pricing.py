from datetime import date
from decimal import Decimal

import pytest

from pyshop.pricing import apply_discount, bulk_discount_pct, promo_pct, tax_for


def test_apply_discount():
    assert apply_discount(Decimal("100.00"), 15) == Decimal("85.00")
    assert apply_discount(Decimal("19.99"), 0) == Decimal("19.99")


def test_apply_discount_rejects_out_of_range():
    with pytest.raises(ValueError):
        apply_discount(10, 120)


@pytest.mark.parametrize("qty,pct", [(1, 0), (9, 0), (10, 5), (49, 5), (50, 10), (99, 10), (100, 15), (500, 15)])
def test_bulk_discount_tiers(qty, pct):
    assert bulk_discount_pct(qty) == pct


def test_promo_window_start_inclusive_end_exclusive():
    assert promo_pct("SPRING10", date(2026, 3, 1)) == 10
    assert promo_pct("SPRING10", date(2026, 5, 31)) == 10
    assert promo_pct("SPRING10", date(2026, 6, 1)) == 0
    assert promo_pct("SPRING10", date(2026, 2, 28)) == 0


def test_promo_case_insensitive_and_unknown():
    assert promo_pct(" spring10 ", date(2026, 4, 1)) == 10
    assert promo_pct("NOPE", date(2026, 4, 1)) == 0
    assert promo_pct(None, date(2026, 4, 1)) == 0


def test_tax_region_case_insensitive():
    assert tax_for(Decimal("100.00"), "CA") == Decimal("7.25")
    assert tax_for(Decimal("100.00"), " ny ") == Decimal("4.00")
    assert tax_for(Decimal("100.00"), "OR") == Decimal("0.00")
