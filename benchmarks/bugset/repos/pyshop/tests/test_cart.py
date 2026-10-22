from datetime import date
from decimal import Decimal

import pytest

from pyshop.cart import Cart, order_total
from pyshop.catalog import Catalog
from pyshop.inventory import Inventory, OutOfStock


@pytest.fixture()
def cart():
    catalog = Catalog()
    catalog.add("MUG-1", "Mug", "12.50")
    catalog.add("LAMP-1", "Lamp", "39.99")
    catalog.add("PEN-1", "Pen", "1.00")
    inv = Inventory({"MUG1": 20, "LAMP1": 2, "PEN1": 500})
    return Cart(catalog, inv)


def test_subtotal_sums_all_lines(cart):
    cart.add("mug-1", 2)
    cart.add("lamp-1", 1)
    cart.add("pen-1", 3)
    assert cart.subtotal() == Decimal("67.99")


def test_adding_same_sku_merges_lines(cart):
    cart.add("MUG-1", 1)
    cart.add("mug1", 2)
    assert len(cart.items) == 1 and cart.item_count() == 3


def test_bulk_discount_applied_per_line(cart):
    cart.add("PEN-1", 100)
    assert cart.subtotal() == Decimal("85.00")


def test_remove_releases_stock(cart):
    cart.add("LAMP-1", 2)
    cart.remove("lamp-1")
    assert cart.inventory.available("LAMP1") == 2
    assert cart.items == []


def test_cannot_overreserve(cart):
    with pytest.raises(OutOfStock):
        cart.add("LAMP-1", 3)


def test_order_total_free_shipping_uses_discounted_subtotal(cart):
    cart.add("MUG-1", 4)  # 50.00
    t = order_total(cart, "or", promo="SPRING10", today=date(2026, 4, 1))
    assert t["discounted"] == Decimal("45.00")
    assert t["shipping"] == Decimal("5.99")
    assert t["total"] == Decimal("50.99")


def test_order_total_with_tax(cart):
    cart.add("LAMP-1", 2)  # 79.98
    t = order_total(cart, "CA", today=date(2026, 7, 1))
    assert t["shipping"] == Decimal("0.00")
    assert t["tax"] == Decimal("5.80")
    assert t["total"] == Decimal("85.78")


def test_empty_cart_total(cart):
    t = order_total(cart, "tx", today=date(2026, 7, 1))
    assert t["total"] == Decimal("0.00")
