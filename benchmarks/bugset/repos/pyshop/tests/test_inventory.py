import pytest

from pyshop.inventory import Inventory, OutOfStock


def test_reserve_reduces_available():
    inv = Inventory({"A": 5})
    inv.reserve("A", 2)
    assert inv.available("A") == 3


def test_reserve_exact_remaining_quantity():
    inv = Inventory({"A": 3})
    inv.reserve("A", 3)
    assert inv.available("A") == 0


def test_reserve_too_many_raises():
    inv = Inventory({"A": 1})
    with pytest.raises(OutOfStock):
        inv.reserve("A", 2)


def test_release_never_negative():
    inv = Inventory({"A": 2})
    inv.reserve("A", 1)
    inv.release("A", 5)
    assert inv.available("A") == 2


def test_snapshot_is_a_copy():
    inv = Inventory({"A": 2, "B": 9})
    snap = inv.snapshot()
    snap["A"] = 100
    assert inv.available("A") == 2


def test_low_stock():
    inv = Inventory({"A": 2, "B": 9, "C": 3})
    assert inv.low_stock() == ["A", "C"]


def test_restock():
    inv = Inventory()
    inv.restock("Z", 4)
    assert inv.available("Z") == 4
