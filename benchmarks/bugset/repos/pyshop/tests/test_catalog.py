from decimal import Decimal

import pytest

from pyshop.catalog import Catalog, normalize_sku


@pytest.fixture()
def catalog():
    c = Catalog()
    c.add("ab-100", "Blue Mug", "12.50", "kitchen", tags=("ceramic", "blue"))
    c.add("cd-200", "Red Mug", 11, "kitchen", tags=("ceramic",))
    c.add("ef-300", "Desk Lamp", "39.99", "office", tags=("led",))
    return c


def test_normalize_sku():
    assert normalize_sku("  ab-100 ") == "AB100"


def test_get_is_case_and_dash_insensitive(catalog):
    assert catalog.get("AB100").name == "Blue Mug"
    assert catalog.get(" ab-100").price == Decimal("12.50")


def test_unknown_sku(catalog):
    with pytest.raises(KeyError):
        catalog.get("zz-999")


def test_duplicate_rejected(catalog):
    with pytest.raises(ValueError):
        catalog.add("AB-100", "Other", 1)


def test_search_name_and_tags_sorted(catalog):
    assert [p.name for p in catalog.search("mug")] == ["Blue Mug", "Red Mug"]
    assert [p.name for p in catalog.search("LED")] == ["Desk Lamp"]


def test_by_category(catalog):
    assert catalog.by_category() == {"kitchen": ["AB100", "CD200"], "office": ["EF300"]}
