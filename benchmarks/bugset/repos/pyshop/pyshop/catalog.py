"""Product catalog."""

from dataclasses import dataclass, field
from decimal import Decimal

from pyshop.money import to_money


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    price: Decimal
    category: str = "general"
    tags: tuple = field(default_factory=tuple)


def normalize_sku(sku: str) -> str:
    """SKUs are case-insensitive and ignore surrounding whitespace and dashes."""
    return sku.strip().upper().replace("-", "")


class Catalog:
    def __init__(self):
        self._products: dict[str, Product] = {}

    def add(self, sku: str, name: str, price, category: str = "general", tags=()) -> Product:
        key = normalize_sku(sku)
        if key in self._products:
            raise ValueError(f"duplicate sku {sku}")
        product = Product(key, name, to_money(price), category, tuple(tags))
        self._products[key] = product
        return product

    def get(self, sku: str) -> Product:
        key = normalize_sku(sku)
        try:
            return self._products[key]
        except KeyError:
            raise KeyError(f"unknown sku {sku}") from None

    def search(self, text: str) -> list[Product]:
        """Case-insensitive substring search over name and tags, sorted by name."""
        needle = text.lower()
        hits = [p for p in self._products.values()
                if needle in p.name.lower() or any(needle in t.lower() for t in p.tags)]
        return sorted(hits, key=lambda p: p.name)

    def by_category(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for p in self._products.values():
            out.setdefault(p.category, []).append(p.sku)
        return {k: sorted(v) for k, v in out.items()}

    def __len__(self) -> int:
        return len(self._products)
