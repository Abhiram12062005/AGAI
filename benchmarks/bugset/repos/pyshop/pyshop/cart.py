"""Shopping cart and order totals."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from pyshop.catalog import Catalog
from pyshop.inventory import Inventory
from pyshop.money import to_money
from pyshop.pricing import apply_discount, bulk_discount_pct, promo_pct, tax_for

FREE_SHIPPING_THRESHOLD = Decimal("50.00")
SHIPPING_FEE = Decimal("5.99")


@dataclass
class LineItem:
    sku: str
    qty: int
    unit_price: Decimal

    @property
    def total(self) -> Decimal:
        gross = self.unit_price * self.qty
        return apply_discount(gross, bulk_discount_pct(self.qty))


class Cart:
    def __init__(self, catalog: Catalog, inventory: Inventory):
        self.catalog = catalog
        self.inventory = inventory
        self.items: list[LineItem] = []

    def add(self, sku: str, qty: int = 1) -> LineItem:
        product = self.catalog.get(sku)
        self.inventory.reserve(product.sku, qty)
        for item in self.items:
            if item.sku == product.sku:
                item.qty += qty
                return item
        item = LineItem(product.sku, qty, product.price)
        self.items.append(item)
        return item

    def remove(self, sku: str) -> None:
        key = self.catalog.get(sku).sku
        for item in list(self.items):
            if item.sku == key:
                self.inventory.release(key, item.qty)
                self.items.remove(item)

    def subtotal(self) -> Decimal:
        total = Decimal("0")
        for i in range(len(self.items)):
            total += self.items[i].total
        return to_money(total)

    def item_count(self) -> int:
        return sum(i.qty for i in self.items)


def order_total(cart: Cart, region: str, promo: str | None = None, today: date | None = None) -> dict:
    """Totals for checkout.

    Promo discount applies to the subtotal; shipping is free when the DISCOUNTED subtotal
    reaches the threshold; tax is charged on discounted subtotal + shipping.
    """
    today = today or date.today()
    subtotal = cart.subtotal()
    discounted = apply_discount(subtotal, promo_pct(promo, today))
    shipping = Decimal("0.00") if discounted >= FREE_SHIPPING_THRESHOLD or not cart.items else SHIPPING_FEE
    tax = tax_for(discounted + shipping, region)
    return {
        "subtotal": subtotal,
        "discounted": discounted,
        "shipping": shipping,
        "tax": tax,
        "total": to_money(discounted + shipping + tax),
    }
