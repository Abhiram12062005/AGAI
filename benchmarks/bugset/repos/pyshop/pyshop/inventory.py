"""Stock levels and reservations."""


class OutOfStock(Exception):
    pass


class Inventory:
    def __init__(self, stock: dict[str, int] | None = None):
        self._stock: dict[str, int] = dict(stock or {})
        self._reserved: dict[str, int] = {}

    def available(self, sku: str) -> int:
        return self._stock.get(sku, 0) - self._reserved.get(sku, 0)

    def reserve(self, sku: str, qty: int) -> None:
        """Reserve `qty` units. Reserving exactly the remaining quantity is allowed."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        if self.available(sku) >= qty:
            self._reserved[sku] = self._reserved.get(sku, 0) + qty
        else:
            raise OutOfStock(f"{sku}: requested {qty}, available {self.available(sku)}")

    def release(self, sku: str, qty: int) -> None:
        current = self._reserved.get(sku, 0)
        self._reserved[sku] = max(0, current - qty)

    def restock(self, sku: str, qty: int) -> None:
        if qty <= 0:
            raise ValueError("qty must be positive")
        self._stock[sku] = self._stock.get(sku, 0) + qty

    def snapshot(self) -> dict[str, int]:
        """Return a copy of available stock per sku; callers may mutate it freely."""
        return {sku: self.available(sku) for sku in self._stock}

    def low_stock(self, threshold: int = 3) -> list[str]:
        return sorted(sku for sku in self._stock if self.available(sku) <= threshold)
