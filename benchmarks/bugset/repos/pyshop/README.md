# pyshop

A tiny storefront library: catalog, inventory reservations, cart, pricing (bulk discounts,
promo codes, regional tax) and sales reports.

## Rules of the domain

- Money is `Decimal`, rounded to cents **half-up** (`pyshop.money.to_money`).
- SKUs are case-insensitive; dashes and surrounding whitespace are ignored.
- Bulk discount tiers: 10+ units 5%, 50+ units 10%, 100+ units 15% (per line item).
- Promo codes are case-insensitive and valid from `start` (inclusive) to `end` (exclusive).
- Free shipping when the *discounted* subtotal reaches $50.00; otherwise $5.99.
- Tax is charged on discounted subtotal + shipping; region codes are case-insensitive.
- Reserving exactly the remaining stock is allowed.

Run the tests with `python -m pytest`.
