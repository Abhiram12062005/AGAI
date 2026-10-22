"""Token -> USD cost accounting using the pricing table in config/models.yaml."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


def cost_usd(model: str, usage: Usage, pricing: dict) -> float:
    price = pricing.get(model)
    if price is None:
        # unknown model (e.g. a fallback we have no row for): best-effort prefix match
        price = next((v for k, v in pricing.items() if model.startswith(k)), None)
    if price is None:
        return 0.0
    p = price if isinstance(price, dict) else price.model_dump()
    return (
        usage.input_tokens * p["input"]
        + usage.output_tokens * p["output"]
        + usage.cache_read_input_tokens * p.get("cache_read", 0.0)
        + usage.cache_creation_input_tokens * p.get("cache_write", 0.0)
    ) / 1_000_000
