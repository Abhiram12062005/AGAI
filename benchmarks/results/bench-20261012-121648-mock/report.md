# CodeFix AI benchmark — `bench-20261012-121648-mock`

> **MOCK RUN** — agents were played by the offline oracle mock. These numbers validate the harness and pipeline mechanics only; they are NOT model results.

48 runs · 16 bugs · architectures: hierarchical, p2p, sequential

## Headline comparison

| Metric | hierarchical | p2p | sequential |
|---|---|---|---|
| Resolved (fresh-clone full suite green) | 16/16 (100%) | 16/16 (100%) | 16/16 (100%) |
| Previously-failing test green | 16/16 | 16/16 | 16/16 |
| Judge verdict = pass | 16/16 (100%) | 16/16 (100%) | 16/16 (100%) |
| Judge verdict = partial | 0 | 0 | 0 |
| **Judge accuracy** (agrees with tests) | 100% | 100% | 100% |
| Mean Fixer attempts | 2.00 | 2.00 | 2.00 |
| Runs needing a retry | 16 | 16 | 16 |
| Escalated to human | 0 | 0 | 0 |
| Errors | 0 | 0 | 0 |
| Mean wall-clock / run | 19.9s | 19.7s | 19.9s |
| Mean pipeline cost / run | $0.0000 | $0.0000 | $0.0000 |
| Total pipeline cost | $0.000 | $0.000 | $0.000 |
| Mean LLM calls / run | 4.0 | 4.0 | 6.0 |
| Mean tokens / run | 5557 | 5558 | 7732 |
| Mean envelope size (bytes) | 336 | 390 | 349 |

## Per-agent mean latency (seconds per run)

| Agent | hierarchical | p2p | sequential |
|---|---|---|---|
| Diagnoser | 0.2 | 0.2 | 0.4 |
| Fixer | 0.2 | 0.2 | 0.2 |
| Tester | 5.4 | 5.5 | 5.4 |
| Manager (LLM time) | 0.0 | 0.0 | 0.0 |

Bottleneck agent per architecture: **hierarchical** → tester, **p2p** → tester, **sequential** → tester

## Failure-stage breakdown (non-passing runs)

_No failures._

## Judge vs objective tests (all architectures)

| Judge \ Tests | green | red |
|---|---|---|
| pass | 48 | 0 |

## Per-bug outcomes (✅ resolved · ❌ not resolved · attempts)

| Bug | difficulty | hierarchical | p2p | sequential |
|---|---|---|---|---|
| js01-slugify-global | easy | ✅2 | ✅2 | ✅2 |
| js02-truncate-length | easy | ✅2 | ✅2 | ✅2 |
| js03-template-falsy | medium | ✅2 | ✅2 | ✅2 |
| js04-camelcase-index | medium | ✅2 | ✅2 | ✅2 |
| py01-subtotal-skips-last | easy | ✅2 | ✅2 | ✅2 |
| py02-discount-returns-discount | easy | ✅2 | ✅2 | ✅2 |
| py03-rounding-truncates | medium | ✅2 | ✅2 | ✅2 |
| py04-reserve-exact-stock | easy | ✅2 | ✅2 | ✅2 |
| py05-region-case | medium | ✅2 | ✅2 | ✅2 |
| py06-free-shipping-threshold | medium | ✅2 | ✅2 | ✅2 |
| py07-promo-end-inclusive | medium | ✅2 | ✅2 | ✅2 |
| py08-mutable-default | hard | ✅2 | ✅2 | ✅2 |
| py09-snapshot-aliasing | medium | ✅2 | ✅2 | ✅2 |
| py10-bulk-tier-boundary | easy | ✅2 | ✅2 | ✅2 |
| py11-sku-dashes | hard | ✅2 | ✅2 | ✅2 |
| py12-split-leftover | medium | ✅2 | ✅2 | ✅2 |

## By difficulty (resolved / runs)

| Difficulty | hierarchical | p2p | sequential |
|---|---|---|---|
| easy | 6/6 | 6/6 | 6/6 |
| medium | 8/8 | 8/8 | 8/8 |
| hard | 2/2 | 2/2 | 2/2 |
