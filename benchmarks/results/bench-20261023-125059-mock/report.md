# CodeFix AI benchmark — `bench-20261023-125059-mock`

> **MOCK RUN** — agents were played by the offline oracle mock. These numbers validate the harness and pipeline mechanics only; they are NOT model results.

24 runs · 4 bugs · architectures: fewshot·default, fewshot·senior, react-reflect·default, react-reflect·senior, react·default, react·senior

## Headline comparison

| Metric | fewshot·default | fewshot·senior | react-reflect·default | react-reflect·senior | react·default | react·senior |
|---|---|---|---|---|---|---|
| Resolved (fresh-clone full suite green) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) |
| Previously-failing test green | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| Judge verdict = pass | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) |
| Judge verdict = partial | 0 | 0 | 0 | 0 | 0 | 0 |
| **Judge accuracy** (agrees with tests) | 100% | 100% | 100% | 100% | 100% | 100% |
| Mean Fixer attempts | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| Runs needing a retry | 4 | 4 | 4 | 4 | 4 | 4 |
| Escalated to human | 0 | 0 | 0 | 0 | 0 | 0 |
| Errors | 0 | 0 | 0 | 0 | 0 | 0 |
| Mean wall-clock / run | 15.8s | 15.5s | 15.4s | 15.9s | 17.9s | 15.6s |
| Mean pipeline cost / run | $0.0000 | $0.0000 | $0.0000 | $0.0000 | $0.0000 | $0.0000 |
| Total pipeline cost | $0.000 | $0.000 | $0.000 | $0.000 | $0.000 | $0.000 |
| Mean LLM calls / run | 3.0 | 3.0 | 6.0 | 6.0 | 4.0 | 4.0 |
| Mean tokens / run | 5520 | 5520 | 11108 | 11108 | 6692 | 6694 |
| Mean envelope size (bytes) | 305 | 305 | 304 | 304 | 304 | 304 |

## Per-agent mean latency (seconds per run)

| Agent | fewshot·default | fewshot·senior | react-reflect·default | react-reflect·senior | react·default | react·senior |
|---|---|---|---|---|---|---|
| Diagnoser | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 | 0.1 |
| Fixer | 0.3 | 0.3 | 0.4 | 0.4 | 1.5 | 0.3 |
| Tester | 2.6 | 2.6 | 2.6 | 2.7 | 2.7 | 2.6 |
| Manager (LLM time) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Bottleneck agent per architecture: **fewshot·default** → tester, **fewshot·senior** → tester, **react-reflect·default** → tester, **react-reflect·senior** → tester, **react·default** → tester, **react·senior** → tester

## Failure-stage breakdown (non-passing runs)

_No failures._

## Judge vs objective tests (all architectures)

| Judge \ Tests | green | red |
|---|---|---|
| pass | 24 | 0 |

## Per-bug outcomes (✅ resolved · ❌ not resolved · attempts)

| Bug | difficulty | fewshot·default | fewshot·senior | react-reflect·default | react-reflect·senior | react·default | react·senior |
|---|---|---|---|---|---|---|---|
| js01-slugify-global | easy | ✅2 | ✅2 | ✅2 | ✅2 | ✅2 | ✅2 |
| py01-subtotal-skips-last | easy | ✅2 | ✅2 | ✅2 | ✅2 | ✅2 | ✅2 |
| py05-region-case | medium | ✅2 | ✅2 | ✅2 | ✅2 | ✅2 | ✅2 |
| py08-mutable-default | hard | ✅2 | ✅2 | ✅2 | ✅2 | ✅2 | ✅2 |

## By difficulty (resolved / runs)

| Difficulty | fewshot·default | fewshot·senior | react-reflect·default | react-reflect·senior | react·default | react·senior |
|---|---|---|---|---|---|---|
| easy | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| medium | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| hard | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
