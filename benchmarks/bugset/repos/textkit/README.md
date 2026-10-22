# textkit

Zero-dependency text utilities for Node.

- `slugify(title)` — lowercase ASCII slug, words joined by single dashes.
- `truncate(text, max, ellipsis='...')` — result length never exceeds `max` (ellipsis included).
- `wordCount(text)` — words separated by any whitespace.
- `camelCase(text)` — splits on spaces, `_` and `-`.
- `render(template, data, {strict})` — `{{ dotted.path }}` templates; falsy values like `0`/`false`
  render as-is, missing keys render empty (or throw when `strict`).

Tests: `npm test` (node:test, no install required).
