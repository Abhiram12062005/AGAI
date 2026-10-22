'use strict';

/**
 * Convert a title to a URL slug: lowercase, ASCII letters/digits, words joined by single dashes.
 */
function slugify(title) {
  return title
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9\s-]/g, '')
    .replace(/[\s-]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

/**
 * Truncate text to at most `max` characters INCLUDING the ellipsis.
 */
function truncate(text, max, ellipsis = '...') {
  if (text.length <= max) return text;
  if (max <= ellipsis.length) return ellipsis.slice(0, max);
  return text.slice(0, max - ellipsis.length) + ellipsis;
}

/**
 * Count words separated by any amount of whitespace.
 */
function wordCount(text) {
  const trimmed = text.trim();
  if (trimmed === '') return 0;
  return trimmed.split(/\s+/).length;
}

/**
 * "hello big world" / "hello_big-world" -> "helloBigWorld"
 */
function camelCase(text) {
  const words = text.split(/[\s_-]+/).filter(Boolean);
  return words
    .map((word, i) => {
      const lower = word.toLowerCase();
      return i === 0 ? lower : lower[0].toUpperCase() + lower.slice(1);
    })
    .join('');
}

module.exports = { slugify, truncate, wordCount, camelCase };
