'use strict';

/**
 * Render "Hello {{ name }}" style templates. Missing keys render as an empty string
 * unless `strict` is set, in which case they throw.
 */
function render(template, data, { strict = false } = {}) {
  return template.replace(/\{\{\s*([\w.]+)\s*\}\}/g, (_, path) => {
    const value = lookup(data, path);
    if (value === undefined || value === null) {
      if (strict) throw new Error(`missing template key: ${path}`);
      return '';
    }
    return String(value);
  });
}

/** Resolve a dotted path like "user.address.city". */
function lookup(data, path) {
  return path.split('.').reduce((obj, key) => (obj == null ? undefined : obj[key]), data);
}

module.exports = { render, lookup };
