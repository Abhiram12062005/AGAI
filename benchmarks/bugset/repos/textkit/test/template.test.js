'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { render, lookup } = require('../src/template');

test('render substitutes nested keys', () => {
  assert.equal(render('Hi {{ user.name }} from {{user.city}}', { user: { name: 'Ana', city: 'Pune' } }),
    'Hi Ana from Pune');
});

test('render keeps falsy-but-present values', () => {
  assert.equal(render('{{count}} items, done={{done}}', { count: 0, done: false }), '0 items, done=false');
});

test('render missing key: empty, or throws when strict', () => {
  assert.equal(render('a{{x}}b', {}), 'ab');
  assert.throws(() => render('{{x}}', {}, { strict: true }), /missing template key: x/);
});

test('lookup tolerates missing intermediate objects', () => {
  assert.equal(lookup({ a: null }, 'a.b.c'), undefined);
});
