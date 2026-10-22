'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { slugify, truncate, wordCount, camelCase } = require('../src/text');

test('slugify joins every word with dashes', () => {
  assert.equal(slugify('Hello Big  World!'), 'hello-big-world');
  assert.equal(slugify('  Already-slugged  title '), 'already-slugged-title');
});

test('truncate keeps total length within max including ellipsis', () => {
  assert.equal(truncate('abcdefghij', 8), 'abcde...');
  assert.equal(truncate('abcdefghij', 8).length, 8);
  assert.equal(truncate('short', 10), 'short');
  assert.equal(truncate('abcdef', 2), '..');
});

test('wordCount handles repeated whitespace', () => {
  assert.equal(wordCount('one  two   three'), 3);
  assert.equal(wordCount('   '), 0);
  assert.equal(wordCount('tab\tseparated\nlines'), 3);
});

test('camelCase', () => {
  assert.equal(camelCase('hello big world'), 'helloBigWorld');
  assert.equal(camelCase('Hello_big-WORLD'), 'helloBigWorld');
});
