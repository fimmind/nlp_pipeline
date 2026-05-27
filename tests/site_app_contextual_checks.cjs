const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function createElementStub() {
  return {
    value: '',
    textContent: '',
    innerHTML: '',
    files: [],
    classList: {
      add() {},
      remove() {},
      contains() { return false; },
    },
    addEventListener() {},
    querySelectorAll() { return []; },
    setAttribute() {},
  };
}

function loadAppModule() {
  const appPath = path.resolve(__dirname, '..', 'vocab_test_site', 'app.js');
  let source = fs.readFileSync(appPath, 'utf8');
  source = source.replace(/main\(\)\.catch\(\(err\) => \{[\s\S]*?\}\);\s*$/, '');
  source += '\nmodule.exports = { state, normalizeWord, isProperNounTag, isNameLikeToken, buildHighConfidenceProperNounLexicon, contextualDeinflectTaggedTerms, makeLemmaCandidates };\n';

  const storage = new Map();
  const context = {
    module: { exports: {} },
    exports: {},
    console,
    document: {
      getElementById: () => createElementStub(),
    },
    localStorage: {
      getItem: (key) => (storage.has(key) ? storage.get(key) : null),
      setItem: (key, value) => {
        storage.set(key, String(value));
      },
    },
    fetch: async () => ({
      ok: false,
      status: 404,
      json: async () => ({}),
      text: async () => '',
    }),
    setTimeout: (fn) => {
      fn();
      return 0;
    },
    clearTimeout: () => {},
    Math,
    Date,
    JSON,
    Array,
    String,
    Number,
    Boolean,
    Object,
    RegExp,
    Map,
    Set,
    window: {
      confirm: () => true,
    },
  };

  vm.runInNewContext(source, context, { filename: appPath });
  return context.module.exports;
}

(function testProperNounTaggingAndExclusions() {
  const app = loadAppModule();

  assert.equal(app.isProperNounTag(new Set(['ProperNoun']), 'Dent', 1), true);
  assert.equal(app.isProperNounTag(new Set(), 'Dent', 1), true);
  assert.equal(app.isProperNounTag(new Set(), 'Monday', 1), false);
  assert.equal(app.isProperNounTag(new Set(), 'The', 0), false);

  assert.equal(app.isNameLikeToken('Dent'), true);
  assert.equal(app.isNameLikeToken('MONDAY'), false);
  assert.equal(app.isNameLikeToken('Monday'), false);
})();

(function testHighConfidenceProperNounLexicon() {
  const app = loadAppModule();
  const taggedSentences = [
    [
      { raw: 'Arthur', normalized: 'arthur', tags: new Set(['ProperNoun']) },
      { raw: 'Dent', normalized: 'dent', tags: new Set(['ProperNoun']) },
    ],
    [
      { raw: 'He', normalized: 'he', tags: new Set() },
      { raw: 'met', normalized: 'met', tags: new Set() },
      { raw: 'Dent', normalized: 'dent', tags: new Set(['ProperNoun']) },
    ],
    [
      { raw: 'Again', normalized: 'again', tags: new Set() },
      { raw: 'Dent', normalized: 'dent', tags: new Set(['ProperNoun']) },
    ],
    [
      { raw: 'Monday', normalized: 'monday', tags: new Set(['ProperNoun']) },
      { raw: 'arrived', normalized: 'arrived', tags: new Set() },
    ],
    [
      { raw: 'Monday', normalized: 'monday', tags: new Set(['ProperNoun']) },
      { raw: 'left', normalized: 'left', tags: new Set() },
    ],
  ];

  const lexicon = app.buildHighConfidenceProperNounLexicon(taggedSentences);
  assert.equal(lexicon.has('dent'), true);
  assert.equal(lexicon.has('monday'), false);
})();

(function testContextualDeinflectionAndProperExclusion() {
  const app = loadAppModule();
  app.state.lemmaDict = {
    running: 'run',
    dogs: 'dog',
  };

  const taggedTerms = [
    { raw: 'running', normalized: 'running', tags: new Set(['Verb']) },
    { raw: 'dogs', normalized: 'dogs', tags: new Set(['Noun']) },
    { raw: 'Dent', normalized: 'dent', tags: new Set(['ProperNoun']) },
  ];
  const lowerToIdx = new Map([
    ['run', 1],
    ['dog', 2],
    ['dent', 3],
  ]);
  const properNounLexicon = new Set(['dent']);

  const result = app.contextualDeinflectTaggedTerms(
    taggedTerms,
    lowerToIdx,
    true,
    properNounLexicon,
  );

  assert.deepStrictEqual(Array.from(result.tokens), ['run', 'dog', '']);
  assert.deepStrictEqual(Array.from(result.properFlags), [false, false, true]);
})();
