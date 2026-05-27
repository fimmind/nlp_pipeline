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
  const appPath = path.resolve(__dirname, '..', 'site', 'app.js');
  let source = fs.readFileSync(appPath, 'utf8');
  source = source.replace(/main\(\)\.catch\(\(err\) => \{[\s\S]*?\}\);\s*$/, '');
  source += '\nmodule.exports = { state, createRng, getQuizWordsAdaptiveUncertaintyLightRandom };\n';

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
      matchMedia: () => ({ matches: true }),
    },
  };

  vm.runInNewContext(source, context, { filename: appPath });
  return context.module.exports;
}

function main() {
  const seed = Number(process.argv[2] || 0);
  const quizSize = Number(process.argv[3] || 60);
  const observed = process.argv[4] ? JSON.parse(process.argv[4]) : {};

  const app = loadAppModule();
  const modelPath = path.resolve(__dirname, '..', 'site', 'data', 'best_grouped_irt_model_model_data.json');
  const model = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
  model.wordToIdx = new Map(model.words.map((word, idx) => [word, idx]));
  model.b = model.accuracy.map((value) => {
    const p = Math.min(1 - 1e-6, Math.max(1e-6, value == null ? 0.5 : value));
    return -Math.log(p / (1 - p));
  });

  app.state.model = model;
  app.state.activeProfile = {
    name: 'test_user',
    observed,
  };

  const rng = app.createRng(seed >>> 0);
  const words = app.getQuizWordsAdaptiveUncertaintyLightRandom(quizSize, rng);
  process.stdout.write(`${JSON.stringify(words)}\n`);
}

main();
