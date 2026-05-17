const STORAGE_KEY = "vocab_book_profiles_v3";
const THEME_KEY = "rasch_book_theme_v1";
const WORD_RE = /[A-Za-z]+(?:['’][A-Za-z]+)?/g;
const SENTENCE_RE = /[^.!?]+[.!?]+|[^.!?]+$/g;

const MODEL_DEFAULT = "best_adaptive";
const STRATEGY_DEFAULT = "static";
const DEFAULT_BOOK_NAME = "The Hitchhiker's Guide to the Galaxy";

const ui = {
  nicknameInput: document.getElementById("nicknameInput"),
  loadProfileBtn: document.getElementById("loadProfileBtn"),
  modelSelect: document.getElementById("modelSelect"),
  quizStrategy: document.getElementById("quizStrategy"),
  questionCount: document.getElementById("questionCount"),
  questionCountValue: document.getElementById("questionCountValue"),
  startBtn: document.getElementById("startBtn"),
  retakeBtn: document.getElementById("retakeBtn"),
  resetBtn: document.getElementById("resetBtn"),
  statusText: document.getElementById("statusText"),
  quizSection: document.getElementById("quizSection"),
  quizProgress: document.getElementById("quizProgress"),
  checklistWrap: document.getElementById("checklistWrap"),
  submitChecklistBtn: document.getElementById("submitChecklistBtn"),
  addWordsSection: document.getElementById("addWordsSection"),
  addWordsTextarea: document.getElementById("addWordsTextarea"),
  submitAddWordsBtn: document.getElementById("submitAddWordsBtn"),
  addWordsStatus: document.getElementById("addWordsStatus"),
  resultsSection: document.getElementById("resultsSection"),
  estimateStats: document.getElementById("estimateStats"),
  knownList: document.getElementById("knownList"),
  unknownList: document.getElementById("unknownList"),
  sentenceList: document.getElementById("sentenceList"),
  themeToggleBtn: document.getElementById("themeToggleBtn"),
  bookUpload: document.getElementById("bookUpload"),
  resetBookBtn: document.getElementById("resetBookBtn"),
  bookName: document.getElementById("bookName"),
  heroBookChip: document.getElementById("heroBookChip"),
};

const state = {
  model: null,
  bookText: "",
  profiles: loadProfilesStore(),
  currentNickname: "default",
  profile: { answers: {}, questionCount: 100, modelKey: MODEL_DEFAULT, strategy: STRATEGY_DEFAULT },
  quizWords: [],
  theme: "light",
  currentBatch: 0,
  batchSize: 10,
  isAdaptiveQuiz: false,
  quizSeed: 0,
  totalQuestionCount: 100,
};

function normalizeWord(token) {
  return token.toLowerCase().replaceAll("’", "'").replace(/^'+|'+$/g, "");
}

function safeNickname(raw) {
  const out = String(raw || "").trim().toLowerCase().replace(/[^a-z0-9_.-]+/g, "_");
  return out || "default";
}

function inferPosHint(prevToken, nextToken) {
  const determiners = new Set(["a", "an", "the", "this", "that", "these", "those", "my", "your", "our", "his", "her", "their"]);
  const beForms = new Set(["am", "is", "are", "was", "were", "be", "been", "being"]);
  const pronouns = new Set(["i", "you", "he", "she", "we", "they", "it"]);
  const auxiliaries = new Set(["do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might", "must"]);
  if (prevToken === "to") return "verb";
  if (beForms.has(prevToken)) return "verb_participle";
  if (determiners.has(prevToken)) return "noun";
  if (pronouns.has(prevToken) || auxiliaries.has(prevToken)) return "verb";
  if (nextToken === "of" || nextToken === "and" || nextToken === "or") return "noun";
  return "any";
}

function deinflectionCandidates(token) {
  const out = [[token, "identity"]];
  const add = (cand, tag) => { if (cand.length >= 2) out.push([cand, tag]); };
  if (token.endsWith("ies") && token.length > 4) add(token.slice(0, -3) + "y", "noun_plural");
  if (token.endsWith("es") && token.length > 3) add(token.slice(0, -2), "noun_plural");
  if (token.endsWith("s") && token.length > 3 && !token.endsWith("ss")) add(token.slice(0, -1), "noun_plural");
  if (token.endsWith("ied") && token.length > 4) add(token.slice(0, -3) + "y", "verb_past");
  if (token.endsWith("ed") && token.length > 3) { add(token.slice(0, -2), "verb_past"); add(token.slice(0, -1), "verb_past"); }
  if (token.endsWith("ing") && token.length > 5) { add(token.slice(0, -3), "verb_ing"); add(token.slice(0, -3) + "e", "verb_ing"); }
  if (token.endsWith("er") && token.length > 4) add(token.slice(0, -2), "adj_comp");
  if (token.endsWith("est") && token.length > 5) add(token.slice(0, -3), "adj_super");
  return out;
}

function pickContextualLemma(token, prevToken, nextToken, vocabSet) {
  const hint = inferPosHint(prevToken, nextToken);
  let best = token;
  let score = -1e9;
  for (const [cand, tag] of deinflectionCandidates(token)) {
    let s = 0;
    if (vocabSet.has(cand)) s += 100;
    if (tag === "identity") s += 5;
    if (hint === "noun" && tag.startsWith("noun_")) s += 4;
    if (hint === "verb" && tag.startsWith("verb_")) s += 4;
    if (hint === "verb_participle" && tag === "verb_ing") s += 4;
    s -= 0.1 * Math.abs(cand.length - token.length);
    if (s > score) { score = s; best = cand; }
  }
  return best;
}

function contextualDeinflect(rawTokens, vocabSet) {
  const normalized = rawTokens.map(normalizeWord);
  return normalized.map((token, i) => {
    const prev = i > 0 ? normalized[i - 1] : null;
    const next = i + 1 < normalized.length ? normalized[i + 1] : null;
    return pickContextualLemma(token, prev, next, vocabSet);
  });
}

function logit(p) { return Math.log(p / (1 - p)); }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function clip01(p) { return Math.min(1 - 1e-6, Math.max(1e-6, p)); }

function loadProfilesStore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : { currentNickname: "default", profiles: {} };
    return {
      currentNickname: safeNickname(parsed.currentNickname || "default"),
      profiles: parsed.profiles || {},
    };
  } catch {
    return { currentNickname: "default", profiles: {} };
  }
}

function loadTheme() {
  const raw = localStorage.getItem(THEME_KEY);
  if (raw === "dark" || raw === "light") return raw;
  return "light";
}

function saveTheme(theme) {
  localStorage.setItem(THEME_KEY, theme);
}

function applyTheme(theme) {
  state.theme = theme === "dark" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", state.theme);
  if (ui.themeToggleBtn) {
    ui.themeToggleBtn.textContent = state.theme === "dark" ? "Light mode" : "Dark mode";
  }
  saveTheme(state.theme);
}

function toggleTheme() {
  applyTheme(state.theme === "dark" ? "light" : "dark");
}

function saveProfilesStore() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    currentNickname: state.currentNickname,
    profiles: state.profiles.profiles,
  }));
}

function loadCurrentProfile() {
  const existing = state.profiles.profiles[state.currentNickname];
  state.profile = existing || {
    answers: {},
    questionCount: 100,
    modelKey: MODEL_DEFAULT,
    strategy: STRATEGY_DEFAULT,
  };
  state.profiles.profiles[state.currentNickname] = state.profile;
  state.profiles.currentNickname = state.currentNickname;
}

function saveCurrentProfile() {
  state.profiles.profiles[state.currentNickname] = state.profile;
  state.profiles.currentNickname = state.currentNickname;
  saveProfilesStore();
}

function initControls() {
  const q = Math.min(200, Math.max(10, Number(state.profile.questionCount) || 100));
  ui.questionCount.value = String(q);
  ui.questionCountValue.textContent = String(q);
  ui.modelSelect.value = state.profile.modelKey || MODEL_DEFAULT;
  ui.quizStrategy.value = state.profile.strategy || STRATEGY_DEFAULT;
}

async function loadData(modelKey) {
  const filename = modelKey === "rasch" ? "rasch_model_data.json" : `${modelKey}_model_data.json`;
  const modelRes = await fetch(`./data/${filename}`);
  if (!modelRes.ok) throw new Error(`Failed to load model data: ${modelRes.status}`);
  state.model = await modelRes.json();
  state.model.wordToIdx = new Map(state.model.words.map((w, i) => [w, i]));
  state.model.vocabSet = new Set(state.model.words);
  state.model.b = state.model.accuracy.map((a) => {
    const p = a == null ? 0.5 : clip01(a);
    return -logit(p);
  });
}

async function loadDefaultBook() {
  const bookRes = await fetch("./data/hitchhikers_guide.txt");
  if (!bookRes.ok) throw new Error(`Failed to load book: ${bookRes.status}`);
  state.bookText = await bookRes.text();
}

function setBookName(name) {
  const text = `Current: ${name}`;
  ui.bookName.textContent = text;
  if (ui.heroBookChip) {
    ui.heroBookChip.textContent = `Book: ${name}`;
  }
}

function handleBookUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) {
    ui.bookName.textContent = "Error: file too large (max 10 MB).";
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    state.bookText = e.target.result;
    setBookName(file.name);
    if (!ui.resultsSection.classList.contains("hidden")) {
      runEstimation();
    }
  };
  reader.onerror = () => {
    ui.bookName.textContent = "Error reading file.";
  };
  reader.readAsText(file);
}

async function resetBook() {
  setBookName("Loading default book...");
  try {
    await loadDefaultBook();
    setBookName(DEFAULT_BOOK_NAME);
    if (!ui.resultsSection.classList.contains("hidden")) {
      runEstimation();
    }
  } catch (err) {
    ui.bookName.textContent = `Error: ${err.message}`;
  }
}

function hashStringToSeed(input) {
  let h = 2166136261;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function createRng(seed) {
  let s = seed >>> 0;
  return function next() {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function weightedSampleWithoutReplacement(items, sampleCount, weightFn, rng) {
  const pool = [...items];
  const out = [];
  const n = Math.min(sampleCount, pool.length);
  for (let k = 0; k < n; k++) {
    let total = 0;
    const weights = [];
    for (const item of pool) {
      const w = Math.max(1e-9, weightFn(item));
      weights.push(w);
      total += w;
    }
    let target = rng() * total;
    let chosenIdx = 0;
    for (let i = 0; i < pool.length; i++) {
      target -= weights[i];
      if (target <= 0) {
        chosenIdx = i;
        break;
      }
    }
    out.push(pool[chosenIdx]);
    pool.splice(chosenIdx, 1);
  }
  return out;
}

function getCandidatePool() {
  const basePool = state.model.query_pool;
  if (!Array.isArray(basePool) || basePool.length === 0) return [];
  return basePool
    .map((word) => {
      const idx = state.model.wordToIdx.get(word);
      if (idx == null) return null;
      const acc = clip01(state.model.accuracy[idx] == null ? 0.5 : state.model.accuracy[idx]);
      return { word, idx, acc };
    })
    .filter(Boolean);
}

function getQuizWordsStatic(questionCount, rng) {
  const q = Math.max(10, Math.min(200, questionCount));
  const candidateInfo = getCandidatePool();
  if (candidateInfo.length <= q) return candidateInfo.map((x) => x.word);

  const bandCount = 5;
  const bands = Array.from({ length: bandCount }, () => []);
  for (const item of candidateInfo) {
    const b = Math.min(bandCount - 1, Math.floor(item.acc * bandCount));
    bands[b].push(item);
  }

  const picks = [];
  const perBand = Math.floor(q / bandCount);
  const remainder = q % bandCount;
  for (let b = 0; b < bandCount; b++) {
    const target = perBand + (b < remainder ? 1 : 0);
    if (bands[b].length === 0) continue;
    const sampled = weightedSampleWithoutReplacement(
      bands[b],
      target,
      (x) => 1 - Math.min(1, Math.abs(x.acc - 0.5) * 1.8),
      rng
    );
    picks.push(...sampled);
  }

  const used = new Set(picks.map((x) => x.word));
  if (picks.length < q) {
    const remaining = candidateInfo.filter((x) => !used.has(x.word));
    const extra = weightedSampleWithoutReplacement(
      remaining,
      q - picks.length,
      (x) => 1 - Math.min(1, Math.abs(x.acc - 0.5) * 1.8),
      rng
    );
    picks.push(...extra);
  }

  for (let i = picks.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = picks[i];
    picks[i] = picks[j];
    picks[j] = tmp;
  }

  return picks.slice(0, q).map((x) => x.word);
}

function getQuizWordsSemirandom(questionCount, rng) {
  const words = getQuizWordsStatic(questionCount, rng);
  // Small random swaps like CLI's semirandom for grouped IRT
  const arr = [...words];
  for (let i = 0; i < arr.length - 1; i++) {
    if (rng() < 0.05) {
      const j = i + 1 + Math.floor(rng() * Math.min(3, arr.length - i - 1));
      if (j < arr.length) {
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
    }
  }
  return arr;
}

function getQuizWordsAdaptiveUncertainty(questionCount, rng) {
  const q = Math.max(10, Math.min(200, questionCount));
  const candidateInfo = getCandidatePool();
  if (candidateInfo.length <= q) return candidateInfo.map((x) => x.word);

  const answered = new Set(Object.keys(state.profile.answers));
  let pool = candidateInfo.filter((x) => !answered.has(x.word));
  if (pool.length < q) pool = candidateInfo;

  const obsIds = [];
  const obsLabels = [];
  for (const w of Object.keys(state.profile.answers)) {
    const idx = state.model.wordToIdx.get(w);
    if (idx != null) {
      obsIds.push(idx);
      obsLabels.push(state.profile.answers[w]);
    }
  }
  const theta = estimateTheta(obsIds, obsLabels);

  // Score by posterior uncertainty = -|p_posterior - 0.5|
  const scored = pool.map((x) => {
    const p = predictProba(theta, x.idx);
    return { ...x, score: -Math.abs(p - 0.5) };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, q).map((x) => x.word);
}

function getQuizWordsAdaptiveUncertaintyLightRandom(questionCount, rng) {
  const q = Math.max(10, Math.min(200, questionCount));
  const candidateInfo = getCandidatePool();
  if (candidateInfo.length <= q) return candidateInfo.map((x) => x.word);

  const answered = new Set(Object.keys(state.profile.answers));
  let pool = candidateInfo.filter((x) => !answered.has(x.word));
  if (pool.length < q) pool = candidateInfo;

  const obsIds = [];
  const obsLabels = [];
  for (const w of Object.keys(state.profile.answers)) {
    const idx = state.model.wordToIdx.get(w);
    if (idx != null) {
      obsIds.push(idx);
      obsLabels.push(state.profile.answers[w]);
    }
  }
  const theta = estimateTheta(obsIds, obsLabels);

  const scored = pool.map((x) => {
    const p = predictProba(theta, x.idx);
    return { ...x, score: -Math.abs(p - 0.5) };
  });
  scored.sort((a, b) => b.score - a.score);

  const topK = 3;
  const temperature = 0.03;
  const out = [];
  const available = scored.map((x, idx) => ({ ...x, _poolIndex: idx }));

  for (let i = 0; i < Math.min(q, available.length); i++) {
    const candidates = available.slice(0, Math.min(topK, available.length));
    const maxScore = Math.max(...candidates.map((c) => c.score));
    const logits = candidates.map((c) => (c.score - maxScore) / temperature);
    const expLogits = logits.map((l) => Math.exp(Math.max(-60, Math.min(60, l))));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map((e) => e / sumExp);

    let r = rng();
    let chosenIdx = 0;
    for (let j = 0; j < probs.length; j++) {
      r -= probs[j];
      if (r <= 0) {
        chosenIdx = j;
        break;
      }
    }
    const removedPoolIndex = candidates[chosenIdx]._poolIndex;
    out.push(candidates[chosenIdx].word);
    available.splice(removedPoolIndex, 1);
    available.forEach((x, idx) => { x._poolIndex = idx; });
  }

  return out;
}

function getQuizWordsAdaptiveEerFast(questionCount, rng) {
  const q = Math.max(10, Math.min(200, questionCount));
  const candidateInfo = getCandidatePool();
  if (candidateInfo.length <= q) return candidateInfo.map((x) => x.word);

  const answered = new Set(Object.keys(state.profile.answers));
  let pool = candidateInfo.filter((x) => !answered.has(x.word));
  if (pool.length < q) pool = candidateInfo;

  const obsIds = [];
  const obsLabels = [];
  for (const w of Object.keys(state.profile.answers)) {
    const idx = state.model.wordToIdx.get(w);
    if (idx != null) {
      obsIds.push(idx);
      obsLabels.push(state.profile.answers[w]);
    }
  }
  const theta = estimateTheta(obsIds, obsLabels);

  const scoredPool = pool.map((x) => {
    const p = predictProba(theta, x.idx);
    return { ...x, uncertainty: -Math.abs(p - 0.5) };
  });
  scoredPool.sort((a, b) => b.uncertainty - a.uncertainty);
  const evalPool = scoredPool.slice(0, 96);
  const candidatePool = scoredPool.slice(0, 48);

  function entropy(p) {
    p = Math.max(1e-8, Math.min(1 - 1e-8, p));
    return -(p * Math.log(p) + (1 - p) * Math.log(1 - p));
  }

  function meanEntropy(t, excludeIdx) {
    let sum = 0;
    let count = 0;
    for (const e of evalPool) {
      if (e.idx === excludeIdx) continue;
      const p = sigmoid(t - state.model.b[e.idx]);
      sum += entropy(p);
      count++;
    }
    return count === 0 ? 0 : sum / count;
  }

  const eerScores = candidatePool.map((c) => {
    const pKnown = clip01(sigmoid(theta - state.model.b[c.idx]));
    const theta1 = estimateTheta([...obsIds, c.idx], [...obsLabels, 1]);
    const theta0 = estimateTheta([...obsIds, c.idx], [...obsLabels, 0]);
    const h1 = meanEntropy(theta1, c.idx);
    const h0 = meanEntropy(theta0, c.idx);
    const eer = -(pKnown * h1 + (1 - pKnown) * h0);
    return { ...c, eer };
  });

  eerScores.sort((a, b) => b.eer - a.eer);

  const temperature = 0.02;
  const topK = 1;
  const out = [];
  const available = eerScores.map((x, idx) => ({ ...x, _poolIndex: idx }));

  for (let i = 0; i < Math.min(q, available.length); i++) {
    const candidates = available.slice(0, Math.min(topK, available.length));
    const maxScore = Math.max(...candidates.map((c) => c.eer));
    const logits = candidates.map((c) => (c.eer - maxScore) / temperature);
    const expLogits = logits.map((l) => Math.exp(Math.max(-60, Math.min(60, l))));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map((e) => e / sumExp);

    let r = rng();
    let chosenIdx = 0;
    for (let j = 0; j < probs.length; j++) {
      r -= probs[j];
      if (r <= 0) {
        chosenIdx = j;
        break;
      }
    }
    const removedPoolIndex = candidates[chosenIdx]._poolIndex;
    out.push(candidates[chosenIdx].word);
    available.splice(removedPoolIndex, 1);
    available.forEach((x, idx) => { x._poolIndex = idx; });
  }

  if (out.length < q) {
    const used = new Set(out);
    const remaining = scoredPool.filter((x) => !used.has(x.word));
    out.push(...remaining.slice(0, q - out.length).map((x) => x.word));
  }

  return out;
}

function getQuizWordsAdaptiveHybridFast(questionCount, rng) {
  const q = Math.max(10, Math.min(200, questionCount));
  const candidateInfo = getCandidatePool();
  if (candidateInfo.length <= q) return candidateInfo.map((x) => x.word);

  const answered = new Set(Object.keys(state.profile.answers));
  let pool = candidateInfo.filter((x) => !answered.has(x.word));
  if (pool.length < q) pool = candidateInfo;

  const obsIds = [];
  const obsLabels = [];
  for (const w of Object.keys(state.profile.answers)) {
    const idx = state.model.wordToIdx.get(w);
    if (idx != null) {
      obsIds.push(idx);
      obsLabels.push(state.profile.answers[w]);
    }
  }
  const theta = estimateTheta(obsIds, obsLabels);

  const scoredPool = pool.map((x) => {
    const p = predictProba(theta, x.idx);
    return { ...x, uncertainty: -Math.abs(p - 0.5) };
  });
  scoredPool.sort((a, b) => b.uncertainty - a.uncertainty);

  const stage1Count = Math.min(30, q);
  const stage1 = scoredPool.slice(0, stage1Count);

  let stage2 = [];
  if (q > stage1Count) {
    const remainingPool = scoredPool.slice(stage1Count);

    const evalPool = remainingPool.slice(0, 96);
    const candidatePool = remainingPool.slice(0, 48);

    function entropy(p) {
      p = Math.max(1e-8, Math.min(1 - 1e-8, p));
      return -(p * Math.log(p) + (1 - p) * Math.log(1 - p));
    }

    function meanEntropy(t, excludeIdx) {
      let sum = 0;
      let count = 0;
      for (const e of evalPool) {
        if (e.idx === excludeIdx) continue;
        const p = sigmoid(t - state.model.b[e.idx]);
        sum += entropy(p);
        count++;
      }
      return count === 0 ? 0 : sum / count;
    }

    const eerScores = candidatePool.map((c) => {
      const pKnown = clip01(sigmoid(theta - state.model.b[c.idx]));
      const theta1 = estimateTheta([...obsIds, c.idx], [...obsLabels, 1]);
      const theta0 = estimateTheta([...obsIds, c.idx], [...obsLabels, 0]);
      const h1 = meanEntropy(theta1, c.idx);
      const h0 = meanEntropy(theta0, c.idx);
      const eer = -(pKnown * h1 + (1 - pKnown) * h0);
      return { ...c, eer };
    });

    eerScores.sort((a, b) => b.eer - a.eer);
    stage2 = eerScores.slice(0, q - stage1Count);
  }

  return [...stage1, ...stage2].map((x) => x.word);
}

function getQuizWordsAdaptiveEntropy(questionCount, rng) {
  // For binary case, entropy is maximized at p=0.5, same as uncertainty
  return getQuizWordsAdaptiveUncertainty(questionCount, rng);
}

function getQuizWordsAdaptiveStochasticEntropy(questionCount, rng) {
  // Same as light_random for binary case
  return getQuizWordsAdaptiveUncertaintyLightRandom(questionCount, rng);
}

function getQuizWords(questionCount, strategy) {
  const q = Math.max(10, Math.min(200, questionCount));
  // Resolve auto like CLI: grouped IRT -> adaptive_uncertainty_light_random, else static
  if (strategy === "auto") {
    strategy = state.profile.modelKey === "best_grouped_irt_model" ? "adaptive_uncertainty_light_random" : "static";
  }
  const nowBucket = Math.floor(Date.now() / 60000);
  const seed = hashStringToSeed(`${state.currentNickname}|${q}|${nowBucket}|${strategy}`);
  const rng = createRng(seed);

  switch (strategy) {
    case "semirandom":
      return getQuizWordsSemirandom(q, rng);
    case "adaptive_uncertainty":
      return getQuizWordsAdaptiveUncertainty(q, rng);
    case "adaptive_uncertainty_light_random":
      return getQuizWordsAdaptiveUncertaintyLightRandom(q, rng);
    case "adaptive_entropy":
      return getQuizWordsAdaptiveEntropy(q, rng);
    case "adaptive_stochastic_entropy":
      return getQuizWordsAdaptiveStochasticEntropy(q, rng);
    case "adaptive_eer_fast":
      return getQuizWordsAdaptiveEerFast(q, rng);
    case "adaptive_hybrid_fast":
      return getQuizWordsAdaptiveHybridFast(q, rng);
    case "static":
    default:
      return getQuizWordsStatic(q, rng);
  }
}

function isAdaptiveStrategy(strategy) {
  return [
    "adaptive_uncertainty",
    "adaptive_uncertainty_light_random",
    "adaptive_entropy",
    "adaptive_stochastic_entropy",
    "adaptive_eer_fast",
    "adaptive_hybrid_fast",
    "auto",
  ].includes(strategy);
}

function getAdaptiveBatchWords(batchSize, strategy, rng) {
  // Resolve auto like getQuizWords does
  if (strategy === "auto") {
    strategy = state.profile.modelKey === "best_grouped_irt_model" ? "adaptive_uncertainty_light_random" : "static";
  }
  switch (strategy) {
    case "adaptive_uncertainty":
      return getQuizWordsAdaptiveUncertainty(batchSize, rng);
    case "adaptive_uncertainty_light_random":
      return getQuizWordsAdaptiveUncertaintyLightRandom(batchSize, rng);
    case "adaptive_entropy":
      return getQuizWordsAdaptiveEntropy(batchSize, rng);
    case "adaptive_stochastic_entropy":
      return getQuizWordsAdaptiveStochasticEntropy(batchSize, rng);
    case "adaptive_eer_fast":
      return getQuizWordsAdaptiveEerFast(batchSize, rng);
    case "adaptive_hybrid_fast":
      return getQuizWordsAdaptiveHybridFast(batchSize, rng);
    default:
      return getQuizWordsStatic(batchSize, rng);
  }
}

function getObservedPairs(quizWords) {
  const ids = [];
  const labels = [];
  for (const w of quizWords) {
    if (Object.prototype.hasOwnProperty.call(state.profile.answers, w)) {
      ids.push(state.model.wordToIdx.get(w));
      labels.push(state.profile.answers[w]);
    }
  }
  return { ids, labels };
}

function estimateTheta(obsIds, obsLabels, priorVar = 25.0, steps = 20) {
  let theta = 0.0;
  if (obsIds.length === 0) return theta;
  for (let k = 0; k < steps; k++) {
    let grad = -theta / priorVar;
    let h = -1.0 / priorVar;
    for (let i = 0; i < obsIds.length; i++) {
      const z = theta - state.model.b[obsIds[i]];
      const p = sigmoid(z);
      grad += obsLabels[i] - p;
      h -= p * (1 - p);
    }
    if (Math.abs(h) < 1e-8) break;
    theta = theta - grad / h;
  }
  return theta;
}

function predictProba(theta, wordIdx) {
  return clip01(sigmoid(theta - state.model.b[wordIdx]));
}

function effectiveWordBelief(theta, word, wordIdx) {
  if (Object.prototype.hasOwnProperty.call(state.profile.answers, word)) {
    return { p: state.profile.answers[word] === 1 ? 1.0 : 0.0, observed: true };
  }
  return { p: predictProba(theta, wordIdx), observed: false };
}

function splitSentences(text) {
  return [...text.matchAll(SENTENCE_RE)].map((m) => m[0].trim()).filter(Boolean);
}

function analyzeBook(theta) {
  const tokens = [...state.bookText.matchAll(WORD_RE)].map((m) => m[0]);
  const lemmas = contextualDeinflect(tokens, state.model.vocabSet);

  const tokenCount = lemmas.length;
  const tokenFreq = new Map();
  const inVocabWords = new Map();

  for (const lemma of lemmas) {
    tokenFreq.set(lemma, (tokenFreq.get(lemma) || 0) + 1);
    const idx = state.model.wordToIdx.get(lemma);
    if (idx != null) inVocabWords.set(lemma, idx);
  }

  let inVocabTokenCount = 0;
  let unknownTokenCount = 0;
  const knownRows = [];
  const unknownRows = [];

  for (const [word, idx] of inVocabWords.entries()) {
    const count = tokenFreq.get(word) || 0;
    const { p, observed } = effectiveWordBelief(theta, word, idx);
    inVocabTokenCount += count;
    if (p >= 0.5) knownRows.push({ word, p, count, observed });
    else {
      unknownRows.push({ word, p, count });
      unknownTokenCount += count;
    }
  }

  const sample = (arr, n) => {
    const xs = [...arr].sort((a, b) => a.word.localeCompare(b.word));
    if (xs.length <= n) return xs;
    const out = [];
    const used = new Set();
    while (out.length < n) {
      const i = Math.floor(Math.random() * xs.length);
      if (!used.has(i)) { used.add(i); out.push(xs[i]); }
    }
    return out.sort((a, b) => a.word.localeCompare(b.word));
  };

  const sentenceRows = [];
  for (const sentence of splitSentences(state.bookText)) {
    const raw = [...sentence.matchAll(WORD_RE)].map((m) => m[0]);
    if (raw.length < 4 || raw.length > 35) continue;
    const sentenceLemmas = contextualDeinflect(raw, state.model.vocabSet);
    const unknowns = [];
    let hasOov = false;
    for (const l of sentenceLemmas) {
      const idx = state.model.wordToIdx.get(l);
      if (idx == null) { hasOov = true; break; }
      const { p } = effectiveWordBelief(theta, l, idx);
      if (p < 0.5) unknowns.push({ word: l, p });
    }
    if (!hasOov && unknowns.length === 1) {
      sentenceRows.push({ sentence, word: unknowns[0].word, p: unknowns[0].p });
    }
  }

  const shuffledSentences = [...sentenceRows].sort(() => Math.random() - 0.5).slice(0, 10);

  return {
    tokenCount,
    inVocabTokenCount,
    oovTokenCount: tokenCount - inVocabTokenCount,
    unknownTokenCount,
    unknownPct: inVocabTokenCount === 0 ? 0 : (100 * unknownTokenCount) / inVocabTokenCount,
    knownRows: sample(knownRows, 25),
    unknownRows: sample(unknownRows, 25),
    oneUnknownSentences: shuffledSentences,
  };
}

function renderStats(a) {
  const rows = [
    ["Word tokens analyzed", a.tokenCount],
    ["In-vocabulary tokens used for estimates", a.inVocabTokenCount],
    ["Out-of-model-vocabulary tokens discarded", a.oovTokenCount],
    ["Estimated unknown in-vocabulary tokens", `${a.unknownTokenCount} (${a.unknownPct.toFixed(2)}%)`],
  ];
  ui.estimateStats.innerHTML = rows.map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
}

function renderWordList(el, rows) {
  if (rows.length === 0) {
    el.innerHTML = `<p class="meta">No words found in this category.</p>`;
    return;
  }
  el.innerHTML = rows.map((r) => {
    const source = r.observed ? "observed" : "model";
    return `<div class="word-item"><strong>${r.word}</strong><br><span class="meta">p_known=${r.p.toFixed(3)} · count=${r.count} · ${source}</span></div>`;
  }).join("");
}

function renderSentences(rows) {
  if (rows.length === 0) {
    ui.sentenceList.innerHTML = `<li class="meta">No matching sentences found under the strict criterion.</li>`;
    return;
  }
  ui.sentenceList.innerHTML = rows.map((r) => `<li><span class="meta">[${r.word}, p_known=${r.p.toFixed(3)}]</span><br>${r.sentence}</li>`).join("");
}

function renderChecklist(quizWords) {
  const html = quizWords.map((word, idx) => {
    const checked = state.profile.answers[word] === 1 ? "checked" : "";
    return `<label class="check-item" for="word_${idx}"><input id="word_${idx}" type="checkbox" data-word="${word}" ${checked} /><span>${word}</span></label>`;
  }).join("");
  ui.checklistWrap.innerHTML = `<div class="checklist-grid">${html}</div>`;
}

function runEstimation() {
  const ids = [];
  const labels = [];
  for (const w of Object.keys(state.profile.answers)) {
    const idx = state.model.wordToIdx.get(w);
    if (idx != null) {
      ids.push(idx);
      labels.push(state.profile.answers[w]);
    }
  }
  const theta = estimateTheta(ids, labels);
  const analysis = analyzeBook(theta);

  renderStats(analysis);
  renderWordList(ui.knownList, analysis.knownRows);
  renderWordList(ui.unknownList, analysis.unknownRows);
  renderSentences(analysis.oneUnknownSentences);

  ui.resultsSection.classList.remove("hidden");
  ui.addWordsSection.classList.remove("hidden");
  ui.statusText.textContent = `Profile '${state.currentNickname}' ready. Model: ${state.profile.modelKey}. Strategy: ${state.profile.strategy}. Observed: ${ids.length}.`;
}

function startChecklist() {
  const q = Number(ui.questionCount.value);
  const strategy = ui.quizStrategy.value;
  state.profile.questionCount = q;
  state.profile.strategy = strategy;
  saveCurrentProfile();

  state.totalQuestionCount = q;
  state.currentBatch = 0;
  state.quizSeed = hashStringToSeed(`${state.currentNickname}|${q}|${Date.now()}|${strategy}`);
  state.isAdaptiveQuiz = isAdaptiveStrategy(strategy);

  if (state.isAdaptiveQuiz) {
    state.quizWords = [];
  } else {
    state.quizWords = getQuizWords(q, strategy);
  }

  loadNextBatch();

  ui.quizSection.classList.remove("hidden");
  ui.resultsSection.classList.add("hidden");
  ui.addWordsSection.classList.add("hidden");
}

function loadNextBatch() {
  const batchStart = state.currentBatch * state.batchSize;
  const batchEnd = Math.min(batchStart + state.batchSize, state.totalQuestionCount);
  const remainingTotal = state.totalQuestionCount - batchStart;
  const batchCount = Math.ceil(state.totalQuestionCount / state.batchSize);

  let batchWords;
  if (state.isAdaptiveQuiz) {
    const batchSeed = hashStringToSeed(`${state.quizSeed}|batch${state.currentBatch}`);
    const rng = createRng(batchSeed);
    const toAsk = Math.min(state.batchSize, remainingTotal);
    batchWords = getAdaptiveBatchWords(toAsk, state.profile.strategy, rng);
    state.quizWords.push(...batchWords);
  } else {
    batchWords = state.quizWords.slice(batchStart, batchEnd);
  }

  renderChecklist(batchWords);

  const isLastBatch = batchEnd >= state.totalQuestionCount;
  ui.submitChecklistBtn.textContent = isLastBatch ? "Submit Answers" : "Next Batch";
  ui.quizProgress.textContent = `Batch ${state.currentBatch + 1} of ${batchCount} (${batchWords.length} words). Strategy: ${state.profile.strategy}. Total observed: ${Object.keys(state.profile.answers).length}.`;
}

function submitBatch() {
  const checks = ui.checklistWrap.querySelectorAll("input[type='checkbox'][data-word]");
  for (const item of checks) {
    const word = item.getAttribute("data-word");
    state.profile.answers[word] = item.checked ? 1 : 0;
  }
  saveCurrentProfile();

  state.currentBatch++;
  const nextBatchStart = state.currentBatch * state.batchSize;

  if (nextBatchStart < state.totalQuestionCount) {
    loadNextBatch();
  } else {
    runEstimation();
  }
}

function retakeTest() {
  state.profile.answers = {};
  state.currentBatch = 0;
  state.quizWords = [];
  saveCurrentProfile();
  ui.quizSection.classList.add("hidden");
  ui.resultsSection.classList.add("hidden");
  ui.addWordsSection.classList.add("hidden");
  ui.statusText.textContent = `Profile '${state.currentNickname}' answers cleared. Click 'Open checklist' to retake.`;
}

function parseAddWordsInput(raw) {
  const out = [];
  const lines = raw.split(/[\n,]+/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const parts = trimmed.split("=");
    if (parts.length !== 2) continue;
    const word = normalizeWord(parts[0]);
    const labelRaw = parts[1].trim().toLowerCase();
    let label = null;
    if (labelRaw === "known" || labelRaw === "1" || labelRaw === "y" || labelRaw === "yes" || labelRaw === "k") label = 1;
    if (labelRaw === "unknown" || labelRaw === "0" || labelRaw === "n" || labelRaw === "no" || labelRaw === "u") label = 0;
    if (word && label !== null) out.push([word, label]);
  }
  return out;
}

function submitAddWords() {
  const raw = ui.addWordsTextarea.value;
  const entries = parseAddWordsInput(raw);
  if (entries.length === 0) {
    ui.addWordsStatus.textContent = "No valid entries found. Use format: word=known or word=unknown.";
    return;
  }
  let added = 0;
  let updated = 0;
  for (const [word, label] of entries) {
    if (!state.model.wordToIdx.has(word)) continue;
    if (Object.prototype.hasOwnProperty.call(state.profile.answers, word)) {
      updated++;
    } else {
      added++;
    }
    state.profile.answers[word] = label;
  }
  saveCurrentProfile();
  ui.addWordsStatus.textContent = `Added ${added} new label(s), updated ${updated}. Total observed: ${Object.keys(state.profile.answers).length}.`;
  ui.addWordsTextarea.value = "";
  // Re-run estimation if results are visible
  if (!ui.resultsSection.classList.contains("hidden")) {
    runEstimation();
  }
}

async function switchModel() {
  const modelKey = ui.modelSelect.value;
  state.profile.modelKey = modelKey;
  saveCurrentProfile();
  ui.statusText.textContent = `Loading model: ${modelKey} ...`;
  try {
    await loadData(modelKey);
    ui.statusText.textContent = `Model loaded: ${state.model.model_name || modelKey}. Profile: ${state.currentNickname}.`;
    if (!ui.resultsSection.classList.contains("hidden")) {
      runEstimation();
    }
  } catch (err) {
    ui.statusText.textContent = `Failed to load model ${modelKey}: ${err.message}`;
  }
}

function switchProfile() {
  state.currentNickname = safeNickname(ui.nicknameInput.value);
  ui.nicknameInput.value = state.currentNickname;
  loadCurrentProfile();
  initControls();
  ui.statusText.textContent = `Using profile '${state.currentNickname}'. Saved answers: ${Object.keys(state.profile.answers).length}.`;
  ui.quizSection.classList.add("hidden");
  ui.resultsSection.classList.add("hidden");
  ui.addWordsSection.classList.add("hidden");
}

function resetProfile() {
  state.profile = {
    answers: {},
    questionCount: Number(ui.questionCount.value) || 100,
    modelKey: ui.modelSelect.value || MODEL_DEFAULT,
    strategy: ui.quizStrategy.value || STRATEGY_DEFAULT,
  };
  saveCurrentProfile();
  ui.quizSection.classList.add("hidden");
  ui.resultsSection.classList.add("hidden");
  ui.addWordsSection.classList.add("hidden");
  ui.statusText.textContent = `Profile '${state.currentNickname}' reset.`;
}

async function main() {
  applyTheme(loadTheme());

  state.currentNickname = state.profiles.currentNickname || "default";
  ui.nicknameInput.value = state.currentNickname;
  loadCurrentProfile();
  initControls();

  ui.statusText.textContent = `Loading model: ${state.profile.modelKey || MODEL_DEFAULT} ...`;
  await loadData(state.profile.modelKey || MODEL_DEFAULT);
  await loadDefaultBook();
  ui.statusText.textContent = `Data loaded. Model: ${state.model.model_name || state.profile.modelKey}. Profile: ${state.currentNickname}.`;

  ui.questionCount.addEventListener("input", () => {
    ui.questionCountValue.textContent = ui.questionCount.value;
  });

  ui.modelSelect.addEventListener("change", switchModel);
  ui.quizStrategy.addEventListener("change", () => {
    state.profile.strategy = ui.quizStrategy.value;
    saveCurrentProfile();
  });
  ui.loadProfileBtn.addEventListener("click", switchProfile);
  ui.nicknameInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") switchProfile();
  });
  ui.startBtn.addEventListener("click", startChecklist);
  ui.retakeBtn.addEventListener("click", retakeTest);
  ui.submitChecklistBtn.addEventListener("click", submitBatch);
  ui.resetBtn.addEventListener("click", resetProfile);
  ui.submitAddWordsBtn.addEventListener("click", submitAddWords);
  if (ui.bookUpload) {
    ui.bookUpload.addEventListener("change", handleBookUpload);
  }
  if (ui.resetBookBtn) {
    ui.resetBookBtn.addEventListener("click", resetBook);
  }
  if (ui.themeToggleBtn) {
    ui.themeToggleBtn.addEventListener("click", toggleTheme);
  }
}

main().catch((err) => {
  ui.statusText.textContent = `Failed to load app data: ${err.message}`;
  console.error(err);
});
