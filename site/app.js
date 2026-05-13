const STORAGE_KEY = "rasch_book_profiles_v2";
const THEME_KEY = "rasch_book_theme_v1";
const WORD_RE = /[A-Za-z]+(?:['’][A-Za-z]+)?/g;
const SENTENCE_RE = /[^.!?]+[.!?]+|[^.!?]+$/g;

const ui = {
  nicknameInput: document.getElementById("nicknameInput"),
  loadProfileBtn: document.getElementById("loadProfileBtn"),
  questionCount: document.getElementById("questionCount"),
  questionCountValue: document.getElementById("questionCountValue"),
  startBtn: document.getElementById("startBtn"),
  resetBtn: document.getElementById("resetBtn"),
  statusText: document.getElementById("statusText"),
  quizSection: document.getElementById("quizSection"),
  quizProgress: document.getElementById("quizProgress"),
  checklistWrap: document.getElementById("checklistWrap"),
  submitChecklistBtn: document.getElementById("submitChecklistBtn"),
  resultsSection: document.getElementById("resultsSection"),
  estimateStats: document.getElementById("estimateStats"),
  knownList: document.getElementById("knownList"),
  unknownList: document.getElementById("unknownList"),
  sentenceList: document.getElementById("sentenceList"),
  themeToggleBtn: document.getElementById("themeToggleBtn"),
};

const state = {
  model: null,
  bookText: "",
  profiles: loadProfilesStore(),
  currentNickname: "default",
  profile: { answers: {}, questionCount: 30 },
  quizWords: [],
  theme: "light",
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
  state.profile = existing || { answers: {}, questionCount: 30 };
  state.profiles.profiles[state.currentNickname] = state.profile;
  state.profiles.currentNickname = state.currentNickname;
}

function saveCurrentProfile() {
  state.profiles.profiles[state.currentNickname] = state.profile;
  state.profiles.currentNickname = state.currentNickname;
  saveProfilesStore();
}

function initQuestionControl() {
  const q = Math.min(50, Math.max(10, Number(state.profile.questionCount) || 30));
  ui.questionCount.value = String(q);
  ui.questionCountValue.textContent = String(q);
}

async function loadData() {
  const [modelRes, bookRes] = await Promise.all([
    fetch("./data/model_data.json"),
    fetch("./data/hitchhikers_guide.txt"),
  ]);
  state.model = await modelRes.json();
  state.bookText = await bookRes.text();
  state.model.wordToIdx = new Map(state.model.words.map((w, i) => [w, i]));
  state.model.vocabSet = new Set(state.model.words);
  state.model.b = state.model.accuracy.map((a) => {
    const p = a == null ? 0.5 : clip01(a);
    return -logit(p);
  });
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

function getQuizWords(questionCount) {
  const q = Math.max(10, Math.min(50, questionCount));
  const basePool = state.model.query_pool;
  if (!Array.isArray(basePool) || basePool.length === 0) return [];

  const candidateCount = Math.max(q * 6, 180);
  const candidates = basePool.slice(0, Math.min(basePool.length, candidateCount));
  const candidateInfo = candidates
    .map((word) => {
      const idx = state.model.wordToIdx.get(word);
      if (idx == null) return null;
      const acc = clip01(state.model.accuracy[idx] == null ? 0.5 : state.model.accuracy[idx]);
      return { word, acc };
    })
    .filter(Boolean);
  if (candidateInfo.length <= q) return candidateInfo.map((x) => x.word);

  const nowBucket = Math.floor(Date.now() / 60000);
  const seed = hashStringToSeed(`${state.currentNickname}|${q}|${nowBucket}`);
  const rng = createRng(seed);

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
      unknownRows.push({ word, p, count, observed });
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
  const { ids, labels } = getObservedPairs(state.quizWords);
  const theta = estimateTheta(ids, labels);
  const analysis = analyzeBook(theta);

  renderStats(analysis);
  renderWordList(ui.knownList, analysis.knownRows);
  renderWordList(ui.unknownList, analysis.unknownRows);
  renderSentences(analysis.oneUnknownSentences);

  ui.resultsSection.classList.remove("hidden");
  ui.statusText.textContent = `Profile '${state.currentNickname}' ready. Observed labels used: ${ids.length}.`;
}

function startChecklist() {
  const q = Number(ui.questionCount.value);
  state.profile.questionCount = q;
  saveCurrentProfile();
  state.quizWords = getQuizWords(q);

  renderChecklist(state.quizWords);
  ui.quizProgress.textContent = `Mark known words, then submit (${state.quizWords.length} total).`;
  ui.quizSection.classList.remove("hidden");
  ui.resultsSection.classList.add("hidden");
}

function submitChecklist() {
  const checks = ui.checklistWrap.querySelectorAll("input[type='checkbox'][data-word]");
  for (const item of checks) {
    const word = item.getAttribute("data-word");
    state.profile.answers[word] = item.checked ? 1 : 0;
  }
  saveCurrentProfile();
  runEstimation();
}

function switchProfile() {
  state.currentNickname = safeNickname(ui.nicknameInput.value);
  ui.nicknameInput.value = state.currentNickname;
  loadCurrentProfile();
  initQuestionControl();
  ui.statusText.textContent = `Using profile '${state.currentNickname}'. Saved answers: ${Object.keys(state.profile.answers).length}.`;
  ui.quizSection.classList.add("hidden");
  ui.resultsSection.classList.add("hidden");
}

function resetProfile() {
  state.profile = { answers: {}, questionCount: Number(ui.questionCount.value) || 30 };
  saveCurrentProfile();
  ui.quizSection.classList.add("hidden");
  ui.resultsSection.classList.add("hidden");
  ui.statusText.textContent = `Profile '${state.currentNickname}' reset.`;
}

async function main() {
  applyTheme(loadTheme());
  await loadData();

  state.currentNickname = state.profiles.currentNickname || "default";
  ui.nicknameInput.value = state.currentNickname;
  loadCurrentProfile();
  initQuestionControl();

  ui.statusText.textContent = `Data loaded. Using profile '${state.currentNickname}'.`;
  ui.questionCount.addEventListener("input", () => {
    ui.questionCountValue.textContent = ui.questionCount.value;
  });

  ui.loadProfileBtn.addEventListener("click", switchProfile);
  ui.nicknameInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") switchProfile();
  });
  ui.startBtn.addEventListener("click", startChecklist);
  ui.submitChecklistBtn.addEventListener("click", submitChecklist);
  ui.resetBtn.addEventListener("click", resetProfile);
  if (ui.themeToggleBtn) {
    ui.themeToggleBtn.addEventListener("click", toggleTheme);
  }
}

main().catch((err) => {
  ui.statusText.textContent = `Failed to load app data: ${err.message}`;
});
