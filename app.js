const PROFILE_STORAGE_KEY = "vocab_reader_profiles_v2";
const UI_STORAGE_KEY = "vocab_reader_ui_state_v1";
const BOOK_DB_NAME = "vocab_reader_books_v1";
const BOOK_STORE = "books";
const WORD_RE = /[A-Za-z]+(?:['’][A-Za-z]+)?/g;
const SENTENCE_RE = /[^.!?]+[.!?]+|[^.!?]+$/g;

const MODEL_KEY = "best_grouped_irt_model";
const QUIZ_STRATEGY = "adaptive_uncertainty_light_random";
const DEFAULT_QUIZ_SIZE = 60;
const DEFAULT_THRESHOLD = 0.5;
const DEFAULT_PROFILE_NAME = "default";
const SUGGESTION_SCORING = {
  freqWeight: 0.65,
  sentenceWeight: 0.35,
};
const CARD_SCORING = {
  freqWeight: 0.7,
  unknownWeight: 0.3,
};

const WORD_TOKEN_RE = /^[A-Za-z]+(?:['’][A-Za-z]+)?$/;
const NAME_LIKE_TOKEN_RE = /^[A-Z][A-Za-z]*(?:['’-][A-Za-z]+)*$/;
const COMPROMISE_PROPER_TAGS = new Set([
  "ProperNoun", "Person", "FirstName", "LastName", "MaleName", "FemaleName", "Place", "City", "Country", "Region", "Organization", "Demonym",
]);
const TITLE_CASE_NOISE_TOKENS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "did", "do", "does", "first",
  "for", "from", "had", "has", "have", "he", "her", "here", "him", "his", "how", "however", "i", "if",
  "in", "is", "it", "its", "me", "my", "no", "not", "of", "oh", "on", "or", "our", "she", "so", "that",
  "the", "their", "them", "then", "there", "these", "they", "this", "those", "to", "was", "we", "were",
  "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
]);
const CALENDAR_WORD_EXCLUSIONS = new Set([
  "monday", "mon", "tuesday", "tue", "tues", "wednesday", "wed", "thursday", "thu", "thur", "thurs", "friday", "fri", "saturday", "sat", "sunday", "sun",
  "january", "jan", "february", "feb", "march", "mar", "april", "apr", "may", "june", "jun", "july", "jul", "august", "aug", "september", "sep", "sept", "october", "oct", "november", "nov", "december", "dec",
]);

const ui = {
  goLibraryBtn: document.getElementById("goLibraryBtn"),
  goReaderBtn: document.getElementById("goReaderBtn"),
  goProfileBtn: document.getElementById("goProfileBtn"),
  statusText: document.getElementById("statusText"),
  onboardingBanner: document.getElementById("onboardingBanner"),

  libraryView: document.getElementById("libraryView"),
  libraryList: document.getElementById("libraryList"),
  librarySuggestions: document.getElementById("librarySuggestions"),
  bookUpload: document.getElementById("bookUpload"),

  readerView: document.getElementById("readerView"),
  chapterNav: document.getElementById("chapterNav"),
  readerBookTitle: document.getElementById("readerBookTitle"),
  readerChapterTitle: document.getElementById("readerChapterTitle"),
  readerParagraphs: document.getElementById("readerParagraphs"),
  suggestRemainingBtn: document.getElementById("suggestRemainingBtn"),
  suggestNextBtn: document.getElementById("suggestNextBtn"),
  suggestWholeReaderBtn: document.getElementById("suggestWholeReaderBtn"),
  toggleAssistBtn: document.getElementById("toggleAssistBtn"),
  readerSuggestions: document.getElementById("readerSuggestions"),

  profileView: document.getElementById("profileView"),
  profileNameInput: document.getElementById("profileNameInput"),
  createProfileBtn: document.getElementById("createProfileBtn"),
  deleteProfileBtn: document.getElementById("deleteProfileBtn"),
  profilesText: document.getElementById("profilesText"),
  quizCountInput: document.getElementById("quizCountInput"),
  startQuizBtn: document.getElementById("startQuizBtn"),
  exportProfileBtn: document.getElementById("exportProfileBtn"),
  importProfileInput: document.getElementById("importProfileInput"),
  quizSection: document.getElementById("quizSection"),
  quizProgress: document.getElementById("quizProgress"),
  checklistWrap: document.getElementById("checklistWrap"),
  submitChecklistBtn: document.getElementById("submitChecklistBtn"),
};

const state = {
  model: null,
  lemmaDict: {},
  profiles: { current: DEFAULT_PROFILE_NAME, items: {} },
  activeProfile: null,
  books: [],
  currentBookId: "",
  currentChapterIdx: 0,
  assistEnabled: true,
  quizWords: [],
  quizBatchSize: 10,
  quizBatchIndex: 0,
  lexiconEntries: new Map(),
};

function normalizeWord(token) {
  if (typeof token !== "string") return "";
  return token.toLowerCase().replaceAll("’", "'").replace(/^'+|'+$/g, "");
}

function safeNickname(raw) {
  const out = String(raw || "").trim().toLowerCase().replace(/[^a-z0-9_.-]+/g, "_");
  return out || DEFAULT_PROFILE_NAME;
}

function clip01(p) { return Math.min(1 - 1e-6, Math.max(1e-6, p)); }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function logit(p) { return Math.log(p / (1 - p)); }

function orderedUnique(items) {
  const out = [];
  const seen = new Set();
  for (const item of items) {
    if (!item || seen.has(item)) continue;
    seen.add(item);
    out.push(item);
  }
  return out;
}

function extractCompromiseTermTags(term) {
  const raw = term && term.tags ? term.tags : null;
  const normalizeTag = (tag) => String(tag || "").replace(/^#/, "").trim();
  if (Array.isArray(raw)) return new Set(raw.map(normalizeTag).filter(Boolean));
  if (raw && typeof raw === "object") return new Set(Object.keys(raw).map(normalizeTag).filter(Boolean));
  return new Set();
}

function isProperNounTag(termTags, rawToken, tokenPos) {
  for (const tag of COMPROMISE_PROPER_TAGS) if (termTags.has(tag)) return true;
  if (!WORD_TOKEN_RE.test(rawToken)) return false;
  if (rawToken.toUpperCase() === rawToken) return false;
  if (!/^[A-Z]/.test(rawToken)) return false;
  const normalized = normalizeWord(rawToken);
  if (!normalized) return false;
  if (CALENDAR_WORD_EXCLUSIONS.has(normalized)) return false;
  if (TITLE_CASE_NOISE_TOKENS.has(normalized)) return false;
  if (tokenPos === 0 && normalized.length <= 2) return false;
  return true;
}

function isNameLikeToken(rawToken) {
  if (rawToken === "") return false;
  if (rawToken.toUpperCase() === rawToken) return false;
  if (NAME_LIKE_TOKEN_RE.test(rawToken) === false) return false;
  const normalized = normalizeWord(rawToken);
  if (!normalized) return false;
  if (CALENDAR_WORD_EXCLUSIONS.has(normalized)) return false;
  if (TITLE_CASE_NOISE_TOKENS.has(normalized)) return false;
  return true;
}

function tagSentenceTerms(sentence) {
  const fallbackTerms = [...sentence.matchAll(WORD_RE)].map((m) => ({ raw: m[0], normalized: normalizeWord(m[0]), tags: new Set() }));
  if (typeof nlp === "undefined") return fallbackTerms;
  let terms;
  try { terms = nlp(sentence).terms().json(); } catch { return fallbackTerms; }
  if (!Array.isArray(terms) || terms.length === 0) return fallbackTerms;
  const out = [];
  for (const termMatch of terms) {
    const termNodes = Array.isArray(termMatch && termMatch.terms) && termMatch.terms.length > 0 ? termMatch.terms : [termMatch];
    for (const node of termNodes) {
      const rawText = String(node && node.text ? node.text : "").trim();
      if (!rawText) continue;
      const tags = extractCompromiseTermTags(node);
      const matches = [...rawText.matchAll(WORD_RE)];
      for (const match of matches) {
        if (!WORD_TOKEN_RE.test(match[0])) continue;
        out.push({ raw: match[0], normalized: normalizeWord(match[0]), tags });
      }
    }
  }
  return out.length > 0 ? out : fallbackTerms;
}

function splitSentences(text) {
  return [...text.matchAll(SENTENCE_RE)].map((m) => m[0].trim()).filter(Boolean);
}

function buildTaggedSentences(text) {
  return splitSentences(text).map((sentence) => ({ sentence, taggedTerms: tagSentenceTerms(sentence) }));
}

function buildHighConfidenceProperNounLexicon(taggedSentences) {
  const stats = new Map();
  for (const sentence of taggedSentences) {
    for (let i = 0; i < sentence.length; i += 1) {
      const row = sentence[i];
      const normalized = row.normalized;
      if (!normalized) continue;
      if (!stats.has(normalized)) stats.set(normalized, { total: 0, proper: 0, sentenceInitialProper: 0, lowercaseSeen: 0, nameLikeProper: 0 });
      const agg = stats.get(normalized);
      agg.total += 1;
      if (row.raw.slice(0, 1).toLowerCase() === row.raw.slice(0, 1)) agg.lowercaseSeen += 1;
      if (isProperNounTag(row.tags, row.raw, i)) {
        agg.proper += 1;
        if (i === 0) agg.sentenceInitialProper += 1;
        if (isNameLikeToken(row.raw)) agg.nameLikeProper += 1;
      }
    }
  }
  const out = new Set();
  for (const [normalized, row] of stats.entries()) {
    if (CALENDAR_WORD_EXCLUSIONS.has(normalized)) continue;
    if (row.proper < 2) continue;
    if (row.total <= 0) continue;
    if ((row.proper / row.total) < 0.60) continue;
    if (row.nameLikeProper < 2) continue;
    if (row.lowercaseSeen > 0) continue;
    if (row.sentenceInitialProper === row.proper && row.proper < 5) continue;
    out.add(normalized);
  }
  return out;
}

function makeLemmaCandidates(rawToken, termTags) {
  const normalized = normalizeWord(rawToken);
  if (!normalized) return [];
  const candidates = [];
  const addCandidate = (value) => {
    if (typeof value !== "string") return;
    const candidate = normalizeWord(value);
    if (!candidate || !WORD_TOKEN_RE.test(candidate)) return;
    candidates.push(candidate);
  };
  if (state.lemmaDict[normalized]) addCandidate(state.lemmaDict[normalized]);
  if (typeof nlp !== "undefined") {
    try {
      const doc = nlp(rawToken);
      if (termTags.has("Verb")) addCandidate(doc.verbs().toInfinitive().text());
      if (termTags.has("Noun")) addCandidate(doc.nouns().toSingular().text());
      if (termTags.has("Adjective")) {
        const adjConj = doc.adjectives().conjugate();
        if (Array.isArray(adjConj) && adjConj.length > 0 && adjConj[0].adjective) addCandidate(adjConj[0].adjective);
      }
      addCandidate(doc.verbs().toInfinitive().text());
      addCandidate(doc.nouns().toSingular().text());
    } catch {
      // No-op.
    }
  }
  addCandidate(normalized);
  return orderedUnique(candidates);
}

function contextualDeinflectTaggedTerms(taggedTerms, lowerToIdx, excludeProperNouns, properNounLexicon) {
  const outTokens = [];
  const outFlags = [];
  for (let i = 0; i < taggedTerms.length; i += 1) {
    const row = taggedTerms[i];
    const token = row.normalized;
    const tagProper = isProperNounTag(row.tags, row.raw, i);
    const properByLexicon = properNounLexicon == null ? tagProper : (tagProper && properNounLexicon.has(token));
    outFlags.push(properByLexicon);
    if (properByLexicon && excludeProperNouns) {
      outTokens.push("");
      continue;
    }
    const candidates = makeLemmaCandidates(row.raw, row.tags);
    const selected = candidates.find((candidate) => lowerToIdx.has(candidate)) || candidates[0] || token;
    outTokens.push(selected);
  }
  return { tokens: outTokens, properFlags: outFlags };
}

function loadProfilesStore() {
  try {
    const raw = localStorage.getItem(PROFILE_STORAGE_KEY);
    if (!raw) return { current: DEFAULT_PROFILE_NAME, items: {} };
    const parsed = JSON.parse(raw);
    return { current: safeNickname(parsed.current || DEFAULT_PROFILE_NAME), items: parsed.items || {} };
  } catch {
    return { current: DEFAULT_PROFILE_NAME, items: {} };
  }
}

function saveProfilesStore() {
  localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(state.profiles));
}

function loadUiPrefs() {
  try {
    const raw = localStorage.getItem(UI_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    state.assistEnabled = parsed.assistEnabled !== false;
  } catch {
    // ignore
  }
}

function saveUiPrefs() {
  localStorage.setItem(UI_STORAGE_KEY, JSON.stringify({ assistEnabled: state.assistEnabled }));
}

function ensureProfile(name) {
  if (!state.profiles.items[name]) {
    state.profiles.items[name] = {
      version: 2,
      name,
      observed: {},
      settings: {
        knownThreshold: DEFAULT_THRESHOLD,
        quizSize: DEFAULT_QUIZ_SIZE,
      },
      quizMeta: {
        strategy: QUIZ_STRATEGY,
        lastTakenAt: null,
      },
    };
  }
  state.profiles.current = name;
  state.activeProfile = state.profiles.items[name];
  saveProfilesStore();
}

function getObservedPairs() {
  const ids = [];
  const labels = [];
  const entries = Object.entries(state.activeProfile.observed);
  for (const [word, label] of entries) {
    const idx = state.model.wordToIdx.get(word);
    if (idx == null) continue;
    ids.push(idx);
    labels.push(label);
  }
  return { ids, labels };
}

function estimateTheta(obsIds, obsLabels, priorVar = 25.0, steps = 20) {
  let theta = 0.0;
  if (obsIds.length === 0) return theta;
  for (let k = 0; k < steps; k += 1) {
    let grad = -theta / priorVar;
    let h = -1.0 / priorVar;
    for (let i = 0; i < obsIds.length; i += 1) {
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
  if (Object.prototype.hasOwnProperty.call(state.activeProfile.observed, word)) {
    return { p: state.activeProfile.observed[word] === 1 ? 1.0 : 0.0, observed: true };
  }
  return { p: predictProba(theta, wordIdx), observed: false };
}

function hashStringToSeed(input) {
  let h = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
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

function getCandidatePool() {
  const adaptivePool = Array.isArray(state.model.adaptive_candidate_pool) && state.model.adaptive_candidate_pool.length > 0
    ? state.model.adaptive_candidate_pool
    : state.model.query_pool;
  return adaptivePool
    .map((word) => {
      const idx = state.model.wordToIdx.get(word);
      if (idx == null) return null;
      const acc = clip01(state.model.accuracy[idx] == null ? 0.5 : state.model.accuracy[idx]);
      return { word, idx, acc };
    })
    .filter(Boolean);
}

function getQuizWordsAdaptiveUncertaintyLightRandom(questionCount, rng) {
  const q = Math.max(20, Math.min(200, questionCount));
  const candidateInfo = getCandidatePool();
  if (candidateInfo.length <= q) return candidateInfo.map((x) => x.word);
  const answered = new Set(Object.keys(state.activeProfile.observed));
  let pool = candidateInfo.filter((x) => !answered.has(x.word));
  if (pool.length < q) pool = candidateInfo;
  const { ids, labels } = getObservedPairs();
  const theta = estimateTheta(ids, labels);
  const scored = pool.map((x, position) => {
    const p = predictProba(theta, x.idx);
    const uncertainty = p * (1 - p);
    return { ...x, uncertainty, position };
  });
  const topK = 3;
  const temperature = 0.03;
  const out = [];
  const available = scored.map((_, idx) => idx);
  for (let i = 0; i < Math.min(q, available.length); i += 1) {
    const candidates = [...available]
      .sort((a, b) => {
        const uncertaintyDelta = scored[b].uncertainty - scored[a].uncertainty;
        if (Math.abs(uncertaintyDelta) > 1e-12) return uncertaintyDelta;
        return scored[a].position - scored[b].position;
      })
      .slice(0, Math.min(topK, available.length));
    const candidateUncertainty = candidates.map((idx) => scored[idx].uncertainty);
    const maxScore = Math.max(...candidateUncertainty);
    const logits = candidateUncertainty.map((value) => (value / temperature) - (maxScore / temperature));
    const expLogits = logits.map((l) => Math.exp(Math.max(-60, Math.min(60, l))));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map((e) => e / sumExp);
    let r = rng();
    let chosenIdx = 0;
    for (let j = 0; j < probs.length; j += 1) {
      r -= probs[j];
      if (r <= 0) { chosenIdx = j; break; }
    }
    const chosenScoredIndex = candidates[chosenIdx];
    out.push(scored[chosenScoredIndex].word);
    const removalIndex = available.indexOf(chosenScoredIndex);
    if (removalIndex >= 0) available.splice(removalIndex, 1);
  }
  return out;
}

function pickQuizWords(quizSize, seed) {
  const rng = createRng(seed);
  return getQuizWordsAdaptiveUncertaintyLightRandom(quizSize, rng);
}

function resolveQuizSeed(quizSize) {
  const maybeSeed = Number(state.activeProfile.quizMeta.seed);
  if (Number.isInteger(maybeSeed) && maybeSeed >= 0) return maybeSeed >>> 0;
  const generated = hashStringToSeed(`${state.activeProfile.name}|${quizSize}|${Date.now()}`);
  state.activeProfile.quizMeta.seed = generated >>> 0;
  return generated >>> 0;
}

function getDb() {
  if (typeof indexedDB === "undefined") return Promise.resolve(null);
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(BOOK_DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(BOOK_STORE)) db.createObjectStore(BOOK_STORE, { keyPath: "id" });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function dbPutBook(book) {
  const db = await getDb();
  if (!db) {
    const raw = localStorage.getItem("fallback_books_v1");
    const arr = raw ? JSON.parse(raw) : [];
    const filtered = arr.filter((x) => x.id !== book.id);
    filtered.push(book);
    localStorage.setItem("fallback_books_v1", JSON.stringify(filtered));
    return;
  }
  await new Promise((resolve, reject) => {
    const tx = db.transaction(BOOK_STORE, "readwrite");
    tx.objectStore(BOOK_STORE).put(book);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function dbDeleteBook(id) {
  const db = await getDb();
  if (!db) {
    const raw = localStorage.getItem("fallback_books_v1");
    const arr = raw ? JSON.parse(raw) : [];
    localStorage.setItem("fallback_books_v1", JSON.stringify(arr.filter((x) => x.id !== id)));
    return;
  }
  await new Promise((resolve, reject) => {
    const tx = db.transaction(BOOK_STORE, "readwrite");
    tx.objectStore(BOOK_STORE).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function dbListBooks() {
  const db = await getDb();
  if (!db) {
    const raw = localStorage.getItem("fallback_books_v1");
    return raw ? JSON.parse(raw) : [];
  }
  return await new Promise((resolve, reject) => {
    const tx = db.transaction(BOOK_STORE, "readonly");
    const req = tx.objectStore(BOOK_STORE).getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
}

function parseTxtBook(name, text) {
  const chapterRegex = /^\s*(chapter\s+\d+.*)$/gim;
  const matches = [...text.matchAll(chapterRegex)];
  let chapters = [];
  if (matches.length === 0) {
    chapters = [{ title: "Chapter 1", text }];
  } else {
    for (let i = 0; i < matches.length; i += 1) {
      const start = matches[i].index;
      const end = i + 1 < matches.length ? matches[i + 1].index : text.length;
      chapters.push({ title: matches[i][1].trim(), text: text.slice(start, end) });
    }
  }
  return {
    id: `book_${hashStringToSeed(name + text.length)}`,
    title: name,
    format: "txt",
    chapters: chapters.map((c, idx) => {
      const paragraphs = c.text.split(/\n\s*\n+/).map((p) => p.replace(/\s+/g, " ").trim()).filter(Boolean);
      return { id: `ch_${idx + 1}`, title: c.title, paragraphs };
    }),
    addedAt: Date.now(),
  };
}

async function parseEpubBook(file) {
  if (typeof JSZip === "undefined") {
    throw new Error("EPUB parser dependency is missing. Reload and try again.");
  }

  const normalizeSpaces = (value) => String(value || "").replace(/\s+/g, " ").trim();
  const sanitizePath = (value) => String(value || "").replace(/\\/g, "/").replace(/^\/+/, "");
  const splitPath = (value) => sanitizePath(value).split("/").filter(Boolean);
  const joinPath = (parts) => parts.join("/");
  const dirname = (value) => {
    const parts = splitPath(value);
    if (parts.length <= 1) return "";
    return joinPath(parts.slice(0, -1));
  };
  const resolveRelativePath = (baseFile, href) => {
    const hrefClean = sanitizePath(href);
    if (!hrefClean) return "";
    const baseDir = splitPath(dirname(baseFile));
    const hrefParts = splitPath(hrefClean);
    const out = [...baseDir];
    for (const part of hrefParts) {
      if (part === ".") continue;
      if (part === "..") {
        if (out.length > 0) out.pop();
        continue;
      }
      out.push(part);
    }
    return joinPath(out);
  };
  const parseXml = (xmlText) => {
    const parser = new DOMParser();
    const doc = parser.parseFromString(xmlText, "application/xml");
    if (doc.querySelector("parsererror")) {
      throw new Error("Invalid EPUB XML.");
    }
    return doc;
  };
  const parseHtml = (htmlText) => {
    const parser = new DOMParser();
    return parser.parseFromString(htmlText, "text/html");
  };
  const pickTitle = (doc, fallback) => {
    const direct = doc.querySelector("h1, h2, title");
    const text = normalizeSpaces(direct ? direct.textContent : "");
    return text || fallback;
  };
  const extractParagraphs = (doc) => {
    const out = [];
    const paragraphNodes = [...doc.querySelectorAll("p")];
    for (const node of paragraphNodes) {
      const text = normalizeSpaces(node.textContent || "");
      if (text.length > 0) out.push(text);
    }
    if (out.length > 0) return out;

    const bodyText = normalizeSpaces(doc.body ? doc.body.textContent : "");
    if (!bodyText) return [];
    return bodyText
      .split(/(?<=[.!?])\s+/)
      .map((row) => normalizeSpaces(row))
      .filter((row) => row.length > 0);
  };

  const buffer = await file.arrayBuffer();
  const zip = await JSZip.loadAsync(buffer);
  const containerEntry = zip.file("META-INF/container.xml");
  if (!containerEntry) {
    throw new Error("EPUB is missing META-INF/container.xml.");
  }

  const containerXml = await containerEntry.async("text");
  const containerDoc = parseXml(containerXml);
  const rootfileNode = containerDoc.querySelector("rootfile");
  const opfPathRaw = rootfileNode ? rootfileNode.getAttribute("full-path") : "";
  const opfPath = sanitizePath(opfPathRaw);
  if (!opfPath) {
    throw new Error("EPUB package path is missing.");
  }

  const opfEntry = zip.file(opfPath);
  if (!opfEntry) {
    throw new Error("EPUB package document is missing.");
  }

  const opfXml = await opfEntry.async("text");
  const opfDoc = parseXml(opfXml);
  const itemNodes = [...opfDoc.querySelectorAll("manifest > item")];
  const manifestById = new Map();
  for (const item of itemNodes) {
    const id = item.getAttribute("id");
    const href = item.getAttribute("href");
    if (!id || !href) continue;
    const mediaType = item.getAttribute("media-type") || "";
    manifestById.set(id, { href, mediaType });
  }

  const spineNodes = [...opfDoc.querySelectorAll("spine > itemref")];
  const spineEntries = spineNodes
    .map((node) => node.getAttribute("idref"))
    .filter(Boolean)
    .map((idref) => ({ idref, item: manifestById.get(idref) }))
    .filter((row) => row.item && row.item.href);

  if (spineEntries.length === 0) {
    throw new Error("EPUB spine is empty.");
  }

  const chapters = [];
  for (let i = 0; i < spineEntries.length; i += 1) {
    const entry = spineEntries[i];
    const contentPath = resolveRelativePath(opfPath, entry.item.href);
    const fileEntry = zip.file(contentPath);
    if (!fileEntry) continue;
    const content = await fileEntry.async("text");
    const doc = parseHtml(content);
    const title = pickTitle(doc, `Chapter ${i + 1}`);
    const paragraphs = extractParagraphs(doc);
    if (paragraphs.length === 0) continue;
    chapters.push({
      id: `ch_${i + 1}`,
      title,
      paragraphs,
    });
  }

  if (chapters.length === 0) {
    throw new Error("No readable chapter content found in EPUB.");
  }

  const fallbackTitle = String(file.name || "Untitled EPUB").replace(/\.epub$/i, "").trim() || "Untitled EPUB";
  const metaTitleNode = opfDoc.querySelector("metadata > title, metadata > dc\\:title");
  const title = normalizeSpaces(metaTitleNode ? metaTitleNode.textContent : "") || fallbackTitle;

  return {
    id: `book_${hashStringToSeed(`${file.name}|${file.size}|${chapters.length}`)}`,
    title,
    format: "epub",
    chapters,
    addedAt: Date.now(),
  };
}

function computeUnknownSet(book) {
  const { ids, labels } = getObservedPairs();
  const theta = estimateTheta(ids, labels);
  const threshold = Number(state.activeProfile.settings.knownThreshold) || DEFAULT_THRESHOLD;
  const unknown = new Set();
  for (const word of state.model.words) {
    const idx = state.model.wordToIdx.get(word);
    const { p } = effectiveWordBelief(theta, word, idx);
    if (p < threshold) unknown.add(word);
  }
  return { unknown, theta, threshold };
}

function analyzeScopeText(text, unknownSet) {
  const taggedSentenceRows = buildTaggedSentences(text);
  const taggedSentences = taggedSentenceRows.map((row) => row.taggedTerms);
  const properLexicon = buildHighConfidenceProperNounLexicon(taggedSentences);
  const freq = new Map();
  const sentenceStats = new Map();

  for (const row of taggedSentenceRows) {
    const terms = contextualDeinflectTaggedTerms(row.taggedTerms, state.model.wordToIdx, true, properLexicon).tokens;
    const sentenceWords = [];
    for (const term of terms) {
      if (!term) continue;
      if (!state.model.wordToIdx.has(term)) continue;
      sentenceWords.push(term);
      if (unknownSet.has(term)) freq.set(term, (freq.get(term) || 0) + 1);
    }
    const unknownInSentence = [...new Set(sentenceWords.filter((w) => unknownSet.has(w)))];
    for (const word of unknownInSentence) {
      if (!sentenceStats.has(word)) sentenceStats.set(word, []);
      sentenceStats.get(word).push({ sentence: row.sentence, unknownCount: unknownInSentence.length, knownCount: sentenceWords.length - unknownInSentence.length });
    }
  }

  const rows = [];
  let maxFreq = 1;
  for (const count of freq.values()) if (count > maxFreq) maxFreq = count;

  for (const [word, count] of freq.entries()) {
    const contexts = sentenceStats.get(word) || [];
    contexts.sort((a, b) => (a.unknownCount - b.unknownCount) || (b.knownCount - a.knownCount));
    const best = contexts.slice(0, 3);
    const score = computeSuggestionScore({
      count,
      maxFreq,
      contexts,
      freqWeight: SUGGESTION_SCORING.freqWeight,
      sentenceWeight: SUGGESTION_SCORING.sentenceWeight,
    });
    rows.push({ word, count, score, examples: best.map((x) => x.sentence) });
  }

  rows.sort((a, b) => b.score - a.score || b.count - a.count || a.word.localeCompare(b.word));
  return rows;
}

function computeSentenceHelpfulnessScore(contexts) {
  const oneUnknownHits = contexts.filter((c) => c.unknownCount === 1).length;
  return contexts.length === 0 ? 0 : (oneUnknownHits / contexts.length);
}

function computeFrequencyScore(count, maxFreq) {
  return Math.log(1 + count) / Math.log(1 + maxFreq);
}

function computeSuggestionScore(params) {
  const sentenceScore = computeSentenceHelpfulnessScore(params.contexts);
  const freqScore = computeFrequencyScore(params.count, params.maxFreq);
  return (params.freqWeight * freqScore) + (params.sentenceWeight * sentenceScore);
}

function computeInlineCardImportance(params) {
  return (CARD_SCORING.freqWeight * params.freq) + (CARD_SCORING.unknownWeight * params.uncertaintyScore);
}

function getLexiconEntry(word) {
  const row = state.lexiconEntries.get(word);
  if (row) {
    return {
      word,
      ipa: typeof row.ipa === "string" && row.ipa.trim() ? row.ipa.trim() : `/${word}/`,
      definition: typeof row.definition === "string" && row.definition.trim()
        ? row.definition.trim()
        : "Definition unavailable in this build.",
    };
  }
  return {
    word,
    ipa: `/${word}/`,
    definition: "Definition unavailable in this build.",
  };
}

function renderSuggestions(container, title, rows) {
  if (rows.length === 0) {
    container.innerHTML = `<h3>${title}</h3><p class=\"meta\">No suggestions in this scope.</p>`;
    container.classList.remove("hidden");
    return;
  }
  container.innerHTML = `<h3>${title}</h3>${rows.slice(0, 30).map((r) => `<div class=\"suggestion-item\"><strong>${r.word}</strong> <span class=\"meta\">score=${r.score.toFixed(3)} · count=${r.count}</span>${r.examples.map((s) => `<p>${escapeHtml(s)}</p>`).join("")}</div>`).join("")}`;
  container.classList.remove("hidden");
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderView(name) {
  ui.libraryView.classList.toggle("hidden", name !== "library");
  ui.readerView.classList.toggle("hidden", name !== "reader");
  ui.profileView.classList.toggle("hidden", name !== "profile");
}

function renderOnboardingBanner() {
  const observedCount = Object.keys(state.activeProfile.observed).length;
  const profileCount = Object.keys(state.profiles.items).length;
  if (observedCount > 0 && profileCount > 1) {
    ui.onboardingBanner.classList.add("hidden");
    return;
  }
  ui.onboardingBanner.innerHTML = `<p class=\"notice-head\"><strong>Start with the quiz</strong></p><p class=\"notice-copy\">First-time setup: enter a profile name, click Use profile, then take the initial quiz for accurate reading assistance.</p><div class=\"notice-actions\"><button id=\"bannerUseProfileBtn\" class=\"btn\" type=\"button\">Use current name</button><button id=\"bannerTakeQuizBtn\" class=\"btn\" type=\"button\">Take quiz</button></div>`;
  ui.onboardingBanner.classList.remove("hidden");
  document.getElementById("bannerUseProfileBtn").addEventListener("click", () => {
    const name = safeNickname(ui.profileNameInput.value || state.profiles.current);
    ensureProfile(name);
    renderProfiles();
    ui.statusText.textContent = `Using profile: ${name}.`;
  });
  document.getElementById("bannerTakeQuizBtn").addEventListener("click", () => {
    renderView("profile");
    startQuizFlow();
  });
}

function renderProfiles() {
  const names = Object.keys(state.profiles.items).sort((a, b) => a.localeCompare(b));
  ui.profilesText.textContent = names.length === 0 ? "No profiles." : `Profiles: ${names.map((n) => n === state.profiles.current ? `${n} (current)` : n).join(", ")}`;
  ui.profileNameInput.value = state.profiles.current;
  ui.quizCountInput.value = String(state.activeProfile.settings.quizSize || DEFAULT_QUIZ_SIZE);
}

function renderLibrary() {
  if (state.books.length === 0) {
    ui.libraryList.innerHTML = `<p class="meta">No books in library yet. Import .txt or .epub.</p>`;
    return;
  }
  ui.libraryList.innerHTML = state.books.map((book) => `
    <div class="library-item">
      <div>
        <strong>${escapeHtml(book.title)}</strong>
        <div class="meta">${book.format.toUpperCase()} · ${book.chapters.length} chapter(s)</div>
      </div>
      <div class="actions">
        <button class="btn" data-open-book="${book.id}" type="button">Open</button>
        <button class="btn danger" data-delete-book="${book.id}" type="button">Delete</button>
      </div>
    </div>
  `).join("");

  ui.libraryList.querySelectorAll("[data-open-book]").forEach((btn) => btn.addEventListener("click", () => openBook(btn.getAttribute("data-open-book"))));
  ui.libraryList.querySelectorAll("[data-delete-book]").forEach((btn) => btn.addEventListener("click", () => deleteBook(btn.getAttribute("data-delete-book"))));
}

function renderReader() {
  const book = state.books.find((b) => b.id === state.currentBookId);
  if (!book) {
    ui.readerBookTitle.textContent = "No book open";
    ui.readerChapterTitle.textContent = "";
    ui.chapterNav.innerHTML = "";
    ui.readerParagraphs.innerHTML = "";
    return;
  }

  const chapter = book.chapters[state.currentChapterIdx] || book.chapters[0];
  ui.readerBookTitle.textContent = book.title;
  ui.readerChapterTitle.textContent = chapter.title;
  ui.toggleAssistBtn.textContent = `Assist: ${state.assistEnabled ? "on" : "off"}`;

  ui.chapterNav.innerHTML = book.chapters.map((c, idx) => `<button class="chapter-btn ${idx === state.currentChapterIdx ? "active" : ""}" data-chapter-idx="${idx}" type="button">${escapeHtml(c.title)}</button>`).join("");
  ui.chapterNav.querySelectorAll("[data-chapter-idx]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.currentChapterIdx = Number(btn.getAttribute("data-chapter-idx"));
      renderReader();
    });
  });

  const { unknown, theta, threshold } = computeUnknownSet(book);
  const properLexicon = buildHighConfidenceProperNounLexicon(chapter.paragraphs.map((p) => tagSentenceTerms(p)));

  ui.readerParagraphs.innerHTML = chapter.paragraphs.map((paragraph, idx) => {
    const tagged = tagSentenceTerms(paragraph);
    const deinflected = contextualDeinflectTaggedTerms(tagged, state.model.wordToIdx, true, properLexicon).tokens;
    const unknownWords = [...new Set(deinflected.filter((w) => w && unknown.has(w)))];

    const maxCards = window.matchMedia("(min-width: 981px)").matches ? 3 : 2;
    const ranked = unknownWords.map((word) => {
      const freq = deinflected.filter((x) => x === word).length;
      const idxWord = state.model.wordToIdx.get(word);
      const { p } = effectiveWordBelief(theta, word, idxWord);
      const uncertaintyScore = (1 - p) / Math.max(1e-6, 1 - threshold);
      const importance = computeInlineCardImportance({ freq, uncertaintyScore });
      return { word, importance };
    }).sort((a, b) => b.importance - a.importance).slice(0, maxCards);

    const cards = state.assistEnabled ? ranked.map(({ word }) => {
      const entry = getLexiconEntry(word);
      return `<article class="word-card"><h4>${entry.word}</h4><div class="ipa">${entry.ipa || "[N/A]"}</div><p>${escapeHtml(entry.definition)}</p><div class="actions"><button class="btn" data-mark-word="${word}" data-mark-label="1" data-source="para_${idx}" type="button">Known</button><button class="btn" data-mark-word="${word}" data-mark-label="0" data-source="para_${idx}" type="button">Unknown</button></div></article>`;
    }).join("") : "";

    return `<div class="paragraph-row"><div class="paragraph-text">${escapeHtml(paragraph)}</div><div class="assist-col">${cards}</div></div>`;
  }).join("");

  ui.readerParagraphs.querySelectorAll("[data-mark-word]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const word = btn.getAttribute("data-mark-word");
      const label = Number(btn.getAttribute("data-mark-label"));
      state.activeProfile.observed[word] = label;
      saveProfilesStore();
      renderOnboardingBanner();
      renderReader();
      ui.statusText.textContent = `Updated '${word}' as ${label === 1 ? "known" : "unknown"}.`;
    });
  });
}

function openBook(bookId) {
  state.currentBookId = bookId;
  state.currentChapterIdx = 0;
  renderView("reader");
  renderReader();
}

function getCurrentChapterRemainingText(book) {
  const chapter = book.chapters[state.currentChapterIdx];
  if (!chapter) return "";
  return chapter.paragraphs.join("\n\n");
}

function getNextChapterText(book) {
  const chapter = book.chapters[state.currentChapterIdx + 1];
  if (!chapter) return "";
  return chapter.paragraphs.join("\n\n");
}

function suggestRemaining() {
  const book = state.books.find((b) => b.id === state.currentBookId);
  if (!book) {
    ui.statusText.textContent = "Open a book first.";
    return;
  }
  const { unknown } = computeUnknownSet(book);
  const text = getCurrentChapterRemainingText(book);
  const rows = analyzeScopeText(text, unknown);
  renderSuggestions(ui.readerSuggestions, "Suggestions For Remaining Chapter", rows);
}

function suggestNextChapter() {
  const book = state.books.find((b) => b.id === state.currentBookId);
  if (!book) {
    ui.statusText.textContent = "Open a book first.";
    return;
  }
  const text = getNextChapterText(book);
  if (!text) {
    ui.statusText.textContent = "No next chapter available.";
    return;
  }
  const { unknown } = computeUnknownSet(book);
  const rows = analyzeScopeText(text, unknown);
  const nextTitle = book.chapters[state.currentChapterIdx + 1].title;
  renderSuggestions(ui.readerSuggestions, `Suggestions For Next Chapter: ${nextTitle}`, rows);
}

function suggestWholeBook(bookId) {
  const book = state.books.find((b) => b.id === bookId);
  if (!book) return;
  const { unknown } = computeUnknownSet(book);
  const text = book.chapters.map((c) => c.paragraphs.join("\n\n")).join("\n\n");
  const rows = analyzeScopeText(text, unknown);
  renderSuggestions(ui.librarySuggestions, `Whole-book suggestions: ${book.title}`, rows);
}

function suggestWholeCurrentBook() {
  if (!state.currentBookId) {
    ui.statusText.textContent = "Open a book first.";
    return;
  }
  const book = state.books.find((b) => b.id === state.currentBookId);
  if (!book) return;
  const { unknown } = computeUnknownSet(book);
  const text = book.chapters.map((c) => c.paragraphs.join("\n\n")).join("\n\n");
  const rows = analyzeScopeText(text, unknown);
  renderSuggestions(ui.readerSuggestions, `Whole-book suggestions: ${book.title}`, rows);
}

async function deleteBook(bookId) {
  await dbDeleteBook(bookId);
  state.books = state.books.filter((b) => b.id !== bookId);
  if (state.currentBookId === bookId) state.currentBookId = "";
  renderLibrary();
  renderReader();
}

function startQuizFlow() {
  if (!state.activeProfile.quizMeta || typeof state.activeProfile.quizMeta !== "object") {
    state.activeProfile.quizMeta = { strategy: QUIZ_STRATEGY, lastTakenAt: null, seed: 0 };
  }
  const q = Math.max(20, Math.min(200, Number(ui.quizCountInput.value) || DEFAULT_QUIZ_SIZE));
  const seed = resolveQuizSeed(q);
  state.activeProfile.settings.quizSize = q;
  state.activeProfile.quizMeta.strategy = QUIZ_STRATEGY;
  state.activeProfile.quizMeta.seed = seed;
  state.activeProfile.quizMeta.lastTakenAt = Date.now();
  state.quizWords = pickQuizWords(q, seed);
  state.quizBatchIndex = 0;
  ui.quizSection.classList.remove("hidden");
  renderQuizBatch();
  saveProfilesStore();
}

function renderQuizBatch() {
  const start = state.quizBatchIndex * state.quizBatchSize;
  const end = Math.min(start + state.quizBatchSize, state.quizWords.length);
  const words = state.quizWords.slice(start, end);
  ui.quizProgress.textContent = `Batch ${state.quizBatchIndex + 1} of ${Math.ceil(state.quizWords.length / state.quizBatchSize)} (${words.length} words)`;
  ui.checklistWrap.innerHTML = `<div class="checklist-grid">${words.map((word, idx) => `<label class="check-item" for="quiz_${idx}"><input id="quiz_${idx}" type="checkbox" data-word="${word}" ${state.activeProfile.observed[word] === 1 ? "checked" : ""} /><span>${word}</span></label>`).join("")}</div>`;
  const isLast = end >= state.quizWords.length;
  ui.submitChecklistBtn.textContent = isLast ? "Finish Quiz" : "Next Batch";
}

function submitQuizBatch() {
  const checks = ui.checklistWrap.querySelectorAll("input[type='checkbox'][data-word]");
  for (const item of checks) {
    const word = item.getAttribute("data-word");
    state.activeProfile.observed[word] = item.checked ? 1 : 0;
  }
  saveProfilesStore();
  state.quizBatchIndex += 1;
  if ((state.quizBatchIndex * state.quizBatchSize) >= state.quizWords.length) {
    ui.quizSection.classList.add("hidden");
    ui.statusText.textContent = `Quiz complete. Observed words: ${Object.keys(state.activeProfile.observed).length}.`;
    renderOnboardingBanner();
    renderReader();
    return;
  }
  renderQuizBatch();
}

function exportCurrentProfile() {
  const payload = {
    version: 2,
    profile: state.activeProfile,
    exportedAt: new Date().toISOString(),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${state.activeProfile.name}_profile.json`;
  a.click();
  URL.revokeObjectURL(url);
}

async function importProfileFromFile(file) {
  const text = await file.text();
  const parsed = JSON.parse(text);
  if (!parsed || parsed.version !== 2 || !parsed.profile || typeof parsed.profile.name !== "string") {
    throw new Error("Invalid profile file format.");
  }
  const name = safeNickname(parsed.profile.name);
  state.profiles.items[name] = parsed.profile;
  state.profiles.current = name;
  state.activeProfile = state.profiles.items[name];
  saveProfilesStore();
  renderProfiles();
  renderOnboardingBanner();
  renderReader();
}

async function loadModel() {
  const res = await fetch(`./data/${MODEL_KEY}_model_data.json`);
  if (!res.ok) throw new Error(`Failed to load model ${MODEL_KEY}`);
  state.model = await res.json();
  state.model.wordToIdx = new Map(state.model.words.map((w, i) => [w, i]));
  state.model.vocabSet = new Set(state.model.words);
  state.model.b = state.model.accuracy.map((a) => {
    const p = a == null ? 0.5 : clip01(a);
    return -logit(p);
  });
}

async function loadLemmaDict() {
  try {
    const res = await fetch("./data/lemma_dict.json");
    if (res.ok) state.lemmaDict = await res.json();
  } catch {
    state.lemmaDict = {};
  }
}

async function loadLexicon() {
  const ingestEntries = (rows) => {
    if (!Array.isArray(rows)) return;
    for (const row of rows) {
      if (!row || typeof row.word !== "string") continue;
      const word = normalizeWord(row.word);
      if (!word) continue;
      if (!state.lexiconEntries.has(word)) {
        state.lexiconEntries.set(word, {
          ipa: typeof row.ipa === "string" ? row.ipa : "",
          definition: typeof row.definition === "string" ? row.definition : "",
        });
      }
    }
  };

  const tryLoad = async (path) => {
    try {
      const res = await fetch(path);
      if (!res.ok) return false;
      const payload = await res.json();
      if (Array.isArray(payload)) {
        ingestEntries(payload);
        return true;
      }
      if (payload && Array.isArray(payload.entries)) {
        ingestEntries(payload.entries);
        return true;
      }
      return false;
    } catch {
      return false;
    }
  };

  const loadedFull = await tryLoad("./data/lexicon_full.json");
  if (loadedFull) return;

  // Optional chunked format: index maps chunk key -> file path.
  try {
    const indexRes = await fetch("./data/lexicon/index.json");
    if (!indexRes.ok) return;
    const indexPayload = await indexRes.json();
    if (!indexPayload || typeof indexPayload !== "object") return;
    const chunkFiles = Object.values(indexPayload).filter((v) => typeof v === "string");
    for (const filePath of chunkFiles) {
      // Load all chunks once at startup for synchronous card rendering.
      await tryLoad(`./data/lexicon/${filePath}`);
    }
  } catch {
    // Lexicon remains optional in this build.
  }
}

async function seedLibraryIfEmpty() {
  const books = await dbListBooks();
  if (books.length > 0) {
    state.books = books;
    return;
  }
  const res = await fetch("./data/hitchhikers_guide.txt");
  if (!res.ok) {
    state.books = [];
    return;
  }
  const text = await res.text();
  const sample = parseTxtBook("The Hitchhiker's Guide to the Galaxy", text);
  await dbPutBook(sample);
  state.books = [sample];
}

function bindEvents() {
  ui.goLibraryBtn.addEventListener("click", () => renderView("library"));
  ui.goReaderBtn.addEventListener("click", () => renderView("reader"));
  ui.goProfileBtn.addEventListener("click", () => renderView("profile"));

  ui.createProfileBtn.addEventListener("click", () => {
    const name = safeNickname(ui.profileNameInput.value);
    ensureProfile(name);
    renderProfiles();
    renderOnboardingBanner();
    renderReader();
  });

  ui.deleteProfileBtn.addEventListener("click", () => {
    const name = safeNickname(ui.profileNameInput.value || state.profiles.current);
    if (!state.profiles.items[name]) return;
    delete state.profiles.items[name];
    const names = Object.keys(state.profiles.items).sort((a, b) => a.localeCompare(b));
    ensureProfile(names[0] || DEFAULT_PROFILE_NAME);
    renderProfiles();
    renderOnboardingBanner();
    renderReader();
  });

  ui.startQuizBtn.addEventListener("click", () => {
    renderView("profile");
    startQuizFlow();
  });
  ui.submitChecklistBtn.addEventListener("click", submitQuizBatch);
  ui.exportProfileBtn.addEventListener("click", exportCurrentProfile);
  ui.importProfileInput.addEventListener("change", async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    try {
      await importProfileFromFile(file);
      ui.statusText.textContent = "Profile imported.";
    } catch (err) {
      ui.statusText.textContent = `Failed to import profile: ${err.message}`;
    }
  });

  ui.toggleAssistBtn.addEventListener("click", () => {
    state.assistEnabled = !state.assistEnabled;
    saveUiPrefs();
    renderReader();
  });

  ui.suggestRemainingBtn.addEventListener("click", suggestRemaining);
  ui.suggestNextBtn.addEventListener("click", suggestNextChapter);
  ui.suggestWholeReaderBtn.addEventListener("click", suggestWholeCurrentBook);

  ui.bookUpload.addEventListener("change", async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    try {
      let book;
      if (file.name.toLowerCase().endsWith(".txt")) {
        const text = await file.text();
        book = parseTxtBook(file.name, text);
      } else if (file.name.toLowerCase().endsWith(".epub")) {
        book = await parseEpubBook(file);
      } else {
        throw new Error("Unsupported file format. Use .txt or .epub.");
      }
      await dbPutBook(book);
      state.books = await dbListBooks();
      renderLibrary();
      ui.statusText.textContent = `Imported ${book.title}.`;
    } catch (err) {
      ui.statusText.textContent = `Import failed: ${err.message}`;
    }
  });
}

async function main() {
  loadUiPrefs();
  state.profiles = loadProfilesStore();
  ensureProfile(state.profiles.current || DEFAULT_PROFILE_NAME);
  await Promise.all([loadModel(), loadLemmaDict(), loadLexicon()]);
  await seedLibraryIfEmpty();
  bindEvents();
  renderProfiles();
  renderOnboardingBanner();
  renderLibrary();
  renderReader();
  const isFirstRun = Object.keys(state.profiles.items).length <= 1
    && Object.keys(state.activeProfile.observed).length === 0;
  renderView(isFirstRun ? "profile" : "library");
  ui.statusText.textContent = `Ready. Model: ${MODEL_KEY}. Profile: ${state.activeProfile.name}.`;
}

main().catch((err) => {
  ui.statusText.textContent = `Failed to load app data: ${err.message}`;
  console.error(err);
});
