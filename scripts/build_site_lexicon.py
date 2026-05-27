#!/usr/bin/env python3
"""Build static lexicon files consumed by the standalone site reader."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nltk.corpus import wordnet as wn


@dataclass(frozen=True)
class LexiconEntry:
    word: str
    ipa: str
    pos: str
    definition: str


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_word(token: str) -> str:
    return token.strip().lower().replace("’", "'")


def _wordnet_pos_label(pos: str) -> str:
    labels = {
        "n": "noun",
        "v": "verb",
        "a": "adjective",
        "s": "adjective",
        "r": "adverb",
    }
    return labels.get(pos, "")


def _wordnet_entry(word: str) -> tuple[str, str]:
    synsets = wn.synsets(word)
    if len(synsets) == 0:
        return "", ""
    syn = synsets[0]
    pos = _wordnet_pos_label(syn.pos())
    definition = str(syn.definition() or "").strip()
    return pos, definition


def _build_entries(model_words: list[str], overrides: dict[str, dict[str, str]]) -> list[LexiconEntry]:
    out: list[LexiconEntry] = []
    for raw_word in model_words:
        word = _normalize_word(raw_word)
        if word == "":
            continue
        override = overrides.get(word, {})
        definition = str(override.get("definition", "")).strip()
        ipa = str(override.get("ipa", "")).strip()
        pos = str(override.get("pos", "")).strip()
        if definition == "":
            wn_pos, wn_definition = _wordnet_entry(word=word)
            if wn_definition != "":
                definition = wn_definition
            if pos == "":
                pos = wn_pos
        if definition == "":
            definition = "Definition unavailable in this build."
        if ipa == "":
            ipa = f"/{word}/"
        out.append(LexiconEntry(word=word, ipa=ipa, pos=pos, definition=definition))
    return out


def _chunk_entries(entries: list[LexiconEntry]) -> dict[str, list[dict[str, str]]]:
    buckets: dict[str, list[dict[str, str]]] = {}
    for row in entries:
        key = row.word[0] if row.word else "_"
        key = key if ("a" <= key <= "z") else "_"
        if key not in buckets:
            buckets[key] = []
        buckets[key].append({"word": row.word, "ipa": row.ipa, "pos": row.pos, "definition": row.definition})
    for key in buckets:
        buckets[key].sort(key=lambda x: x["word"])
    return buckets


def build_site_lexicon(site_data_dir: Path) -> None:
    model_path = site_data_dir / "best_grouped_irt_model_model_data.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_payload = _load_json(model_path)
    words = model_payload.get("words")
    if not isinstance(words, list):
        raise ValueError("Model payload must include a 'words' list.")

    overrides_path = site_data_dir / "lexicon_overrides.json"
    overrides_payload: dict[str, dict[str, str]] = {}
    if overrides_path.exists():
        parsed = _load_json(overrides_path)
        if not isinstance(parsed, dict):
            raise ValueError("lexicon_overrides.json must be an object.")
        overrides_payload = {
            _normalize_word(str(key)): value
            for key, value in parsed.items()
            if isinstance(value, dict)
        }

    entries = _build_entries(model_words=[str(w) for w in words], overrides=overrides_payload)
    entries.sort(key=lambda x: x.word)

    full_payload = [{"word": x.word, "ipa": x.ipa, "pos": x.pos, "definition": x.definition} for x in entries]
    full_path = site_data_dir / "lexicon_full.json"
    with full_path.open("w", encoding="utf-8") as handle:
        json.dump(full_payload, handle, ensure_ascii=False)

    chunk_dir = site_data_dir / "lexicon"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_index: dict[str, str] = {}
    for key, rows in _chunk_entries(entries).items():
        file_name = f"{key}.json"
        with (chunk_dir / file_name).open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False)
        chunk_index[key] = file_name

    with (chunk_dir / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(chunk_index, handle, ensure_ascii=False)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    build_site_lexicon(site_data_dir=repo_root / "site" / "data")
