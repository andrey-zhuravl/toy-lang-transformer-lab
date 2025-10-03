from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence


class Lexicon:
    """Loads lexical categories from ``*.txt`` files."""

    def __init__(self, dict_dir: Path) -> None:
        self._entries: Dict[str, List[str]] = {}
        for path in sorted(dict_dir.glob("*.txt")):
            key = path.stem.lower()
            with path.open("r", encoding="utf-8") as fh:
                words = [line.strip() for line in fh if line.strip()]
            if words:
                self._entries[key] = words
        self._alias_map = self._build_alias_map()

    def _build_alias_map(self) -> Dict[str, str]:
        aliases: Dict[str, str] = {}
        mapping = {
            "n": "nouns",
            "noun": "nouns",
            "nouns": "nouns",
            "s": "nouns",
            "subj": "nouns",
            "obj": "nouns",
            "adj": "adjectives",
            "adjective": "adjectives",
            "adjectives": "adjectives",
            "v": "verbs",
            "verb": "verbs",
            "verbs": "verbs",
            "num": "numerals",
            "numeral": "numerals",
            "number": "numerals",
            "quant": "numerals",
            "prep": "prepositions",
            "preposition": "prepositions",
            "prepositions": "prepositions",
            "conj": "conjunctions",
            "conjunction": "conjunctions",
            "conjunctions": "conjunctions",
        }
        for alias, target in mapping.items():
            if target in self._entries:
                aliases[alias] = target
        return aliases

    def has_category(self, name: str) -> bool:
        return name.lower() in self._entries

    def sample(self, category: str) -> str:
        key = category.lower()
        key = self._alias_map.get(key, key)
        if key.endswith("s") and key[:-1] in self._entries:
            key = key[:-1]
        if key not in self._entries:
            raise KeyError(f"No entries for category '{category}'")
        return random.choice(self._entries[key])

    def maybe_sample(self, category: str) -> Optional[str]:
        try:
            return self.sample(category)
        except KeyError:
            return None

    def words(self, category: str) -> Sequence[str]:
        key = category.lower()
        key = self._alias_map.get(key, key)
        if key.endswith("s") and key[:-1] in self._entries:
            key = key[:-1]
        return self._entries.get(key, [])
