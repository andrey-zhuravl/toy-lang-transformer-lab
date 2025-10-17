from __future__ import annotations

import random
from typing import Dict, List

from .grammar import Grammar
from .lexicon import Lexicon


class SemanticsGenerator:
    """Produces toy semantic structures."""

    def __init__(self, lexicon: Lexicon, grammar: Grammar) -> None:
        self.lexicon = lexicon
        self.grammar = grammar

    def generate(self) -> Dict[str, object]:
        verb = self.lexicon.sample("verbs")
        semantics: Dict[str, object] = {"действие": verb}

        subject = self._build_entity()
        semantics["субъект"] = subject

        if random.random() < 0.7:
            semantics["объект"] = self._build_entity()

        return semantics

    def _build_entity(self) -> Dict[str, object]:
        entity: Dict[str, object] = {"тип": self.lexicon.sample("nouns")}
        noun_info = self.grammar.noun_prefix_info

        if noun_info.get("allow_adj") and self.lexicon.has_category("adjectives") and random.random() < 0.6:
            entity["признак"] = self.lexicon.sample("adjectives")

        quantity_options: List[str] = []
        if noun_info.get("allow_num") and self.lexicon.has_category("numerals"):
            quantity_options.extend([word for word in self.lexicon.words("numerals") if word != "много"])
        numeral_words = set(self.lexicon.words("numerals"))
        if "много" in noun_info.get("literals", set()) or "много" in numeral_words:
            quantity_options.append("много")
        if quantity_options and random.random() < 0.5:
            entity["количество"] = random.choice(quantity_options)
        return entity
