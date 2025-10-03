from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .grammar import Grammar
from .lexicon import Lexicon
from .role_queue import RoleQueue


class SurfaceRealizer:
    """Turns semantic structures into token sequences."""

    def __init__(self, grammar: Grammar, lexicon: Lexicon, max_length: int) -> None:
        self.grammar = grammar
        self.lexicon = lexicon
        self.max_length = max_length

    def realize(self, semantics: Dict[str, object]) -> List[str]:
        tokens = self._expand_symbol("S", semantics, RoleQueue(semantics), depth=0)
        if len(tokens) > self.max_length:
            raise ValueError("Sentence exceeds maximum length")
        return tokens

    def _expand_symbol(
        self,
        symbol: str,
        semantics: Dict[str, object],
        roles: RoleQueue,
        depth: int,
    ) -> List[str]:
        if depth > 32:
            raise RuntimeError("Grammar expansion exceeded maximum depth")

        if symbol == "S":
            return self._expand_sentence(semantics, roles, depth)
        if symbol == "N":
            return self._expand_noun_phrase(roles.next_role())
        if symbol == "V":
            return [str(semantics.get("действие", self.lexicon.sample("verbs")))]
        if symbol in {"Adj", "ADJ"}:
            return [self.lexicon.sample("adjectives")]
        if symbol in {"Num", "NUM"}:
            return [self.lexicon.sample("numerals")]

        if symbol in self.grammar.rules:
            expansion = random.choice(self.grammar.rules[symbol])
            tokens: List[str] = []
            next_roles = roles.clone()
            for part in expansion:
                tokens.extend(self._expand_symbol(part, semantics, next_roles, depth + 1))
            return tokens

        try:
            return [self.lexicon.sample(symbol)]
        except KeyError:
            return [symbol]

    def _expand_sentence(self, semantics: Dict[str, object], roles: RoleQueue, depth: int) -> List[str]:
        expansions = self.grammar.expansions("S")
        target_nouns = 1 + (1 if "объект" in semantics else 0)
        preferred: List[List[str]] = []
        for expansion in expansions:
            if expansion.count("N") == target_nouns:
                preferred.append(expansion)
        choices = preferred or expansions or [["N", "V"]]
        expansion = random.choice(choices)
        tokens: List[str] = []
        next_roles = roles.clone()
        for part in expansion:
            tokens.extend(self._expand_symbol(part, semantics, next_roles, depth + 1))
        return tokens

    def _expand_noun_phrase(self, role: Tuple[str, Dict[str, object]]) -> List[str]:
        _, features = role
        tokens: List[str] = []
        noun_info = self.grammar.noun_prefix_info

        quantity = features.get("количество")
        if quantity:
            if quantity == "много" and "много" in noun_info.get("literals", set()):
                tokens.append("много")
            elif noun_info.get("allow_num"):
                tokens.append(str(quantity))

        adjective = features.get("признак")
        if adjective and noun_info.get("allow_adj"):
            tokens.append(str(adjective))

        noun = features.get("тип")
        if not noun:
            noun = self.lexicon.sample("nouns")
        tokens.append(str(noun))
        return tokens
