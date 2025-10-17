from __future__ import annotations

from pathlib import Path
from typing import Dict, List


class Grammar:
    """Loads context-free grammar rules for sentence generation."""

    def __init__(self, grammar_dir: Path) -> None:
        self.rules: Dict[str, List[List[str]]] = {}
        rules_path = grammar_dir / "rules.txt"
        if not rules_path.exists():
            raise FileNotFoundError(f"Grammar rules not found: {rules_path}")
        with rules_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "->" not in stripped:
                    continue
                left, right = [part.strip() for part in stripped.split("->", 1)]
                tokens = [token.strip() for token in right.split() if token.strip()]
                if not tokens:
                    continue
                self.rules.setdefault(left, []).append(tokens)
        self.noun_prefix_info = self._collect_noun_prefix_info()

    def _collect_noun_prefix_info(self) -> Dict[str, object]:
        info = {
            "allow_adj": False,
            "allow_num": False,
            "literals": set(),
        }
        for expansion in self.rules.get("N", []):
            if len(expansion) == 2 and expansion[1] == "N":
                prefix = expansion[0]
                lower = prefix.lower()
                if lower in {"adj", "adjective"}:
                    info["allow_adj"] = True
                elif lower in {"num", "numeral", "number"}:
                    info["allow_num"] = True
                else:
                    info["literals"].add(prefix)
        return info

    def expansions(self, symbol: str) -> List[List[str]]:
        return self.rules.get(symbol, [])
