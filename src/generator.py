from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class GeneratorConfig:
    dataset_size: int
    max_sentence_length: int
    tasks: List[str]
    random_seed: Optional[int] = None

    @classmethod
    def load(cls, path: Path) -> "GeneratorConfig":
        with path.open("r", encoding="utf-8") as fh:
            text = fh.read()
        data = cls._parse_config_text(text)
        return cls(
            dataset_size=int(data.get("dataset_size", 100)),
            max_sentence_length=int(data.get("max_sentence_length", 12)),
            tasks=list(data.get("tasks", ["lm", "parsing", "nl2sem", "sem2nl"])),
            random_seed=data.get("random_seed"),
        )

    @staticmethod
    def _parse_config_text(text: str) -> Dict[str, object]:
        if yaml is not None:
            loaded = yaml.safe_load(text) or {}
            if isinstance(loaded, dict):
                return loaded
            raise ValueError("YAML configuration must be a mapping")

        # Fallback parser for a limited subset of YAML used in config.yaml
        result: Dict[str, object] = {}
        current_list_key: Optional[str] = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("- ") and current_list_key:
                item = line[2:].strip()
                result[current_list_key].append(GeneratorConfig._parse_scalar(item))
                continue
            if ":" in line:
                key, value = [part.strip() for part in line.split(":", 1)]
                if not value:
                    result[key] = []
                    current_list_key = key
                elif value.startswith("[") and value.endswith("]"):
                    items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
                    result[key] = [GeneratorConfig._parse_scalar(item) for item in items]
                    current_list_key = None
                else:
                    result[key] = GeneratorConfig._parse_scalar(value)
                    current_list_key = None
            else:
                raise ValueError(f"Cannot parse configuration line: {raw_line}")
        return result

    @staticmethod
    def _parse_scalar(value: str) -> object:
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            pass
        if (value.startswith("\"") and value.endswith("\"")) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1]
        return value


class Lexicon:
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


class Grammar:
    def __init__(self, grammar_dir: Path) -> None:
        self.rules: Dict[str, List[List[str]]] = {}
        rules_path = grammar_dir / "rules.txt"
        print("rules_path")
        print(rules_path)
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

    def _collect_noun_prefix_info(self) -> Dict[str, bool]:
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


class RoleQueue:
    def __init__(self, semantics: Dict[str, object]):
        self._queue: List[Tuple[str, Dict[str, object]]] = []
        subject = semantics.get("субъект")
        if isinstance(subject, dict):
            self._queue.append(("субъект", subject))
        obj = semantics.get("объект")
        if isinstance(obj, dict):
            self._queue.append(("объект", obj))

    def next_role(self) -> Tuple[str, Dict[str, object]]:
        if not self._queue:
            return ("N", {})
        return self._queue.pop(0)

    def clone(self) -> "RoleQueue":
        clone = RoleQueue({})
        clone._queue = list(self._queue)
        return clone


class SemanticsGenerator:
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


class SurfaceRealizer:
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


class DatasetWriter:
    def __init__(self, output_dir: Path, tasks: Sequence[str]) -> None:
        self.output_dir = output_dir
        self.tasks = set(task.lower() for task in tasks)
        self.buffers: Dict[str, List[str]] = {task: [] for task in self.tasks}

    def add_entry(self, sentence_tokens: List[str], semantics: Dict[str, object]) -> None:
        sentence = " ".join(sentence_tokens)
        semantics_json = json.dumps(semantics, ensure_ascii=False, sort_keys=True)

        if "lm" in self.tasks:
            for index in range(1, len(sentence_tokens)):
                input_text = " ".join(sentence_tokens[:index])
                output_text = sentence_tokens[index]
                self.buffers["lm"].append(
                    json.dumps({"input": input_text, "output": output_text}, ensure_ascii=False)
                )

        if "parsing" in self.tasks:
            self.buffers["parsing"].append(
                json.dumps({"input": sentence, "output": json.loads(semantics_json)}, ensure_ascii=False)
            )

        if "nl2sem" in self.tasks:
            self.buffers["nl2sem"].append(
                json.dumps({"input": sentence, "output": json.loads(semantics_json)}, ensure_ascii=False)
            )

        if "sem2nl" in self.tasks:
            self.buffers["sem2nl"].append(
                json.dumps({"input": json.loads(semantics_json), "output": sentence}, ensure_ascii=False)
            )

    def flush(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for task, lines in self.buffers.items():
            path = self.output_dir / f"{task}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(line + "\n")


def main() -> None:
    base_dir: Path = Path().resolve().parent
    config = GeneratorConfig.load(Path(__file__).resolve().parent / "generator/config.yaml")
    if config.random_seed is not None:
        random.seed(config.random_seed)

    lexicon = Lexicon(base_dir / "data/dicts")
    grammar = Grammar(base_dir / "data/grammar")
    semantics_gen = SemanticsGenerator(lexicon, grammar)
    realizer = SurfaceRealizer(grammar, lexicon, config.max_sentence_length)
    writer = DatasetWriter(base_dir / "data/datasets", config.tasks)

    generated = 0
    attempts = 0
    max_attempts = config.dataset_size * 10
    while generated < config.dataset_size and attempts < max_attempts:
        attempts += 1
        semantics = semantics_gen.generate()
        try:
            tokens = realizer.realize(semantics)
        except (ValueError, RuntimeError):
            continue
        if not tokens:
            continue
        writer.add_entry(tokens, semantics)
        generated += 1

    writer.flush()


if __name__ == "__main__":
    #run_generator(Path(__file__).resolve().parent)
    main()
