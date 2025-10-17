from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
