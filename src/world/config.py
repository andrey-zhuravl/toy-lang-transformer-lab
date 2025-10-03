"""Configuration helpers for the toy world generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class WorldSettings:
    """Scalar configuration for the world setup."""

    num_agents: int = 1
    num_objects: int = 1
    num_locations: int = 2
    max_episode_len: int = 4
    allow_coref: bool = False
    names_mode: str = "closed"
    random_seed: Optional[int] = None


@dataclass
class ActionConfig:
    """Description of a single action available in the world."""

    name: str
    schema: List[str]
    pre: List[str] = field(default_factory=list)
    eff: List[str] = field(default_factory=list)


@dataclass
class RenderConfig:
    template_style: str = "grammar"
    use_quotes_for_spans: bool = False


@dataclass
class WorldConfig:
    """Container for the full world configuration."""

    world: WorldSettings = field(default_factory=WorldSettings)
    actions: List[ActionConfig] = field(default_factory=list)
    render: RenderConfig = field(default_factory=RenderConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldConfig":
        world_section = data.get("world", {})
        actions_section = data.get("actions", [])
        render_section = data.get("render", {})

        world_settings = WorldSettings(**world_section)
        actions = [ActionConfig(**action) for action in actions_section]
        render = RenderConfig(**render_section)
        return cls(world=world_settings, actions=actions, render=render)


def load_world_config(path: Any) -> WorldConfig:
    """Load a :class:`WorldConfig` from a YAML file."""

    text = Path(path).read_text(encoding="utf-8")
    data: Dict[str, Any]
    if yaml is not None:
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError("World configuration must be a mapping")
        data = loaded
    else:
        data = _parse_simple_yaml(text)
    return WorldConfig.from_dict(data)


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse a minimal subset of YAML used by the default configuration."""

    data: Dict[str, Any] = {}
    current_section: Optional[str] = None
    current_action: Optional[Dict[str, Any]] = None
    pending_list: Optional[List[Any]] = None

    lines = text.splitlines()
    for raw_line in lines:
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        if indent == 0 and stripped.endswith(":"):
            section = stripped[:-1]
            current_section = section
            pending_list = None
            current_action = None
            if section == "actions":
                data[section] = []
            else:
                data[section] = {}
            continue

        if current_section is None:
            continue

        if stripped.startswith("- "):
            item_value = stripped[2:].strip()
            if current_section == "actions" and indent <= 2:
                current_action = {}
                data.setdefault("actions", []).append(current_action)
                pending_list = None
                if item_value:
                    key, value = _split_key_value(item_value)
                    parsed = _parse_value(value)
                    current_action[key] = parsed
                    pending_list = parsed if isinstance(parsed, list) and value == "" else None
                continue
            if pending_list is not None:
                pending_list.append(_parse_scalar(item_value))
            continue

        key, value = _split_key_value(stripped)
        container: Dict[str, Any]
        if current_section == "actions":
            if current_action is None:
                continue
            container = current_action
        else:
            container = data[current_section]

        parsed_value = _parse_value(value)
        container[key] = parsed_value
        pending_list = parsed_value if isinstance(parsed_value, list) and value == "" else None

    return data


def _split_key_value(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"Invalid line in world config: {text}")
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def _parse_value(value: str) -> Any:
    if value == "":
        return []
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",") if item.strip()]
    return _parse_scalar(value)


def _parse_scalar(value: str) -> Any:
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


def iter_action_names(config: WorldConfig) -> Iterable[str]:
    """Yield names of actions declared in the configuration."""

    for action in config.actions:
        yield action.name
