"""Ontology definitions for the toy world."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List


class EntityType(str, Enum):
    """Types of entities in the world."""

    AGENT = "АГЕНТ"
    OBJECT = "ПРЕДМЕТ"
    LOCATION = "ЛОКАЦИЯ"

    @classmethod
    def from_token(cls, token: str) -> "EntityType":
        for member in cls:
            if member.value == token or member.name == token:
                return member
        raise KeyError(f"Unknown entity type token: {token}")


@dataclass
class Entity:
    """A single entity that exists in the world."""

    entity_id: str
    etype: EntityType
    name: str

    def to_json(self) -> Dict[str, str]:
        return {"id": self.entity_id, "type": self.etype.value, "name": self.name}


@dataclass
class ActionDefinition:
    """A canonical action that can be instantiated in the world."""

    name: str
    schema: List[EntityType]
    preconditions: List[str]
    effects: List[str]

    @classmethod
    def from_config(cls, name: str, schema: Iterable[str], pre: Iterable[str], eff: Iterable[str]) -> "ActionDefinition":
        return cls(
            name=name,
            schema=[EntityType.from_token(token) for token in schema],
            preconditions=list(pre),
            effects=list(eff),
        )

    def argument_roles(self) -> List[str]:
        return [entity_type.value for entity_type in self.schema]
