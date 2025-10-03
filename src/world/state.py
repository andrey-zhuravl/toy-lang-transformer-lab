"""World state representation and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class StateDelta:
    """Difference between two states."""

    loc_changes: Dict[str, str] = field(default_factory=dict)
    has_added: Set[Tuple[str, str]] = field(default_factory=set)
    has_removed: Set[Tuple[str, str]] = field(default_factory=set)

    def to_json(self) -> Dict[str, object]:
        return {
            "loc": dict(self.loc_changes),
            "has": {
                "added": sorted([list(pair) for pair in self.has_added]),
                "removed": sorted([list(pair) for pair in self.has_removed]),
            },
        }


@dataclass
class WorldState:
    """Mutable state of the world."""

    loc: Dict[str, str] = field(default_factory=dict)
    has: Set[Tuple[str, str]] = field(default_factory=set)

    def copy(self) -> "WorldState":
        return WorldState(loc=dict(self.loc), has=set(self.has))

    def to_json(self) -> Dict[str, object]:
        return {
            "loc": dict(self.loc),
            "has": sorted([list(pair) for pair in self.has]),
        }

    @classmethod
    def from_json(cls, data: Dict[str, object]) -> "WorldState":
        loc = {str(key): str(value) for key, value in (data.get("loc") or {}).items()}
        has = {tuple(item) for item in data.get("has", [])}
        return cls(loc=loc, has={(str(a), str(b)) for a, b in has})

    def diff(self, previous: Optional["WorldState"]) -> StateDelta:
        if previous is None:
            return StateDelta(loc_changes=dict(self.loc), has_added=set(self.has), has_removed=set())

        delta = StateDelta()
        for entity_id, location_id in self.loc.items():
            if previous.loc.get(entity_id) != location_id:
                delta.loc_changes[entity_id] = location_id

        previous_entities = set(previous.loc)
        for entity_id in previous_entities - set(self.loc):
            delta.loc_changes[entity_id] = self.loc.get(entity_id, "")

        for pair in self.has:
            if pair not in previous.has:
                delta.has_added.add(pair)
        for pair in previous.has:
            if pair not in self.has:
                delta.has_removed.add(pair)
        return delta

    def agents(self) -> Iterable[str]:
        for entity_id, location in self.loc.items():
            if entity_id.startswith("agent"):
                yield entity_id

    def objects(self) -> Iterable[str]:
        for entity_id, location in self.loc.items():
            if entity_id.startswith("object"):
                yield entity_id

    def locations(self) -> Iterable[str]:
        seen: Set[str] = set()
        for location in self.loc.values():
            if location.startswith("location") and location not in seen:
                seen.add(location)
                yield location

    def set_location(self, entity_id: str, location_id: str) -> None:
        self.loc[entity_id] = location_id

    def set_has(self, agent_id: str, object_id: str, value: bool) -> None:
        pair = (agent_id, object_id)
        if value:
            # Ensure uniqueness: an object can be held by only one agent
            for existing_pair in list(self.has):
                if existing_pair[1] == object_id:
                    self.has.discard(existing_pair)
            self.has.add(pair)
        else:
            self.has.discard(pair)

    def agent_inventory(self, agent_id: str) -> List[str]:
        return [object_id for (agent, object_id) in self.has if agent == agent_id]

    def holder_of(self, object_id: str) -> Optional[str]:
        for agent_id, obj in self.has:
            if obj == object_id:
                return agent_id
        return None
