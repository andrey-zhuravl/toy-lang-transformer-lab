"""World simulator: apply actions to world states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .ontology import ActionDefinition, Entity, EntityType
from .state import StateDelta, WorldState


@dataclass
class ActionInstance:
    """Concrete action with arguments."""

    definition: ActionDefinition
    arguments: Sequence[Entity]

    def argument_map(self) -> Dict[str, Entity]:
        return {role: entity for role, entity in zip(self.definition.argument_roles(), self.arguments)}

    def to_semantic(self) -> Dict[str, str]:
        semantic: Dict[str, str] = {"action": self.definition.name}
        for role, entity in self.argument_map().items():
            semantic_role = _role_to_key(role)
            semantic[semantic_role] = entity.name
        return semantic


class WorldSimulator:
    """A simple deterministic simulator that checks preconditions before applying effects."""

    def __init__(self, actions: Iterable[ActionDefinition]):
        self.actions: Dict[str, ActionDefinition] = {action.name: action for action in actions}

    def enumerate_applicable_actions(
        self,
        state: WorldState,
        entities: Mapping[str, Entity],
    ) -> List[ActionInstance]:
        """Enumerate all valid action instances for the given state."""

        instances: List[ActionInstance] = []
        entity_lists: Dict[EntityType, List[Entity]] = {
            EntityType.AGENT: [entity for entity in entities.values() if entity.etype == EntityType.AGENT],
            EntityType.OBJECT: [entity for entity in entities.values() if entity.etype == EntityType.OBJECT],
            EntityType.LOCATION: [entity for entity in entities.values() if entity.etype == EntityType.LOCATION],
        }

        for definition in self.actions.values():
            candidates = _cartesian_arguments(definition.schema, entity_lists)
            for candidate in candidates:
                instance = ActionInstance(definition, candidate)
                if self._check_preconditions(state, instance):
                    instances.append(instance)
        return instances

    def apply(self, state: WorldState, instance: ActionInstance) -> Tuple[WorldState, StateDelta]:
        if instance.definition.name not in self.actions:
            raise KeyError(f"Unknown action: {instance.definition.name}")

        if not self._check_preconditions(state, instance):
            raise ValueError("Preconditions violated for action")

        new_state = state.copy()
        argument_map = instance.argument_map()

        for effect in instance.definition.effects:
            self._apply_effect(new_state, argument_map, effect)

        # ensure that objects being carried move with their agents
        for agent_id, object_id in list(new_state.has):
            agent_entity = _get_entity_by_id(argument_map, agent_id)
            if agent_entity:
                new_state.set_location(object_id, new_state.loc.get(agent_id, new_state.loc.get(object_id, "")))

        delta = new_state.diff(state)
        self._validate_state(new_state)
        return new_state, delta

    def _check_preconditions(self, state: WorldState, instance: ActionInstance) -> bool:
        argument_map = instance.argument_map()
        for predicate in instance.definition.preconditions:
            if not _evaluate_predicate(predicate, state, argument_map):
                return False
        return True

    def _apply_effect(self, state: WorldState, argument_map: Mapping[str, Entity], effect: str) -> None:
        predicate, value = _split_predicate(effect)
        if predicate.startswith("loc("):
            entity_role = predicate[len("loc(") : -1]
            entity = argument_map[entity_role]
            location_value = _resolve_value(value, state, argument_map)
            state.set_location(entity.entity_id, location_value)
        elif predicate.startswith("has("):
            inner = predicate[len("has(") : -1]
            agent_role, object_role = [item.strip() for item in inner.split(",")]
            agent_entity = argument_map[agent_role]
            object_entity = argument_map[object_role]
            bool_value = value.lower() == "true"
            state.set_has(agent_entity.entity_id, object_entity.entity_id, bool_value)
            if bool_value:
                state.set_location(object_entity.entity_id, state.loc.get(agent_entity.entity_id, ""))
        else:
            raise ValueError(f"Unsupported effect predicate: {effect}")

    def _validate_state(self, state: WorldState) -> None:
        # Ensure no object is possessed by two agents simultaneously
        seen_objects: Dict[str, str] = {}
        for agent_id, object_id in state.has:
            if object_id in seen_objects and seen_objects[object_id] != agent_id:
                raise ValueError("Object possessed by multiple agents")
            seen_objects[object_id] = agent_id


def _cartesian_arguments(schema: Sequence[EntityType], pools: Mapping[EntityType, Sequence[Entity]]) -> List[Sequence[Entity]]:
    if not schema:
        return [[]]
    first, *rest = schema
    combinations: List[Sequence[Entity]] = []
    for entity in pools[first]:
        for combination in _cartesian_arguments(rest, pools):
            combinations.append([entity, *combination])
    return combinations


def _evaluate_predicate(predicate: str, state: WorldState, argument_map: Mapping[str, Entity]) -> bool:
    predicate, value = _split_predicate(predicate)
    if predicate.startswith("loc("):
        entity_role = predicate[len("loc(") : -1]
        entity = argument_map[entity_role]
        expected_location = _resolve_value(value, state, argument_map)
        return state.loc.get(entity.entity_id) == expected_location
    if predicate.startswith("has("):
        inner = predicate[len("has(") : -1]
        agent_role, object_role = [item.strip() for item in inner.split(",")]
        agent_entity = argument_map[agent_role]
        object_entity = argument_map[object_role]
        expected = value.lower() == "true"
        return (agent_entity.entity_id, object_entity.entity_id) in state.has if expected else (
            (agent_entity.entity_id, object_entity.entity_id) not in state.has
        )
    raise ValueError(f"Unsupported predicate: {predicate}")


def _split_predicate(raw: str) -> Tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Predicate must contain '=': {raw}")
    left, right = raw.split("=", 1)
    return left.strip(), right.strip()


def _resolve_value(value: str, state: WorldState, argument_map: Mapping[str, Entity]) -> str:
    if value.lower() in {"true", "false"}:
        return value.lower()
    if value.startswith("loc(") and value.endswith(")"):
        role = value[len("loc(") : -1]
        entity = argument_map[role]
        return state.loc.get(entity.entity_id, "")
    if value in argument_map:
        return argument_map[value].entity_id
    # constant location id or name
    for entity in argument_map.values():
        if entity.entity_id == value:
            return value
    return value


def _get_entity_by_id(argument_map: Mapping[str, Entity], entity_id: str) -> Optional[Entity]:
    for entity in argument_map.values():
        if entity.entity_id == entity_id:
            return entity
    return None


def _role_to_key(role: str) -> str:
    mapping = {
        EntityType.AGENT.value: "agent",
        EntityType.LOCATION.value: "location",
        EntityType.OBJECT.value: "object",
    }
    return mapping.get(role, role.lower())
