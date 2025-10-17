"""Episode generation for the toy world."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import WorldConfig
from .ontology import ActionDefinition, Entity, EntityType
from .simulator import ActionInstance, WorldSimulator
from .state import WorldState


@dataclass
class EpisodeStep:
    action: ActionInstance
    state_before: WorldState
    state_after: WorldState

    def to_semantic(self) -> Dict[str, object]:
        return {
            "before": self.state_before.to_json(),
            "after": self.state_after.to_json(),
            "action": self.action.to_semantic(),
        }


@dataclass
class Episode:
    entities: Dict[str, Entity]
    steps: List[EpisodeStep] = field(default_factory=list)
    initial_state: WorldState = field(default_factory=WorldState)

    def render_history(self, renderer) -> List[str]:  # pragma: no cover - convenience wrapper
        return [renderer.render_action(step.action) for step in self.steps]


def _build_entities(settings: WorldConfig) -> Dict[str, Entity]:
    entities: Dict[str, Entity] = {}

    def make_name(prefix: str, index: int) -> str:
        return f"{prefix}_{index+1}"

    for idx in range(settings.world.num_agents):
        entity_id = f"agent_{idx}"
        entities[entity_id] = Entity(entity_id=entity_id, etype=EntityType.AGENT, name=make_name("агент", idx))
    for idx in range(settings.world.num_objects):
        entity_id = f"object_{idx}"
        entities[entity_id] = Entity(entity_id=entity_id, etype=EntityType.OBJECT, name=make_name("предмет", idx))
    for idx in range(settings.world.num_locations):
        entity_id = f"location_{idx}"
        entities[entity_id] = Entity(entity_id=entity_id, etype=EntityType.LOCATION, name=make_name("место", idx))
    return entities


def _initial_state(config: WorldConfig, entities: Dict[str, Entity], rng: random.Random) -> WorldState:
    state = WorldState()
    locations = [entity for entity in entities.values() if entity.etype == EntityType.LOCATION]

    for entity in entities.values():
        if entity.etype in {EntityType.AGENT, EntityType.OBJECT}:
            location = rng.choice(locations)
            state.set_location(entity.entity_id, location.entity_id)
        else:
            state.set_location(entity.entity_id, entity.entity_id)
    return state


def _build_action_definitions(config: WorldConfig) -> List[ActionDefinition]:
    return [
        ActionDefinition.from_config(action.name, action.schema, action.pre, action.eff)
        for action in config.actions
    ]


def generate_episode(config: WorldConfig, seed: Optional[int] = None) -> Episode:
    """Generate a single episode by sampling a sequence of actions."""

    rng = random.Random(seed if seed is not None else config.world.random_seed)
    entities = _build_entities(config)
    simulator = WorldSimulator(_build_action_definitions(config))
    state = _initial_state(config, entities, rng)

    steps: List[EpisodeStep] = []
    for _ in range(config.world.max_episode_len):
        applicable = simulator.enumerate_applicable_actions(state, entities)
        if not applicable:
            break
        action_instance = rng.choice(applicable)
        new_state, _ = simulator.apply(state, action_instance)
        steps.append(EpisodeStep(action=action_instance, state_before=state, state_after=new_state))
        state = new_state

    return Episode(entities=entities, steps=steps, initial_state=steps[0].state_before if steps else state)


def generate_episodes(config: WorldConfig, num_episodes: int, seed: Optional[int] = None) -> List[Episode]:
    rng = random.Random(seed if seed is not None else config.world.random_seed)
    seeds = [rng.randint(0, 10_000_000) for _ in range(num_episodes)]
    return [generate_episode(config, seed=episode_seed) for episode_seed in seeds]
