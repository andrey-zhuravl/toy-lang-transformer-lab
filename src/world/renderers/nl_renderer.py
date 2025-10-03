"""Natural language renderer for world actions and states."""

from __future__ import annotations

from typing import Iterable, List

from ..simulator import ActionInstance
from ..state import WorldState


class NLRenderer:
    """Very small collection of templates used for rendering text."""

    def __init__(self, use_quotes_for_spans: bool = False):
        self.use_quotes_for_spans = use_quotes_for_spans

    def render_action(self, action: ActionInstance) -> str:
        agent = action.argument_map().get("АГЕНТ")
        obj = action.argument_map().get("ПРЕДМЕТ")
        location = action.argument_map().get("ЛОКАЦИЯ")

        agent_name = self._maybe_quote(agent.name) if agent else ""
        object_name = self._maybe_quote(obj.name) if obj else ""
        location_name = self._maybe_quote(location.name) if location else ""

        if action.definition.name == "идти" and agent and location:
            return f"{agent_name} идти в {location_name}"
        if action.definition.name == "взять" and agent and obj:
            return f"{agent_name} взять {object_name}"
        if action.definition.name == "положить" and agent and obj:
            return f"{agent_name} положить {object_name}"
        parts = [agent_name, action.definition.name]
        if object_name:
            parts.append(object_name)
        if location_name:
            parts.append(location_name)
        return " ".join(part for part in parts if part)

    def render_state_facts(self, state: WorldState, name_lookup) -> List[str]:
        facts: List[str] = []
        for entity_id, location in state.loc.items():
            if entity_id.startswith("object"):
                facts.append(f"{name_lookup(entity_id)} в {name_lookup(location)}")
        for agent_id, object_id in sorted(state.has):
            facts.append(f"{name_lookup(agent_id)} имеет {name_lookup(object_id)}")
        return facts

    def render_episode(self, steps: Iterable[ActionInstance]) -> List[str]:
        return [self.render_action(step) for step in steps]

    def _maybe_quote(self, text: str) -> str:
        if not self.use_quotes_for_spans:
            return text
        return f"«{text}»"


def render_action(action: ActionInstance, use_quotes_for_spans: bool = False) -> str:
    return NLRenderer(use_quotes_for_spans=use_quotes_for_spans).render_action(action)
