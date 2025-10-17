"""Semantic (JSON) renderer for world structures."""

from __future__ import annotations

from typing import Dict

from ..simulator import ActionInstance
from ..state import WorldState


def render_action_sem(action: ActionInstance) -> Dict[str, object]:
    return {
        "action": action.definition.name,
        "roles": {
            role: entity.name for role, entity in action.argument_map().items()
        },
    }


def render_state_sem(state: WorldState) -> Dict[str, object]:
    return state.to_json()
