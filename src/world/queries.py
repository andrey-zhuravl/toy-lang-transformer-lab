"""Question generators for the toy world."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .episode import Episode
from .renderers.nl_renderer import NLRenderer


@dataclass
class Question:
    text: str
    answer: str
    qtype: str


class QuestionGenerator:
    def __init__(self, renderer: Optional[NLRenderer] = None, seed: Optional[int] = None):
        self.renderer = renderer or NLRenderer()

    def questions_for_episode(self, episode: Episode) -> List[Question]:
        questions: List[Question] = []
        history_text = "\n".join(self.renderer.render_action(step.action) for step in episode.steps)
        final_state = episode.steps[-1].state_after if episode.steps else episode.initial_state

        # where_is questions
        for entity_id, entity in episode.entities.items():
            if entity_id.startswith("object"):
                location_id = final_state.loc.get(entity_id, "")
                location_name = episode.entities.get(location_id, entity).name if location_id in episode.entities else location_id
                question_text = f"{history_text}\nВопрос: где {entity.name}?" if history_text else f"Вопрос: где {entity.name}?"
                questions.append(Question(text=question_text, answer=location_name, qtype="where"))

        # who_has questions
        for object_id, entity in episode.entities.items():
            if not object_id.startswith("object"):
                continue
            holder_id = final_state.holder_of(object_id)
            answer = episode.entities[holder_id].name if holder_id else "никто"
            question_text = f"{history_text}\nВопрос: у кого {entity.name}?" if history_text else f"Вопрос: у кого {entity.name}?"
            questions.append(Question(text=question_text, answer=answer, qtype="who_has"))

        # was_at questions (history)
        for step in episode.steps:
            agent = step.action.argument_map().get("АГЕНТ")
            location = step.action.argument_map().get("ЛОКАЦИЯ")
            if agent and location:
                question = f"{history_text}\nВопрос: был ли {agent.name} в {location.name}?" if history_text else f"Вопрос: был ли {agent.name} в {location.name}?"
                questions.append(Question(text=question, answer="да", qtype="was_at"))
        return questions


def build_qa_examples(episodes: Iterable[Episode], seed: Optional[int] = None) -> List[Dict[str, object]]:
    generator = QuestionGenerator(seed=seed)
    examples: List[Dict[str, object]] = []
    for episode in episodes:
        for question in generator.questions_for_episode(episode):
            examples.append({
                "input": question.text.strip(),
                "output": question.answer,
                "meta": {"qtype": question.qtype},
            })
    return examples


def build_parsing_examples(episodes: Iterable[Episode]) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    renderer = NLRenderer()
    for episode in episodes:
        for step in episode.steps:
            text = renderer.render_action(step.action)
            examples.append({
                "input": text,
                "output": step.action.to_semantic(),
                "meta": {"action": step.action.definition.name},
            })
    return examples


def build_sem2nl_examples(episodes: Iterable[Episode]) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    renderer = NLRenderer()
    for episode in episodes:
        for step in episode.steps:
            examples.append({
                "input": step.action.to_semantic(),
                "output": renderer.render_action(step.action),
                "meta": {"action": step.action.definition.name},
            })
    return examples


def build_effect_examples(episodes: Iterable[Episode]) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    renderer = NLRenderer()
    for episode in episodes:
        history_lines = [renderer.render_action(step.action) for step in episode.steps]
        name_lookup = {entity_id: entity.name for entity_id, entity in episode.entities.items()}
        for idx, step in enumerate(episode.steps):
            history_prefix = "\n".join(history_lines[:idx])
            delta = step.state_after.diff(step.state_before)
            loc_changes = {
                name_lookup.get(entity_id, entity_id): name_lookup.get(location_id, location_id)
                for entity_id, location_id in delta.loc_changes.items()
            }
            examples.append({
                "input": (history_prefix + "\n" if history_prefix else "") + f"Действие: {renderer.render_action(step.action)}",
                "output": {
                    "loc": loc_changes,
                    "has": [
                        {
                            "agent": name_lookup.get(agent, agent),
                            "object": name_lookup.get(obj, obj),
                            "value": True,
                        }
                        for agent, obj in sorted(delta.has_added)
                    ],
                    "has_removed": [
                        {
                            "agent": name_lookup.get(agent, agent),
                            "object": name_lookup.get(obj, obj),
                            "value": False,
                        }
                        for agent, obj in sorted(delta.has_removed)
                    ],
                },
                "meta": {"action": step.action.definition.name},
            })
    return examples


def build_lm_examples(episodes: Iterable[Episode]) -> List[Dict[str, object]]:
    renderer = NLRenderer()
    examples: List[Dict[str, object]] = []
    for episode in episodes:
        lines = [renderer.render_action(step.action) for step in episode.steps]
        if lines:
            examples.append({"input": "\n".join(lines), "output": "", "meta": {"task": "lm"}})
    return examples
