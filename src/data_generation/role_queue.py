from __future__ import annotations

from typing import Dict, List, Tuple


class RoleQueue:
    """Maintains semantic roles awaiting realization."""

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
