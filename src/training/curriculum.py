from __future__ import annotations

from typing import Dict


def compute_curriculum_fraction(epoch: int, total_epochs: int, curriculum_cfg: Dict[str, object]) -> float:
    if not curriculum_cfg.get("enabled", False):
        return 1.0
    schedule = curriculum_cfg.get("schedule", "linear")
    start = float(curriculum_cfg.get("start_fraction", 0.2))
    end = float(curriculum_cfg.get("end_fraction", 1.0))
    if schedule == "linear":
        if total_epochs <= 1:
            return end
        return start + (end - start) * (epoch / (total_epochs - 1))
    if schedule == "exponential":
        progress = epoch / max(total_epochs - 1, 1)
        return start * (end / max(start, 1e-3)) ** progress
    if schedule == "manual":
        boundaries = curriculum_cfg.get("manual_boundaries", [])
        fraction = start
        for entry in boundaries:
            boundary_epoch, boundary_fraction = entry
            if epoch >= boundary_epoch:
                fraction = boundary_fraction
        return fraction
    return end
