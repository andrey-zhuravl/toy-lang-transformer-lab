from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence


class DatasetWriter:
    """Aggregates generated pairs and writes task-specific JSONL files."""

    def __init__(self, output_dir: Path, tasks: Sequence[str]) -> None:
        self.output_dir = output_dir
        self.tasks = set(task.lower() for task in tasks)
        self.buffers: Dict[str, List[str]] = {task: [] for task in self.tasks}

    def add_entry(self, sentence_tokens: List[str], semantics: Dict[str, object]) -> None:
        sentence = " ".join(sentence_tokens)
        semantics_json = json.dumps(semantics, ensure_ascii=False, sort_keys=True)

        if "lm" in self.tasks:
            for index in range(1, len(sentence_tokens)):
                input_text = " ".join(sentence_tokens[:index])
                output_text = sentence_tokens[index]
                self.buffers["lm"].append(
                    json.dumps({"input": input_text, "output": output_text}, ensure_ascii=False)
                )

        if "parsing" in self.tasks:
            self.buffers["parsing"].append(
                json.dumps({"input": sentence, "output": json.loads(semantics_json)}, ensure_ascii=False)
            )

        if "nl2sem" in self.tasks:
            self.buffers["nl2sem"].append(
                json.dumps({"input": sentence, "output": json.loads(semantics_json)}, ensure_ascii=False)
            )

        if "sem2nl" in self.tasks:
            self.buffers["sem2nl"].append(
                json.dumps({"input": json.loads(semantics_json), "output": sentence}, ensure_ascii=False)
            )

    def flush(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for task, lines in self.buffers.items():
            path = self.output_dir / f"{task}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(line + "\n")
