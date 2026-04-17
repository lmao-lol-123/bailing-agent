from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class EvalSample(BaseModel):
    question: str
    expected_source_name: str
    expected_snippet: str
    reference_answer: str | None = None
    allow_uncertain: bool = False


def load_eval_samples(path: Path) -> list[EvalSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [EvalSample.model_validate(item) for item in payload]
