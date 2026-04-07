from __future__ import annotations

import re
from collections.abc import Iterable


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value).strip()


def build_snippet(value: str, limit: int = 180) -> str:
    text = normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def is_probably_garbled(value: str, threshold: float) -> bool:
    text = normalize_text(value)
    if not text:
        return True

    suspicious_chars = sum(
        1
        for char in text
        if not (char.isalnum() or char.isspace() or char in ".,;:!?-_/#()[]{}<>@'")
    )
    ratio = suspicious_chars / max(len(text), 1)
    return ratio >= threshold


def merge_texts(values: Iterable[str]) -> str:
    return "\n".join(part for part in values if part).strip()

