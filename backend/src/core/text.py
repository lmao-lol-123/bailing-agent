from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

WHITESPACE_RE = re.compile(r"\s+")
_SEARCH_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.%/-]+|[\u4e00-\u9fff]+")
_CLAUSE_ID_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+){1,3})(?:条)?")
_TABLE_ID_PATTERN = re.compile(r"表\s*(\d+(?:\.\d+)+)")
_ENGLISH_ALIAS_PATTERN = re.compile(r"[\u4e00-\u9fff]+\s+([A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*)+)")
_NUMERIC_ANCHOR_PATTERN = re.compile(r"\d+(?:\.\d+)?(?:%|MPa|天)", re.IGNORECASE)
_PSEUDO_TABLE_ROW_PATTERN = re.compile(
    r"^\s*(?:表\s*\d+(?:\.\d+)+\s+)?([\u4e00-\u9fffA-Za-z]+)\s+(\d+(?:\.\d+)?)\s*$"
)
_NORMATIVE_PATTERN = re.compile(r"不应|不得|应|宜|可")


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


def tokenize_search_text(value: str) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []

    tokens: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        normalized_token = token.strip().lower()
        if not normalized_token or normalized_token in seen:
            return
        seen.add(normalized_token)
        tokens.append(normalized_token)

    anchors = extract_regulation_anchors(text)
    clause_id = str(anchors.get("clause_id") or "")
    table_id = str(anchors.get("table_id") or "")
    if clause_id:
        add(clause_id)
    if table_id:
        add(table_id)
        add(f"表{table_id}")

    for match in _SEARCH_TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        if re.fullmatch(r"[A-Za-z0-9_.%/-]+", token):
            add(token)
            continue
        if len(token) <= 4:
            add(token)
            continue
        for size in (2, 3):
            for index in range(0, max(len(token) - size + 1, 0)):
                add(token[index : index + size])
        if len(token) <= 8:
            add(token)
    return tokens


def extract_regulation_anchors(value: str) -> dict[str, Any]:
    text = normalize_text(value)
    clause_match = _CLAUSE_ID_PATTERN.search(text)
    table_match = _TABLE_ID_PATTERN.search(text)
    english_match = _ENGLISH_ALIAS_PATTERN.search(text)
    numeric_terms = [match.group(0) for match in _NUMERIC_ANCHOR_PATTERN.finditer(text)]

    pseudo_row_key: str | None = None
    pseudo_row_value: str | None = None
    for line in value.splitlines():
        row_match = _PSEUDO_TABLE_ROW_PATTERN.match(normalize_text(line))
        if row_match is None:
            continue
        pseudo_row_key = row_match.group(1)
        pseudo_row_value = row_match.group(2)
        if pseudo_row_value not in numeric_terms:
            numeric_terms.append(pseudo_row_value)
        break

    english_alias = normalize_text(english_match.group(1)) if english_match else None
    unique_numeric_terms: list[str] = []
    seen_numeric_terms: set[str] = set()
    for term in numeric_terms:
        normalized_term = normalize_text(term)
        if not normalized_term or normalized_term in seen_numeric_terms:
            continue
        seen_numeric_terms.add(normalized_term)
        unique_numeric_terms.append(normalized_term)

    regulation_anchor_terms = [
        term
        for term in [
            clause_match.group(1) if clause_match else None,
            table_match.group(1) if table_match else None,
            english_alias,
            pseudo_row_key,
            *unique_numeric_terms,
        ]
        if term
    ]

    return {
        "clause_id": clause_match.group(1) if clause_match else None,
        "table_id": table_match.group(1) if table_match else None,
        "english_alias": english_alias,
        "is_normative_clause": bool(_NORMATIVE_PATTERN.search(text)),
        "has_numeric_anchor": bool(unique_numeric_terms),
        "numeric_anchor_terms": unique_numeric_terms,
        "pseudo_table_row_key": pseudo_row_key,
        "pseudo_table_row_value": pseudo_row_value,
        "is_regulation_anchor": bool(
            clause_match
            or table_match
            or english_alias
            or unique_numeric_terms
            or _NORMATIVE_PATTERN.search(text)
        ),
        "regulation_anchor_terms": regulation_anchor_terms,
    }
