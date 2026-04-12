from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParentRecord:
    parent_chunk_id: str
    doc_id: str
    source_block_ids: list[str]
    parent_content: str
    parent_wordpiece_count: int
    parent_content_hash: str
    section_path: list[str]
    page: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw_value: dict[str, Any]) -> ParentRecord:
        return cls(
            parent_chunk_id=str(raw_value.get("parent_chunk_id", "")),
            doc_id=str(raw_value.get("doc_id", "")),
            source_block_ids=[str(item) for item in raw_value.get("source_block_ids", [])],
            parent_content=str(raw_value.get("parent_content", "")),
            parent_wordpiece_count=int(raw_value.get("parent_wordpiece_count", 0)),
            parent_content_hash=str(raw_value.get("parent_content_hash", "")),
            section_path=[str(item) for item in raw_value.get("section_path", [])],
            page=str(raw_value["page"]) if raw_value.get("page") is not None else None,
            title=str(raw_value["title"]) if raw_value.get("title") is not None else None,
            metadata=dict(raw_value.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonParentStore:
    def __init__(self, processed_directory: Path) -> None:
        self._parents_directory = processed_directory / "parents"

    def save_records(self, doc_id: str, records: list[ParentRecord]) -> None:
        if not records:
            return
        self._parents_directory.mkdir(parents=True, exist_ok=True)
        payload = {record.parent_chunk_id: record.to_dict() for record in records}
        self._path_for_doc(doc_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, parent_chunk_id: str) -> ParentRecord | None:
        if not self._parents_directory.exists():
            return None
        for path in self._parents_directory.glob("*.parents.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            raw_record = payload.get(parent_chunk_id)
            if isinstance(raw_record, dict):
                return ParentRecord.from_dict(raw_record)
        return None


    def delete_records(self, doc_id: str) -> None:
        path = self._path_for_doc(doc_id)
        if path.exists():
            path.unlink()
    def _path_for_doc(self, doc_id: str) -> Path:
        safe_doc_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in doc_id)
        return self._parents_directory / f"{safe_doc_id}.parents.json"

