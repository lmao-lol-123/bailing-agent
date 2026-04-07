from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.src.core.config import Settings
from backend.src.core.models import Citation, DocumentSummary, IngestedChunk, RetrievalFilter, SourceType
from backend.src.core.text import build_snippet

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class VectorStoreService:
    def __init__(self, settings: Settings, embeddings: object) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._index_path = settings.faiss_index_directory / "default_index"
        self._vector_store = self._load_or_create_store()

    def _load_or_create_store(self) -> FAISS:
        if self._index_path.exists():
            return FAISS.load_local(
                folder_path=str(self._index_path),
                embeddings=self._embeddings,
                allow_dangerous_deserialization=True,
            )

        bootstrap = Document(
            page_content="FAISS bootstrap document",
            metadata={
                "doc_id": "__bootstrap__",
                "source_name": "bootstrap",
                "source_type": SourceType.TXT.value,
                "source_uri_or_path": "internal://bootstrap",
                "page_or_section": None,
                "page": None,
                "title": "bootstrap",
                "section_path": ["bootstrap"],
                "doc_type": SourceType.TXT.value,
                "updated_at": None,
            },
            id="__bootstrap__",
        )
        store = FAISS.from_documents([bootstrap], self._embeddings)
        store.delete(ids=["__bootstrap__"])
        self._save(store)
        return store

    def _save(self, store: FAISS | None = None) -> None:
        target = store or self._vector_store
        self._index_path.mkdir(parents=True, exist_ok=True)
        target.save_local(str(self._index_path))

    def add_chunks(self, chunks: list[IngestedChunk]) -> None:
        if not chunks:
            return

        documents = [
            Document(page_content=chunk.content, metadata=chunk.metadata, id=chunk.chunk_id)
            for chunk in chunks
        ]
        ids = [chunk.chunk_id for chunk in chunks]
        self._vector_store.add_documents(documents=documents, ids=ids)
        self._save()

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[Document]:
        top_k = k or self._settings.retriever_top_k
        fetch_k = max(top_k * self._settings.rerank_candidate_multiplier, top_k)
        scored_documents = self._vector_store.similarity_search_with_score(query=query, k=fetch_k)

        filtered_scored_documents = [
            (document, float(distance))
            for document, distance in scored_documents
            if document.metadata.get("doc_id") != "__bootstrap__"
            and self._matches_metadata_filter(document.metadata or {}, metadata_filter)
        ]
        reranked_documents = self._rerank_documents(query=query, scored_documents=filtered_scored_documents)
        return [document for document, _score in reranked_documents[:top_k]]

    def list_documents(self) -> list[DocumentSummary]:
        metadatas = list(self._vector_store.docstore._dict.values())

        buckets: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"source_name": "", "source_type": "", "source_uri_or_path": "", "count": 0, "sections": set()}
        )

        for entry in metadatas:
            metadata = entry.metadata if hasattr(entry, "metadata") else None
            if metadata is None or metadata.get("doc_id") == "__bootstrap__":
                continue
            doc_id = str(metadata["doc_id"])
            bucket = buckets[doc_id]
            bucket["source_name"] = metadata.get("source_name", "")
            bucket["source_type"] = metadata.get("doc_type") or metadata.get("source_type", "")
            bucket["source_uri_or_path"] = metadata.get("source_uri_or_path", "")
            bucket["count"] = int(bucket["count"]) + 1

            location = self._format_location(metadata)
            if location:
                bucket["sections"].add(location)

        summaries: list[DocumentSummary] = []
        for doc_id, bucket in buckets.items():
            source_type_value = str(bucket["source_type"] or SourceType.TXT.value)
            summaries.append(
                DocumentSummary(
                    doc_id=doc_id,
                    source_name=str(bucket["source_name"]),
                    source_type=SourceType(source_type_value),
                    source_uri_or_path=str(bucket["source_uri_or_path"]),
                    chunk_count=int(bucket["count"]),
                    page_or_sections=sorted(bucket["sections"]),
                )
            )
        return sorted(summaries, key=lambda item: item.source_name)

    @staticmethod
    def build_citations(documents: list[Document]) -> list[Citation]:
        citations: list[Citation] = []
        seen_sources: set[tuple[str, str | None, tuple[str, ...]]] = set()

        for document in documents:
            metadata = document.metadata or {}
            source_uri_or_path = str(metadata.get("source_uri_or_path", ""))
            source_name = str(metadata.get("source_name", "unknown"))
            page = metadata.get("page") or metadata.get("page_or_section")
            section_path = VectorStoreService._coerce_section_path(metadata.get("section_path"))
            dedupe_key = (
                source_uri_or_path or source_name or str(metadata.get("doc_id", "")),
                str(page) if page else None,
                tuple(section_path),
            )
            if dedupe_key in seen_sources:
                continue
            seen_sources.add(dedupe_key)

            doc_type = metadata.get("doc_type") or metadata.get("source_type")
            citations.append(
                Citation(
                    index=len(citations) + 1,
                    doc_id=str(metadata.get("doc_id", "")),
                    source_name=source_name,
                    source_uri_or_path=source_uri_or_path,
                    page_or_section=str(metadata.get("page_or_section")) if metadata.get("page_or_section") else None,
                    page=str(page) if page else None,
                    title=str(metadata.get("title")) if metadata.get("title") else None,
                    section_path=section_path,
                    doc_type=SourceType(doc_type) if doc_type else None,
                    updated_at=VectorStoreService._parse_datetime(metadata.get("updated_at")),
                    snippet=build_snippet(document.page_content),
                )
            )
        return citations

    def _rerank_documents(
        self,
        *,
        query: str,
        scored_documents: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        query_tokens = self._tokenize(query)
        lexical_weight = self._settings.rerank_lexical_weight
        vector_weight = 1.0 - lexical_weight

        reranked: list[tuple[int, float, Document]] = []
        for rank, (document, distance) in enumerate(scored_documents):
            vector_score = 1.0 / (1.0 + max(distance, 0.0))
            lexical_score = self._lexical_score(query_tokens=query_tokens, document=document)
            final_score = vector_weight * vector_score + lexical_weight * lexical_score
            reranked.append((rank, final_score, document))

        reranked.sort(key=lambda item: (-item[1], item[0]))
        return [(document, final_score) for rank, final_score, document in reranked]

    def _lexical_score(self, *, query_tokens: set[str], document: Document) -> float:
        if not query_tokens:
            return 0.0

        metadata = document.metadata or {}
        section_text = " ".join(self._coerce_section_path(metadata.get("section_path")))
        title = str(metadata.get("title") or "")
        content_tokens = self._tokenize(f"{title} {section_text} {document.page_content}")
        overlap = len(query_tokens & content_tokens)
        return overlap / len(query_tokens)

    @staticmethod
    def _matches_metadata_filter(metadata: dict[str, Any], metadata_filter: RetrievalFilter | None) -> bool:
        if metadata_filter is None:
            return True

        if metadata_filter.doc_ids and str(metadata.get("doc_id", "")) not in set(metadata_filter.doc_ids):
            return False

        if metadata_filter.source_names:
            source_name = str(metadata.get("source_name", ""))
            source_uri_or_path = str(metadata.get("source_uri_or_path", ""))
            if source_name not in metadata_filter.source_names and source_uri_or_path not in metadata_filter.source_names:
                return False

        if metadata_filter.doc_types:
            doc_type = metadata.get("doc_type") or metadata.get("source_type")
            allowed_doc_types = {item.value for item in metadata_filter.doc_types}
            if str(doc_type) not in allowed_doc_types:
                return False

        if metadata_filter.pages:
            page = metadata.get("page") or metadata.get("page_or_section")
            if str(page) not in set(metadata_filter.pages):
                return False

        if metadata_filter.section_path_prefix:
            section_path = VectorStoreService._coerce_section_path(metadata.get("section_path"))
            prefix = metadata_filter.section_path_prefix
            if section_path[: len(prefix)] != prefix:
                return False

        if metadata_filter.title_contains:
            title = str(metadata.get("title") or metadata.get("source_name") or "").lower()
            if metadata_filter.title_contains.lower() not in title:
                return False

        updated_at = VectorStoreService._parse_datetime(metadata.get("updated_at"))
        if metadata_filter.updated_after and (updated_at is None or updated_at < metadata_filter.updated_after):
            return False
        if metadata_filter.updated_before and (updated_at is None or updated_at > metadata_filter.updated_before):
            return False

        return True

    @staticmethod
    def _coerce_section_path(section_path: Any) -> list[str]:
        if isinstance(section_path, list):
            return [str(item) for item in section_path if str(item).strip()]
        if isinstance(section_path, str) and section_path.strip():
            return [item.strip() for item in section_path.split("/") if item.strip()]
        return []

    @staticmethod
    def _parse_datetime(raw_value: Any) -> datetime | None:
        if isinstance(raw_value, datetime):
            return raw_value
        if isinstance(raw_value, str) and raw_value.strip():
            try:
                return datetime.fromisoformat(raw_value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _format_location(metadata: dict[str, Any]) -> str | None:
        section_path = VectorStoreService._coerce_section_path(metadata.get("section_path"))
        page = metadata.get("page") or metadata.get("page_or_section")
        if section_path and page:
            return f"{' > '.join(section_path)} | p{page}"
        if section_path:
            return " > ".join(section_path)
        if page:
            return str(page)
        return None

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token.lower() for token in _TOKEN_PATTERN.findall(text)}
