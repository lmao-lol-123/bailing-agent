from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Any

from langchain_core.documents import Document

from src.core.config import Settings
from src.core.models import Citation, DocumentSummary, IngestedChunk, SourceType
from src.core.text import build_snippet


class VectorStoreService:
    def __init__(self, settings: Settings, embeddings: object) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._memory_chunks: list[tuple[list[float], IngestedChunk]] = []
        self._vector_store = None
        self._backend = "memory"

        try:
            from langchain_chroma import Chroma

            self._vector_store = Chroma(
                collection_name=settings.chroma_collection_name,
                embedding_function=embeddings,
                persist_directory=str(settings.chroma_persist_directory),
            )
            self._backend = "chroma"
        except Exception:
            self._vector_store = None

    def add_chunks(self, chunks: list[IngestedChunk]) -> None:
        if not chunks:
            return

        if self._backend == "chroma" and self._vector_store is not None:
            documents = [
                Document(page_content=chunk.content, metadata=chunk.metadata, id=chunk.chunk_id)
                for chunk in chunks
            ]
            ids = [chunk.chunk_id for chunk in chunks]
            self._vector_store.add_documents(documents=documents, ids=ids)
            return

        embeddings = self._embeddings.embed_documents([chunk.content for chunk in chunks])
        for vector, chunk in zip(embeddings, chunks, strict=True):
            self._memory_chunks.append((vector, chunk))

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        top_k = k or self._settings.retriever_top_k
        if self._backend == "chroma" and self._vector_store is not None:
            return self._vector_store.similarity_search(query=query, k=top_k)

        query_vector = self._embeddings.embed_query(query)
        scored = []
        for vector, chunk in self._memory_chunks:
            scored.append((self._cosine_similarity(query_vector, vector), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            Document(page_content=chunk.content, metadata=chunk.metadata, id=chunk.chunk_id)
            for _, chunk in scored[:top_k]
        ]

    def list_documents(self) -> list[DocumentSummary]:
        if self._backend == "chroma" and self._vector_store is not None:
            collection = self._vector_store.get(include=["metadatas"])
            metadatas = collection.get("metadatas", [])
        else:
            metadatas = [chunk.metadata for _, chunk in self._memory_chunks]

        buckets: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"source_name": "", "source_type": "", "source_uri_or_path": "", "count": 0, "sections": set()}
        )

        for metadata in metadatas:
            if metadata is None:
                continue
            doc_id = str(metadata["doc_id"])
            bucket = buckets[doc_id]
            bucket["source_name"] = metadata.get("source_name", "")
            bucket["source_type"] = metadata.get("source_type", "")
            bucket["source_uri_or_path"] = metadata.get("source_uri_or_path", "")
            bucket["count"] = int(bucket["count"]) + 1
            if metadata.get("page_or_section"):
                bucket["sections"].add(str(metadata["page_or_section"]))

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
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata or {}
            citations.append(
                Citation(
                    index=index,
                    doc_id=str(metadata.get("doc_id", "")),
                    source_name=str(metadata.get("source_name", "unknown")),
                    source_uri_or_path=str(metadata.get("source_uri_or_path", "")),
                    page_or_section=str(metadata.get("page_or_section")) if metadata.get("page_or_section") else None,
                    snippet=build_snippet(document.page_content),
                )
            )
        return citations

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = sqrt(sum(a * a for a in left)) or 1.0
        right_norm = sqrt(sum(b * b for b in right)) or 1.0
        return numerator / (left_norm * right_norm)
