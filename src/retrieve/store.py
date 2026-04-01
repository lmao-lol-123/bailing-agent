from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.core.config import Settings
from src.core.models import Citation, DocumentSummary, IngestedChunk, SourceType
from src.core.text import build_snippet


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

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        top_k = k or self._settings.retriever_top_k
        documents = self._vector_store.similarity_search(query=query, k=top_k)
        return [document for document in documents if document.metadata.get("doc_id") != "__bootstrap__"]

    def list_documents(self) -> list[DocumentSummary]:
        metadatas = list(self._vector_store.docstore._dict.values())

        buckets: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"source_name": "", "source_type": "", "source_uri_or_path": "", "count": 0, "sections": set()}
        )

        for entry in metadatas:
            metadata = entry.metadata if hasattr(entry, "metadata") else None
            if metadata is None:
                continue
            if metadata.get("doc_id") == "__bootstrap__":
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
