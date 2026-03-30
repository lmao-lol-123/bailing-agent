from __future__ import annotations

from typing import Iterable
from uuid import uuid4

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.models import IngestedChunk, NormalizedDocument


class SemanticChunkingService:
    def __init__(self, embeddings: object) -> None:
        self._embeddings = embeddings
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )

    def chunk_documents(self, documents: Iterable[NormalizedDocument]) -> list[IngestedChunk]:
        source_documents = [
            Document(page_content=document.content, metadata=document.model_dump(exclude={"content"}))
            for document in documents
            if document.content.strip()
        ]
        if not source_documents:
            return []

        try:
            splitter = SemanticChunker(self._embeddings)
            split_documents = splitter.split_documents(source_documents)
        except Exception:
            split_documents = self._fallback_splitter.split_documents(source_documents)

        chunks: list[IngestedChunk] = []
        for split_document in split_documents:
            metadata = dict(split_document.metadata)
            chunks.append(
                IngestedChunk(
                    chunk_id=str(uuid4()),
                    doc_id=metadata["doc_id"],
                    source_name=metadata["source_name"],
                    source_uri_or_path=metadata["source_uri_or_path"],
                    page_or_section=metadata.get("page_or_section"),
                    content=split_document.page_content,
                    metadata=metadata,
                )
            )
        return chunks

