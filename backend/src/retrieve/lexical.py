from __future__ import annotations

import re
from dataclasses import dataclass
from math import log
from typing import Any, Callable

from langchain_core.documents import Document

from backend.src.core.config import Settings
from backend.src.core.models import RetrievalFilter
from backend.src.retrieve.index_manager import ActiveChunkRecord, IndexManager
from backend.src.retrieve.rerank import RetrievedCandidate

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
except ImportError:
    class _BM25Okapi:  # pragma: no cover - compatibility fallback when dependency is unavailable
        def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
            self._corpus = corpus
            self._k1 = k1
            self._b = b
            self._doc_lengths = [len(document) for document in corpus]
            self._avgdl = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
            self._term_frequencies: list[dict[str, int]] = []
            self._document_frequencies: dict[str, int] = {}
            for document in corpus:
                frequencies: dict[str, int] = {}
                for token in document:
                    frequencies[token] = frequencies.get(token, 0) + 1
                self._term_frequencies.append(frequencies)
                for token in frequencies:
                    self._document_frequencies[token] = self._document_frequencies.get(token, 0) + 1
            corpus_size = len(corpus)
            self._idf = {
                token: log(1.0 + (corpus_size - freq + 0.5) / (freq + 0.5))
                for token, freq in self._document_frequencies.items()
            }

        def get_scores(self, query_tokens: list[str]) -> list[float]:
            scores: list[float] = []
            for index, frequencies in enumerate(self._term_frequencies):
                score = 0.0
                doc_length = self._doc_lengths[index] if index < len(self._doc_lengths) else 0
                norm = 1.0 - self._b + self._b * (doc_length / self._avgdl) if self._avgdl else 1.0
                for token in query_tokens:
                    frequency = frequencies.get(token, 0)
                    if frequency <= 0:
                        continue
                    idf = self._idf.get(token, 0.0)
                    numerator = frequency * (self._k1 + 1.0)
                    denominator = frequency + self._k1 * norm
                    score += idf * (numerator / denominator)
                scores.append(score)
            return scores


@dataclass(frozen=True)
class BM25CorpusSnapshot:
    version_token: str
    records: list[ActiveChunkRecord]
    tokenized_corpus: list[list[str]]
    bm25: _BM25Okapi


class LexicalRetrievalService:
    def __init__(
        self,
        *,
        settings: Settings,
        index_manager: IndexManager,
        metadata_matcher: Callable[[dict[str, Any], RetrievalFilter | None], bool],
    ) -> None:
        self._settings = settings
        self._index_manager = index_manager
        self._metadata_matcher = metadata_matcher
        self._snapshot: BM25CorpusSnapshot | None = None

    def search(
        self,
        *,
        query: str,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
        record_matcher: Callable[[ActiveChunkRecord], bool] | None = None,
    ) -> list[RetrievedCandidate]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        snapshot = self._ensure_snapshot()
        if not snapshot.records:
            return []

        scores = snapshot.bm25.get_scores(query_tokens)
        scored_candidates: list[tuple[int, float, ActiveChunkRecord]] = []
        for index, record in enumerate(snapshot.records):
            if not self._metadata_matcher(record.metadata, metadata_filter):
                continue
            if record_matcher is not None and not record_matcher(record):
                continue
            score = float(scores[index]) if index < len(scores) else 0.0
            if score <= 0.0:
                continue
            scored_candidates.append((index, score, record))

        scored_candidates.sort(key=lambda item: (-item[1], item[0]))
        results: list[RetrievedCandidate] = []
        for rank, (_, score, record) in enumerate(scored_candidates[:k], start=1):
            results.append(
                RetrievedCandidate(
                    chunk_id=record.chunk_id,
                    document=record.to_document(),
                    retrieval_sources=("lexical",),
                    lexical_rank=rank,
                    lexical_score=score,
                )
            )
        return results

    def _ensure_snapshot(self) -> BM25CorpusSnapshot:
        manifest = self._index_manager.get_manifest()
        version_token = str(manifest.get("updated_at") or "")
        if self._snapshot is not None and self._snapshot.version_token == version_token:
            return self._snapshot

        records = self._index_manager.list_active_chunk_records()
        tokenized_corpus = [self._tokenize(record.retrieval_text) for record in records]
        bm25 = _BM25Okapi(tokenized_corpus) if tokenized_corpus else _BM25Okapi([[]])
        self._snapshot = BM25CorpusSnapshot(
            version_token=version_token,
            records=records,
            tokenized_corpus=tokenized_corpus,
            bm25=bm25,
        )
        return self._snapshot

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_PATTERN.findall(text)]
