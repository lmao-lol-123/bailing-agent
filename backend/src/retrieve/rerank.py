from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from backend.src.core.config import Settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


@dataclass(frozen=True)
class RetrievedCandidate:
    chunk_id: str
    document: Document
    retrieval_sources: tuple[str, ...] = ()
    query_variant_ids: tuple[str, ...] = ()
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    fusion_score: float = 0.0
    query_fusion_score: float | None = None
    rerank_score: float | None = None

    def best_rank(self) -> int:
        ranks = [rank for rank in (self.dense_rank, self.lexical_rank) if rank is not None]
        return min(ranks) if ranks else 10_000


class HeuristicReranker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def rerank(self, *, query: str, candidates: list[RetrievedCandidate], limit: int | None = None) -> list[RetrievedCandidate]:
        query_tokens = self._tokenize(query)
        reranked: list[tuple[int, float, RetrievedCandidate]] = []
        for index, candidate in enumerate(candidates):
            score = self._score_candidate(query=query, query_tokens=query_tokens, candidate=candidate)
            reranked.append((index, score, replace(candidate, rerank_score=score)))

        reranked.sort(key=lambda item: (-item[1], item[0]))
        ordered = [candidate for _, _, candidate in reranked]
        return ordered[:limit] if limit is not None else ordered

    def _score_candidate(self, *, query: str, query_tokens: set[str], candidate: RetrievedCandidate) -> float:
        metadata = dict(candidate.document.metadata or {})
        title = str(metadata.get("title") or "")
        section_text = " ".join(self._coerce_section_path(metadata.get("section_path")))
        caption_text = str(metadata.get("caption_text") or "")
        block_type = str(metadata.get("source_block_type") or metadata.get("block_type") or "paragraph").lower()

        content_tokens = self._tokenize(candidate.document.page_content)
        lexical_overlap = len(query_tokens & content_tokens) / len(query_tokens) if query_tokens else 0.0
        title_overlap = self._overlap_ratio(query_tokens, title)
        section_overlap = self._overlap_ratio(query_tokens, section_text)
        caption_overlap = self._overlap_ratio(query_tokens, caption_text)
        exact_hit = 1.0 if self._normalized_contains(candidate.document.page_content, query) else 0.0
        prior_score = 1.0 / (candidate.best_rank() + 1.0)
        multi_query_bonus = min(0.08, max(len(candidate.query_variant_ids) - 1, 0) * 0.03)
        targeting_bias = float(metadata.get("targeting_bias") or 0.0)

        block_boost = 0.0
        if block_type == "table" and query_tokens & {"table", "row", "column", "schema", "field"}:
            block_boost = 0.06
        elif block_type == "image" and query_tokens & {"image", "figure", "fig", "diagram", "screenshot"}:
            block_boost = 0.06
        elif block_type == "formula" and query_tokens & {"formula", "equation", "math"}:
            block_boost = 0.06
        elif block_type == "caption" and query_tokens & {"caption", "figure", "table"}:
            block_boost = 0.04

        token_count = max(len(content_tokens), 1)
        length_penalty = min(0.05, max(token_count - self._settings.chunk_max_word_pieces, 0) / max(self._settings.chunk_max_word_pieces, 1) * 0.05)

        return (
            0.40 * float(candidate.fusion_score)
            + 0.22 * lexical_overlap
            + 0.10 * exact_hit
            + 0.09 * title_overlap
            + 0.08 * section_overlap
            + 0.06 * caption_overlap
            + 0.05 * prior_score
            + multi_query_bonus
            + block_boost
            + targeting_bias
            - length_penalty
        )

    @staticmethod
    def _overlap_ratio(query_tokens: set[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        text_tokens = HeuristicReranker._tokenize(text)
        if not text_tokens:
            return 0.0
        return len(query_tokens & text_tokens) / len(query_tokens)

    @staticmethod
    def _normalized_contains(text: str, query: str) -> bool:
        normalized_text = " ".join(_TOKEN_PATTERN.findall(text.lower()))
        normalized_query = " ".join(_TOKEN_PATTERN.findall(query.lower()))
        return bool(normalized_query) and normalized_query in normalized_text

    @staticmethod
    def _coerce_section_path(section_path: object) -> list[str]:
        if isinstance(section_path, list):
            return [str(item) for item in section_path if str(item).strip()]
        if isinstance(section_path, str) and section_path.strip():
            return [item.strip() for item in section_path.split("/") if item.strip()]
        return []

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token.lower() for token in _TOKEN_PATTERN.findall(text)}


class CrossEncoderReranker:
    def __init__(self, settings: Settings, fallback: HeuristicReranker) -> None:
        self._settings = settings
        self._fallback = fallback
        self._model: CrossEncoder | None = None
        self._model_load_failed = False

    def rerank(self, *, query: str, candidates: list[RetrievedCandidate], limit: int | None = None) -> list[RetrievedCandidate]:
        baseline = self._fallback.rerank(query=query, candidates=candidates, limit=None)
        if not baseline:
            return []

        model = self._load_model()
        if model is None:
            return baseline[:limit] if limit is not None else baseline

        candidate_limit = min(self._settings.rerank_cross_encoder_top_k, len(baseline))
        to_score = baseline[:candidate_limit]
        try:
            scores = model.predict([(query, candidate.document.page_content) for candidate in to_score])
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Cross-encoder rerank failed, falling back to heuristic rerank: %s", exc)
            return baseline[:limit] if limit is not None else baseline

        rescored = [replace(candidate, rerank_score=float(score)) for candidate, score in zip(to_score, scores, strict=False)]
        rescored.sort(
            key=lambda item: (
                -(item.rerank_score or 0.0),
                -(item.fusion_score or 0.0),
                item.best_rank(),
            )
        )
        ordered = rescored + baseline[candidate_limit:]
        return ordered[:limit] if limit is not None else ordered

    def _load_model(self) -> CrossEncoder | None:
        if self._model is not None:
            return self._model
        if self._model_load_failed:
            return None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._settings.rerank_cross_encoder_model)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Unable to initialize cross-encoder reranker, falling back to heuristic rerank: %s", exc)
            self._model_load_failed = True
            self._model = None
        return self._model

