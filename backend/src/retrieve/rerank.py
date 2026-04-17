from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from backend.src.core.config import Settings
from backend.src.core.text import normalize_text, tokenize_search_text

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


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

    def rerank(
        self, *, query: str, candidates: list[RetrievedCandidate], limit: int | None = None
    ) -> list[RetrievedCandidate]:
        query_tokens = self._tokenize(query)
        reranked: list[tuple[int, float, RetrievedCandidate]] = []
        for index, candidate in enumerate(candidates):
            score = self._score_candidate(
                query=query, query_tokens=query_tokens, candidate=candidate
            )
            reranked.append((index, score, replace(candidate, rerank_score=score)))

        reranked.sort(key=lambda item: (-item[1], item[0]))
        ordered = [candidate for _, _, candidate in reranked]
        return ordered[:limit] if limit is not None else ordered

    def _score_candidate(
        self, *, query: str, query_tokens: set[str], candidate: RetrievedCandidate
    ) -> float:
        metadata = dict(candidate.document.metadata or {})
        title = str(metadata.get("title") or "")
        section_text = " ".join(self._coerce_section_path(metadata.get("section_path")))
        caption_text = str(metadata.get("caption_text") or "")
        block_type = str(
            metadata.get("source_block_type") or metadata.get("block_type") or "paragraph"
        ).lower()
        clause_id = str(metadata.get("clause_id") or "")
        table_id = str(metadata.get("table_id") or "")
        english_alias = str(metadata.get("english_alias") or "")
        pseudo_table_row_key = str(metadata.get("pseudo_table_row_key") or "")
        numeric_anchor_terms = [
            str(item) for item in metadata.get("numeric_anchor_terms", []) if str(item).strip()
        ]
        normalized_query = normalize_text(query)
        normalized_content = normalize_text(candidate.document.page_content)

        content_tokens = self._tokenize(candidate.document.page_content)
        lexical_overlap = (
            len(query_tokens & content_tokens) / len(query_tokens) if query_tokens else 0.0
        )
        title_overlap = self._overlap_ratio(query_tokens, title)
        section_overlap = self._overlap_ratio(query_tokens, section_text)
        caption_overlap = self._overlap_ratio(query_tokens, caption_text)
        exact_hit = (
            1.0 if self._normalized_contains(candidate.document.page_content, query) else 0.0
        )
        prior_score = 1.0 / (candidate.best_rank() + 1.0)
        multi_query_bonus = min(0.08, max(len(candidate.query_variant_ids) - 1, 0) * 0.03)
        targeting_bias = float(metadata.get("targeting_bias") or 0.0)
        clause_hit = 1.0 if clause_id and clause_id in normalized_query else 0.0
        table_hit = 1.0 if table_id and (table_id in normalized_query or f"表{table_id}" in normalized_query) else 0.0
        normative_hit = 1.0 if metadata.get("is_normative_clause") and re.search(
            r"目的|适用|规定|要求|不应|应满足|用于|检测", normalized_query
        ) else 0.0
        english_hit = 1.0 if english_alias and re.search(r"英文|english", normalized_query, re.IGNORECASE) else 0.0
        numeric_hit = 1.0 if numeric_anchor_terms and re.search(
            r"%|mpa|多少|几|天|要求|阈值|强度|数值", normalized_query, re.IGNORECASE
        ) else 0.0
        pseudo_row_hit = 1.0 if pseudo_table_row_key and pseudo_table_row_key in normalized_query else 0.0
        regulation_anchor_bonus = 1.0 if metadata.get("is_regulation_anchor") else 0.0
        purpose_hit = 1.0 if re.search(r"目的|宗旨", normalized_query) and "制定本规范" in normalized_content else 0.0
        applicability_hit = 1.0 if "适用" in normalized_query and "适用于" in normalized_content else 0.0
        usage_hit = 1.0 if "主要用于" in normalized_query and re.search(
            r"主要用于|检测[^。；]*判定", normalized_content
        ) else 0.0
        capacity_hit = 1.0 if re.search(r"承载力指标|确定什么", normalized_query) and "极限承载力" in normalized_content else 0.0

        block_boost = 0.0
        if block_type == "table" and query_tokens & {"table", "row", "column", "schema", "field"}:
            block_boost = 0.06
        elif block_type == "image" and query_tokens & {
            "image",
            "figure",
            "fig",
            "diagram",
            "screenshot",
        }:
            block_boost = 0.06
        elif block_type == "formula" and query_tokens & {"formula", "equation", "math"}:
            block_boost = 0.06
        elif block_type == "caption" and query_tokens & {"caption", "figure", "table"}:
            block_boost = 0.04

        token_count = max(len(content_tokens), 1)
        length_penalty = min(
            0.05,
            max(token_count - self._settings.chunk_max_word_pieces, 0)
            / max(self._settings.chunk_max_word_pieces, 1)
            * 0.05,
        )

        return (
            0.30 * float(candidate.fusion_score)
            + 0.20 * lexical_overlap
            + 0.10 * exact_hit
            + 0.09 * title_overlap
            + 0.08 * section_overlap
            + 0.06 * caption_overlap
            + 0.05 * prior_score
            + 0.06 * clause_hit
            + 0.06 * table_hit
            + 0.05 * normative_hit
            + 0.05 * english_hit
            + 0.05 * numeric_hit
            + 0.06 * pseudo_row_hit
            + 0.04 * regulation_anchor_bonus
            + 0.08 * purpose_hit
            + 0.08 * applicability_hit
            + 0.07 * usage_hit
            + 0.07 * capacity_hit
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
        normalized_text = " ".join(tokenize_search_text(text))
        normalized_query = " ".join(tokenize_search_text(query))
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
        return set(tokenize_search_text(text))


class CrossEncoderReranker:
    def __init__(self, settings: Settings, fallback: HeuristicReranker) -> None:
        self._settings = settings
        self._fallback = fallback
        self._model: CrossEncoder | None = None
        self._model_load_failed = False

    def rerank(
        self, *, query: str, candidates: list[RetrievedCandidate], limit: int | None = None
    ) -> list[RetrievedCandidate]:
        baseline = self._fallback.rerank(query=query, candidates=candidates, limit=None)
        if not baseline:
            return []

        model = self._load_model()
        if model is None:
            return baseline[:limit] if limit is not None else baseline

        candidate_limit = min(self._settings.rerank_cross_encoder_top_k, len(baseline))
        to_score = baseline[:candidate_limit]
        try:
            scores = model.predict(
                [(query, candidate.document.page_content) for candidate in to_score]
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Cross-encoder rerank failed, falling back to heuristic rerank: %s", exc)
            return baseline[:limit] if limit is not None else baseline

        rescored = [
            replace(candidate, rerank_score=float(score))
            for candidate, score in zip(to_score, scores, strict=False)
        ]
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
            logger.warning(
                "Unable to initialize cross-encoder reranker, falling back to heuristic rerank: %s",
                exc,
            )
            self._model_load_failed = True
            self._model = None
        return self._model
