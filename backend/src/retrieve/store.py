from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.src.core.config import Settings
from backend.src.core.models import (
    Citation,
    DocumentSummary,
    IngestedChunk,
    RetrievalFilter,
    SourceType,
)
from backend.src.core.text import build_snippet
from backend.src.ingest.parent_store import JsonParentStore
from backend.src.retrieve.lexical import LexicalRetrievalService
from backend.src.retrieve.query_expansion import DeterministicQueryExpander, QueryVariant
from backend.src.retrieve.query_router import QueryAnalysis, QueryRouteDecision, QueryRouter
from backend.src.retrieve.rerank import CrossEncoderReranker, HeuristicReranker, RetrievedCandidate
from backend.src.retrieve.retry import RetrievalRetryService, TargetedRetrievalPlan

logger = logging.getLogger(__name__)
_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class VectorStoreService:
    def __init__(self, settings: Settings, embeddings: object) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._index_path = settings.faiss_index_directory / "default_index"
        self._index_manager = None
        self._vector_store = self._load_or_create_store()
        self._parent_store = JsonParentStore(settings.processed_directory)
        self._lexical_retrieval: LexicalRetrievalService | None = None
        self._query_router = QueryRouter(settings)
        self._query_expander = DeterministicQueryExpander(settings)
        self._retry_service = RetrievalRetryService(settings)
        self._heuristic_reranker = HeuristicReranker(settings)
        self._cross_encoder_reranker = CrossEncoderReranker(
            settings, fallback=self._heuristic_reranker
        )

    def set_index_manager(self, index_manager: object) -> None:
        self._index_manager = index_manager
        self._lexical_retrieval = LexicalRetrievalService(
            settings=self._settings,
            index_manager=index_manager,
            metadata_matcher=self._matches_metadata_filter,
        )

    def _load_or_create_store(self) -> FAISS:
        if self._index_path.exists():
            try:
                return FAISS.load_local(
                    folder_path=str(self._index_path),
                    embeddings=self._embeddings,
                    allow_dangerous_deserialization=True,
                )
            except ModuleNotFoundError as exc:
                if "src" not in str(exc):
                    raise
                legacy_suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
                legacy_path = self._index_path.with_name(
                    f"{self._index_path.name}.legacy-{legacy_suffix}"
                )
                try:
                    self._index_path.rename(legacy_path)
                except OSError:
                    logger.warning(
                        "Failed to backup legacy FAISS index at %s; recreating in-place.",
                        self._index_path,
                    )
                else:
                    logger.warning(
                        "Detected legacy FAISS pickle module path issue (%s). Backed up old index to %s and will recreate an empty index.",
                        exc,
                        legacy_path,
                    )
                store = self._create_empty_store()
                self._save(store)
                return store
        store = self._create_empty_store()
        self._save(store)
        return store

    def _create_empty_store(self) -> FAISS:
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
        self.add_documents(documents=documents, ids=ids)

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        if not documents:
            return
        self._vector_store.add_documents(documents=documents, ids=ids)
        self._save()

    def delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        self._vector_store.delete(ids=ids)
        self._save()

    def rebuild_documents(self, documents: list[Document], ids: list[str]) -> None:
        rebuilt_store = self._create_empty_store()
        if documents:
            rebuilt_store.add_documents(documents=documents, ids=ids)
        self._vector_store = rebuilt_store
        self._save()

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[Document]:
        documents, _ = self.similarity_search_with_trace(
            query=query, k=k, metadata_filter=metadata_filter
        )
        return documents

    def similarity_search_with_trace(
        self,
        query: str,
        k: int | None = None,
        metadata_filter: RetrievalFilter | None = None,
    ) -> tuple[list[Document], dict[str, Any]]:
        top_k = k or self._settings.retriever_top_k
        analysis, decision = self._plan_query(query=query, requested_top_k=top_k)
        query_variants = self._expand_query_variants(
            query=query, analysis=analysis, decision=decision
        )
        targeted_plan = self._build_targeted_plan(decision=decision)

        initial_candidates = self._run_retrieval_round(
            query_variants=query_variants,
            decision=decision,
            metadata_filter=metadata_filter,
            targeted_plan=targeted_plan,
        )
        if not initial_candidates:
            return [], {
                "request": {
                    "query_raw": query,
                    "route_name": decision.route_name,
                    "retrieval_query": query,
                    "metadata_filter": metadata_filter.model_dump(mode="json")
                    if metadata_filter
                    else None,
                    "query_variants": [variant.variant_id for variant in query_variants],
                    "retry_triggered": False,
                    "retry_reason": "none",
                },
                "retrieval": {
                    "dense_top_hits": [],
                    "lexical_top_hits": [],
                    "fused_top_hits": [],
                    "rerank_before": [],
                    "rerank_after": [],
                    "hydration_summary": [],
                },
                "generation": {},
            }

        reranked_candidates = self._rerank_candidates(
            query=query, candidates=initial_candidates, top_k=decision.rerank_top_k
        )
        sufficiency = self._retry_service.assess(
            candidates=reranked_candidates, top_k=top_k, plan=targeted_plan
        )
        retry_decision = self._retry_service.decide_retry(
            sufficiency=sufficiency, plan=targeted_plan, attempt=0
        )
        final_candidates = reranked_candidates
        final_sufficiency = sufficiency

        if retry_decision.should_retry:
            retry_plan = self._retry_service.build_targeted_plan(
                decision=decision,
                attempt=1,
                use_hard_targeting=retry_decision.use_hard_targeting,
                dense_k=retry_decision.dense_k,
                lexical_k=retry_decision.lexical_k,
                fusion_top_k=retry_decision.fusion_top_k,
            )
            retry_round_candidates = self._run_retrieval_round(
                query_variants=query_variants,
                decision=decision,
                metadata_filter=metadata_filter,
                targeted_plan=retry_plan,
            )
            reranked_retry_candidates = self._rerank_candidates(
                query=query, candidates=retry_round_candidates, top_k=decision.rerank_top_k
            )
            retry_sufficiency = self._retry_service.assess(
                candidates=reranked_retry_candidates, top_k=top_k, plan=retry_plan
            )
            final_sufficiency = retry_sufficiency

            if not retry_sufficiency.is_sufficient and retry_decision.allow_keyword_variant:
                retry_variant = self._query_expander.build_retry_variant(
                    query=query, analysis=analysis
                )
                if retry_variant is not None and retry_variant.variant_id not in {
                    variant.variant_id for variant in query_variants
                }:
                    retry_round_candidates = self._run_retrieval_round(
                        query_variants=[*query_variants, retry_variant],
                        decision=decision,
                        metadata_filter=metadata_filter,
                        targeted_plan=retry_plan,
                    )
                    reranked_retry_candidates = self._rerank_candidates(
                        query=query, candidates=retry_round_candidates, top_k=decision.rerank_top_k
                    )
                    final_sufficiency = self._retry_service.assess(
                        candidates=reranked_retry_candidates, top_k=top_k, plan=retry_plan
                    )

            merged_candidates = self._fuse_retry_rounds_rrf(
                initial_candidates=reranked_candidates,
                retry_candidates=reranked_retry_candidates,
                top_k=decision.rerank_top_k,
            )
            final_candidates = self._rerank_candidates(
                query=query, candidates=merged_candidates, top_k=decision.rerank_top_k
            )
            final_sufficiency = self._retry_service.assess(
                candidates=final_candidates, top_k=top_k, plan=retry_plan
            )

        reranked_documents = [
            (
                self._candidate_to_document(candidate, route_name=decision.route_name),
                float(
                    candidate.rerank_score or candidate.query_fusion_score or candidate.fusion_score
                ),
            )
            for candidate in final_candidates
        ]
        hydrated_documents = self._replace_child_hits_with_parent_content(
            reranked_documents=reranked_documents, top_k=top_k
        )

        request_trace = {
            "query_raw": query,
            "route_name": decision.route_name,
            "retrieval_query": query,
            "metadata_filter": metadata_filter.model_dump(mode="json") if metadata_filter else None,
            "query_variants": [variant.variant_id for variant in query_variants],
            "retry_triggered": bool(retry_decision.should_retry),
            "retry_reason": retry_decision.reason if retry_decision.should_retry else "none",
        }
        retrieval_trace = {
            "dense_top_hits": self._summarize_candidates(
                [item for item in initial_candidates if "dense" in item.retrieval_sources],
                limit=10,
            ),
            "lexical_top_hits": self._summarize_candidates(
                [item for item in initial_candidates if "lexical" in item.retrieval_sources],
                limit=10,
            ),
            "fused_top_hits": self._summarize_candidates(initial_candidates, limit=10),
            "rerank_before": self._summarize_candidates(initial_candidates, limit=10),
            "rerank_after": self._summarize_candidates(final_candidates, limit=10),
            "hydration_summary": self._summarize_hydrated_documents(hydrated_documents, limit=10),
            "final_status": "ok" if final_sufficiency.is_sufficient else "weak_hit",
            "final_reason": final_sufficiency.reason,
        }
        trace = {
            "request": request_trace,
            "retrieval": retrieval_trace,
            "generation": {},
        }
        return hydrated_documents, trace

    def _plan_query(
        self, *, query: str, requested_top_k: int
    ) -> tuple[QueryAnalysis, QueryRouteDecision]:
        try:
            analysis = self._query_router.analyze(query)
            decision = self._query_router.route(query, requested_top_k=requested_top_k)
            return analysis, decision
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Query routing failed, falling back to general retrieval: %s", exc)
            tokens = [token.lower() for token in _TOKEN_PATTERN.findall(query)]
            analysis = QueryAnalysis(
                raw_query=query,
                normalized_query=" ".join(query.split()),
                tokens=tokens,
                token_count=len(tokens),
                has_structure_signal=False,
                has_explain_signal=False,
                has_code_signal=False,
                has_error_signal=False,
                has_broad_intent=False,
                is_short_query=False,
                is_followup_style=False,
                literal_ratio=0.0,
                has_regulation_signal=False,
            )
            decision = QueryRouteDecision(
                route_name="general",
                enable_multi_query=False,
                preserve_literals=False,
                prefer_lexical=False,
                dense_k=max(self._settings.hybrid_dense_k, requested_top_k),
                lexical_k=max(self._settings.hybrid_lexical_k, requested_top_k),
                fusion_top_k=max(self._settings.hybrid_fusion_top_k, requested_top_k),
                rerank_top_k=max(self._settings.hybrid_fusion_top_k, requested_top_k),
                max_query_variants=1,
                target_block_types=(),
                target_block_types_soft=(),
                target_section_terms=(),
                enforce_targeting=False,
                retry_policy="none",
            )
            return analysis, decision

    def _expand_query_variants(
        self,
        *,
        query: str,
        analysis: QueryAnalysis,
        decision: QueryRouteDecision,
    ) -> list[QueryVariant]:
        try:
            return self._query_expander.expand(query=query, analysis=analysis, decision=decision)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Query expansion failed, falling back to original query: %s", exc)
            normalized_query = analysis.normalized_query or query.strip()
            return [
                QueryVariant(
                    variant_id="original", text=normalized_query, kind="original", weight=1.0
                )
            ]

    def _build_targeted_plan(self, *, decision: QueryRouteDecision) -> TargetedRetrievalPlan:
        return self._retry_service.build_targeted_plan(decision=decision)

    def _run_retrieval_round(
        self,
        *,
        query_variants: list[QueryVariant],
        decision: QueryRouteDecision,
        metadata_filter: RetrievalFilter | None,
        targeted_plan: TargetedRetrievalPlan,
    ) -> list[RetrievedCandidate]:
        per_variant_candidates: list[tuple[QueryVariant, list[RetrievedCandidate]]] = []
        for variant in query_variants:
            candidates = self._retrieve_query_variant(
                variant=variant,
                decision=decision,
                metadata_filter=metadata_filter,
                targeted_plan=targeted_plan,
            )
            if candidates:
                per_variant_candidates.append((variant, candidates))

        if not per_variant_candidates:
            return []
        return self._fuse_query_variants_rrf(
            per_variant_candidates=per_variant_candidates, top_k=targeted_plan.fusion_top_k
        )

    def _retrieve_query_variant(
        self,
        *,
        variant: QueryVariant,
        decision: QueryRouteDecision,
        metadata_filter: RetrievalFilter | None,
        targeted_plan: TargetedRetrievalPlan,
    ) -> list[RetrievedCandidate]:
        dense_candidates = self._dense_recall(
            query=variant.text, k=targeted_plan.dense_k, metadata_filter=metadata_filter
        )
        dense_candidates = self._apply_targeting_to_dense_candidates(
            candidates=dense_candidates, targeted_plan=targeted_plan
        )

        lexical_candidates = self._lexical_recall(
            query=variant.text,
            k=targeted_plan.lexical_k,
            metadata_filter=metadata_filter,
            targeted_plan=targeted_plan,
        )
        lexical_candidates = self._apply_targeting_to_lexical_candidates(
            candidates=lexical_candidates, targeted_plan=targeted_plan
        )

        fused_candidates = self._fuse_candidates_rrf(
            dense_candidates=dense_candidates,
            lexical_candidates=lexical_candidates,
            top_k=targeted_plan.fusion_top_k,
        )
        biased_candidates = self._apply_soft_targeting_bias(
            candidates=fused_candidates, targeted_plan=targeted_plan
        )
        return [
            replace(
                candidate,
                query_variant_ids=tuple(
                    dict.fromkeys((*candidate.query_variant_ids, variant.variant_id))
                ),
                query_fusion_score=candidate.fusion_score,
            )
            for candidate in biased_candidates
        ]

    def _dense_recall(
        self,
        *,
        query: str,
        k: int,
        metadata_filter: RetrievalFilter | None,
    ) -> list[RetrievedCandidate]:
        scored_documents = self._vector_store.similarity_search_with_score(query=query, k=k)
        candidates: list[RetrievedCandidate] = []
        rank = 0
        for document, distance in scored_documents:
            metadata = dict(document.metadata or {})
            if metadata.get("doc_id") == "__bootstrap__":
                continue
            if not self._matches_metadata_filter(metadata, metadata_filter):
                continue
            rank += 1
            chunk_id = self._resolve_chunk_id(
                document=document, metadata=metadata, fallback_prefix="dense"
            )
            metadata.setdefault("chunk_id", chunk_id)
            normalized_document = Document(
                page_content=document.page_content, metadata=metadata, id=chunk_id
            )
            vector_score = 1.0 / (1.0 + max(float(distance), 0.0))
            candidates.append(
                RetrievedCandidate(
                    chunk_id=chunk_id,
                    document=normalized_document,
                    retrieval_sources=("dense",),
                    dense_rank=rank,
                    dense_score=vector_score,
                    fusion_score=vector_score,
                )
            )
        return candidates

    def _lexical_recall(
        self,
        *,
        query: str,
        k: int,
        metadata_filter: RetrievalFilter | None,
        targeted_plan: TargetedRetrievalPlan,
    ) -> list[RetrievedCandidate]:
        if not self._settings.hybrid_lexical_enabled or self._lexical_retrieval is None:
            return []

        record_matcher = None
        if targeted_plan.use_hard_targeting and targeted_plan.target_block_types:
            target_block_types = set(targeted_plan.target_block_types)

            def record_matcher(record: Any) -> bool:
                block_type = str(
                    record.metadata.get("source_block_type")
                    or record.metadata.get("block_type")
                    or "paragraph"
                ).lower()
                return block_type in target_block_types

        return self._lexical_retrieval.search(
            query=query,
            k=k,
            metadata_filter=metadata_filter,
            record_matcher=record_matcher,
        )

    def _apply_targeting_to_dense_candidates(
        self,
        *,
        candidates: list[RetrievedCandidate],
        targeted_plan: TargetedRetrievalPlan,
    ) -> list[RetrievedCandidate]:
        if not candidates:
            return []
        if not targeted_plan.use_hard_targeting or not targeted_plan.target_block_types:
            return self._annotate_candidates(
                candidates=candidates, targeted_plan=targeted_plan, targeting_mode="soft"
            )

        targeted_candidates = [
            candidate
            for candidate in candidates
            if self._candidate_matches_target(candidate, targeted_plan)
        ]
        if len(targeted_candidates) < targeted_plan.hard_min_candidates:
            return self._annotate_candidates(
                candidates=candidates, targeted_plan=targeted_plan, targeting_mode="soft_fallback"
            )
        return self._annotate_candidates(
            candidates=targeted_candidates, targeted_plan=targeted_plan, targeting_mode="hard"
        )

    def _apply_targeting_to_lexical_candidates(
        self,
        *,
        candidates: list[RetrievedCandidate],
        targeted_plan: TargetedRetrievalPlan,
    ) -> list[RetrievedCandidate]:
        if not candidates:
            return []
        if (
            targeted_plan.use_hard_targeting
            and targeted_plan.target_block_types
            and len(candidates) < targeted_plan.hard_min_candidates
        ):
            return self._annotate_candidates(
                candidates=candidates, targeted_plan=targeted_plan, targeting_mode="soft_fallback"
            )
        if targeted_plan.use_hard_targeting and targeted_plan.target_block_types:
            return self._annotate_candidates(
                candidates=candidates, targeted_plan=targeted_plan, targeting_mode="hard"
            )
        return self._annotate_candidates(
            candidates=candidates, targeted_plan=targeted_plan, targeting_mode="soft"
        )

    def _apply_soft_targeting_bias(
        self,
        *,
        candidates: list[RetrievedCandidate],
        targeted_plan: TargetedRetrievalPlan,
    ) -> list[RetrievedCandidate]:
        if not candidates:
            return []
        return sorted(
            candidates,
            key=lambda candidate: (
                self._candidate_target_priority(candidate, targeted_plan),
                -float(
                    candidate.query_fusion_score
                    if candidate.query_fusion_score is not None
                    else candidate.fusion_score
                ),
                candidate.best_rank(),
                candidate.chunk_id,
            ),
        )

    def _annotate_candidates(
        self,
        *,
        candidates: list[RetrievedCandidate],
        targeted_plan: TargetedRetrievalPlan,
        targeting_mode: str,
    ) -> list[RetrievedCandidate]:
        annotated: list[RetrievedCandidate] = []
        for candidate in candidates:
            block_type = self._candidate_block_type(candidate)
            target_hit = self._candidate_matches_target(candidate, targeted_plan)
            soft_hit = self._candidate_matches_soft_target(candidate, targeted_plan)
            section_hit = self._candidate_matches_section_terms(candidate, targeted_plan)
            bias = 0.0
            if target_hit:
                bias += targeted_plan.soft_bias_weight
            elif soft_hit:
                bias += targeted_plan.soft_bias_weight * 0.5
            if section_hit:
                bias += min(0.03, targeted_plan.soft_bias_weight * 0.5)
            annotated.append(
                self._replace_candidate_metadata(
                    candidate=candidate,
                    metadata_updates={
                        "source_block_type": block_type,
                        "targeting_mode": targeting_mode,
                        "target_block_hit": target_hit,
                        "soft_target_hit": soft_hit,
                        "target_section_hit": section_hit,
                        "targeting_bias": bias,
                    },
                )
            )
        return annotated

    def _fuse_candidates_rrf(
        self,
        *,
        dense_candidates: list[RetrievedCandidate],
        lexical_candidates: list[RetrievedCandidate],
        top_k: int,
    ) -> list[RetrievedCandidate]:
        if not dense_candidates and not lexical_candidates:
            return []

        fused: dict[str, RetrievedCandidate] = {}
        rrf_k = self._settings.hybrid_rrf_k
        for candidates, source_name in (
            (dense_candidates, "dense"),
            (lexical_candidates, "lexical"),
        ):
            for rank, candidate in enumerate(candidates, start=1):
                contribution = 1.0 / (rrf_k + rank)
                existing = fused.get(candidate.chunk_id)
                if existing is None:
                    fused[candidate.chunk_id] = RetrievedCandidate(
                        chunk_id=candidate.chunk_id,
                        document=candidate.document,
                        retrieval_sources=tuple(
                            dict.fromkeys((*candidate.retrieval_sources, source_name))
                        ),
                        query_variant_ids=candidate.query_variant_ids,
                        dense_rank=candidate.dense_rank,
                        dense_score=candidate.dense_score,
                        lexical_rank=candidate.lexical_rank,
                        lexical_score=candidate.lexical_score,
                        fusion_score=contribution,
                        query_fusion_score=candidate.query_fusion_score,
                    )
                    continue
                fused[candidate.chunk_id] = RetrievedCandidate(
                    chunk_id=candidate.chunk_id,
                    document=self._merge_candidate_documents(existing.document, candidate.document),
                    retrieval_sources=tuple(
                        dict.fromkeys(
                            (*existing.retrieval_sources, *candidate.retrieval_sources, source_name)
                        )
                    ),
                    query_variant_ids=tuple(
                        dict.fromkeys((*existing.query_variant_ids, *candidate.query_variant_ids))
                    ),
                    dense_rank=existing.dense_rank
                    if existing.dense_rank is not None
                    else candidate.dense_rank,
                    dense_score=max(existing.dense_score or 0.0, candidate.dense_score or 0.0)
                    or None,
                    lexical_rank=existing.lexical_rank
                    if existing.lexical_rank is not None
                    else candidate.lexical_rank,
                    lexical_score=max(existing.lexical_score or 0.0, candidate.lexical_score or 0.0)
                    or None,
                    fusion_score=existing.fusion_score + contribution,
                    query_fusion_score=existing.query_fusion_score
                    if existing.query_fusion_score is not None
                    else candidate.query_fusion_score,
                )

        ordered = sorted(
            fused.values(), key=lambda item: (-item.fusion_score, item.best_rank(), item.chunk_id)
        )
        return ordered[:top_k]

    def _fuse_query_variants_rrf(
        self,
        *,
        per_variant_candidates: list[tuple[QueryVariant, list[RetrievedCandidate]]],
        top_k: int,
    ) -> list[RetrievedCandidate]:
        if not per_variant_candidates:
            return []
        if len(per_variant_candidates) == 1:
            return per_variant_candidates[0][1][:top_k]

        fused: dict[str, RetrievedCandidate] = {}
        rrf_k = self._settings.hybrid_rrf_k
        for variant, candidates in per_variant_candidates:
            for rank, candidate in enumerate(candidates, start=1):
                contribution = float(variant.weight) / (rrf_k + rank)
                existing = fused.get(candidate.chunk_id)
                if existing is None:
                    fused[candidate.chunk_id] = replace(
                        candidate,
                        query_variant_ids=tuple(
                            dict.fromkeys((*candidate.query_variant_ids, variant.variant_id))
                        ),
                        fusion_score=contribution,
                        query_fusion_score=contribution,
                    )
                    continue
                fused[candidate.chunk_id] = RetrievedCandidate(
                    chunk_id=candidate.chunk_id,
                    document=self._merge_candidate_documents(existing.document, candidate.document),
                    retrieval_sources=tuple(
                        dict.fromkeys((*existing.retrieval_sources, *candidate.retrieval_sources))
                    ),
                    query_variant_ids=tuple(
                        dict.fromkeys(
                            (
                                *existing.query_variant_ids,
                                variant.variant_id,
                                *candidate.query_variant_ids,
                            )
                        )
                    ),
                    dense_rank=existing.dense_rank
                    if existing.dense_rank is not None
                    else candidate.dense_rank,
                    dense_score=max(existing.dense_score or 0.0, candidate.dense_score or 0.0)
                    or None,
                    lexical_rank=existing.lexical_rank
                    if existing.lexical_rank is not None
                    else candidate.lexical_rank,
                    lexical_score=max(existing.lexical_score or 0.0, candidate.lexical_score or 0.0)
                    or None,
                    fusion_score=existing.fusion_score + contribution,
                    query_fusion_score=(existing.query_fusion_score or 0.0) + contribution,
                    rerank_score=existing.rerank_score,
                )

        ordered = sorted(
            fused.values(),
            key=lambda item: (
                -item.fusion_score,
                0 if "original" in item.query_variant_ids else 1,
                item.best_rank(),
                item.chunk_id,
            ),
        )
        return ordered[:top_k]

    def _fuse_retry_rounds_rrf(
        self,
        *,
        initial_candidates: list[RetrievedCandidate],
        retry_candidates: list[RetrievedCandidate],
        top_k: int,
    ) -> list[RetrievedCandidate]:
        if not retry_candidates:
            return initial_candidates[:top_k]

        fused: dict[str, RetrievedCandidate] = {}
        rrf_k = self._settings.hybrid_rrf_k
        for round_name, candidates, weight in (
            ("initial", initial_candidates, 1.0),
            ("retry", retry_candidates, 0.95),
        ):
            for rank, candidate in enumerate(candidates, start=1):
                contribution = weight / (rrf_k + rank)
                existing = fused.get(candidate.chunk_id)
                if existing is None:
                    fused[candidate.chunk_id] = RetrievedCandidate(
                        chunk_id=candidate.chunk_id,
                        document=candidate.document,
                        retrieval_sources=tuple(
                            dict.fromkeys((*candidate.retrieval_sources, round_name))
                        ),
                        query_variant_ids=candidate.query_variant_ids,
                        dense_rank=candidate.dense_rank,
                        dense_score=candidate.dense_score,
                        lexical_rank=candidate.lexical_rank,
                        lexical_score=candidate.lexical_score,
                        fusion_score=contribution,
                        query_fusion_score=candidate.query_fusion_score
                        if candidate.query_fusion_score is not None
                        else contribution,
                    )
                    continue
                fused[candidate.chunk_id] = RetrievedCandidate(
                    chunk_id=candidate.chunk_id,
                    document=self._merge_candidate_documents(existing.document, candidate.document),
                    retrieval_sources=tuple(
                        dict.fromkeys(
                            (*existing.retrieval_sources, *candidate.retrieval_sources, round_name)
                        )
                    ),
                    query_variant_ids=tuple(
                        dict.fromkeys((*existing.query_variant_ids, *candidate.query_variant_ids))
                    ),
                    dense_rank=existing.dense_rank
                    if existing.dense_rank is not None
                    else candidate.dense_rank,
                    dense_score=max(existing.dense_score or 0.0, candidate.dense_score or 0.0)
                    or None,
                    lexical_rank=existing.lexical_rank
                    if existing.lexical_rank is not None
                    else candidate.lexical_rank,
                    lexical_score=max(existing.lexical_score or 0.0, candidate.lexical_score or 0.0)
                    or None,
                    fusion_score=existing.fusion_score + contribution,
                    query_fusion_score=max(
                        existing.query_fusion_score or 0.0, candidate.query_fusion_score or 0.0
                    )
                    or None,
                )

        ordered = sorted(
            fused.values(),
            key=lambda item: (
                -item.fusion_score,
                0 if "original" in item.query_variant_ids else 1,
                item.best_rank(),
                item.chunk_id,
            ),
        )
        return ordered[:top_k]

    def _rerank_candidates(
        self,
        *,
        query: str,
        candidates: list[RetrievedCandidate],
        top_k: int,
    ) -> list[RetrievedCandidate]:
        if not candidates:
            return []

        rerank_mode = str(self._settings.rerank_mode or "heuristic").lower()
        if rerank_mode == "off":
            return candidates[:top_k]
        if rerank_mode == "cross_encoder":
            return self._cross_encoder_reranker.rerank(
                query=query, candidates=candidates, limit=top_k
            )
        return self._heuristic_reranker.rerank(query=query, candidates=candidates, limit=top_k)

    def _candidate_to_document(self, candidate: RetrievedCandidate, *, route_name: str) -> Document:
        metadata = dict(candidate.document.metadata or {})
        metadata["chunk_id"] = candidate.chunk_id
        metadata["dense_rank"] = candidate.dense_rank
        metadata["dense_score"] = candidate.dense_score
        metadata["lexical_rank"] = candidate.lexical_rank
        metadata["lexical_score"] = candidate.lexical_score
        metadata["fusion_score"] = candidate.fusion_score
        metadata["query_fusion_score"] = (
            candidate.query_fusion_score
            if candidate.query_fusion_score is not None
            else candidate.fusion_score
        )
        metadata["rerank_score"] = (
            candidate.rerank_score if candidate.rerank_score is not None else candidate.fusion_score
        )
        metadata["retrieval_sources"] = list(candidate.retrieval_sources)
        metadata["query_variant_ids"] = list(candidate.query_variant_ids)
        metadata["matched_query_count"] = len(candidate.query_variant_ids)
        metadata["route_name"] = route_name
        return Document(
            page_content=candidate.document.page_content, metadata=metadata, id=candidate.chunk_id
        )

    def _summarize_candidates(
        self, candidates: list[RetrievedCandidate], *, limit: int
    ) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for candidate in candidates[:limit]:
            metadata = candidate.document.metadata or {}
            summary.append(
                {
                    "chunk_id": candidate.chunk_id,
                    "doc_id": str(metadata.get("doc_id") or ""),
                    "block_type": str(
                        metadata.get("source_block_type")
                        or metadata.get("block_type")
                        or "paragraph"
                    ),
                    "score": float(
                        candidate.rerank_score
                        or candidate.query_fusion_score
                        or candidate.fusion_score
                    ),
                    "retrieval_sources": list(candidate.retrieval_sources),
                    "query_variant_ids": list(candidate.query_variant_ids),
                }
            )
        return summary

    def _summarize_hydrated_documents(
        self, documents: list[Document], *, limit: int
    ) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for document in documents[:limit]:
            metadata = document.metadata or {}
            summary.append(
                {
                    "doc_id": str(metadata.get("doc_id") or ""),
                    "chunk_id": str(metadata.get("chunk_id") or getattr(document, "id", "") or ""),
                    "parent_chunk_id": str(metadata.get("parent_chunk_id") or ""),
                    "hydration_source": str(
                        metadata.get("parent_content_source") or "child_fallback"
                    ),
                    "matched_child_count": int(metadata.get("matched_child_count") or 0),
                }
            )
        return summary

    def _replace_child_hits_with_parent_content(
        self,
        *,
        reranked_documents: list[tuple[Document, float]],
        top_k: int,
    ) -> list[Document]:
        parent_buckets: dict[str, dict[str, Any]] = {}
        passthrough_buckets: dict[
            tuple[str, str | None, tuple[str, ...]], tuple[float, Document]
        ] = {}

        for document, score in reranked_documents:
            metadata = dict(document.metadata or {})
            parent_chunk_id = metadata.get("parent_chunk_id")
            if not parent_chunk_id:
                dedupe_key = self._build_source_dedupe_key(metadata)
                existing = passthrough_buckets.get(dedupe_key)
                if existing is None or score > existing[0]:
                    passthrough_buckets[dedupe_key] = (score, document)
                continue

            parent_key = str(parent_chunk_id)
            chunk_id = self._resolve_chunk_id(
                document=document, metadata=metadata, fallback_prefix="child"
            )
            bucket = parent_buckets.get(parent_key)
            if bucket is None:
                parent_buckets[parent_key] = {
                    "score": score,
                    "document": document,
                    "child_ids": [chunk_id],
                }
                continue
            bucket["child_ids"].append(chunk_id)
            if score > float(bucket["score"]):
                bucket["score"] = score
                bucket["document"] = document

        hydrated_results: list[tuple[float, Document]] = []
        for parent_key, bucket in parent_buckets.items():
            source_document = bucket["document"]
            metadata = dict(source_document.metadata or {})
            matched_child_ids = sorted({str(item) for item in bucket["child_ids"] if str(item)})
            metadata["matched_child_ids"] = matched_child_ids
            metadata["matched_child_count"] = len(matched_child_ids)

            page_content, content_source, hydrated = self._hydrate_parent_content(
                metadata=metadata,
                source_document=source_document,
                parent_chunk_id=parent_key,
            )
            metadata["parent_content_source"] = content_source
            metadata["parent_hydrated"] = hydrated
            hydrated_results.append(
                (
                    float(bucket["score"]),
                    Document(
                        page_content=page_content,
                        metadata=metadata,
                        id=getattr(source_document, "id", None),
                    ),
                )
            )

        hydrated_results.extend(passthrough_buckets.values())
        hydrated_results.sort(key=lambda item: -item[0])
        return [document for _, document in hydrated_results[:top_k]]

    def _hydrate_parent_content(
        self,
        *,
        metadata: dict[str, Any],
        source_document: Document,
        parent_chunk_id: str,
    ) -> tuple[str, str, bool]:
        parent_content = metadata.get("parent_content")
        if isinstance(parent_content, str) and parent_content.strip():
            return parent_content, "metadata", True

        parent_store_ref = str(metadata.get("parent_store_ref") or "").strip()
        parent_record = (
            self._parent_store.load_by_ref(parent_store_ref) if parent_store_ref else None
        )
        if parent_record is None and parent_chunk_id:
            parent_record = self._parent_store.load(parent_chunk_id)
        if parent_record is not None and parent_record.parent_content.strip():
            return parent_record.parent_content, "store", True

        child_content = metadata.get("child_content") or source_document.page_content
        return str(child_content), "child_fallback", False

    def list_documents(self) -> list[DocumentSummary]:
        if self._index_manager is not None:
            return self._index_manager.list_documents()

        metadatas = list(self._vector_store.docstore._dict.values())
        buckets: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "source_name": "",
                "source_type": "",
                "source_uri_or_path": "",
                "count": 0,
                "sections": set(),
            }
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
                    page_or_section=str(metadata.get("page_or_section"))
                    if metadata.get("page_or_section")
                    else None,
                    page=str(page) if page else None,
                    title=str(metadata.get("title")) if metadata.get("title") else None,
                    section_path=section_path,
                    doc_type=SourceType(doc_type) if doc_type else None,
                    updated_at=VectorStoreService._parse_datetime(metadata.get("updated_at")),
                    snippet=build_snippet(
                        str(metadata.get("child_content") or document.page_content)
                    ),
                )
            )
        return citations

    @staticmethod
    def _matches_metadata_filter(
        metadata: dict[str, Any], metadata_filter: RetrievalFilter | None
    ) -> bool:
        doc_id = str(metadata.get("doc_id", ""))
        scope = str(metadata.get("scope") or "").lower()
        if scope == "session":
            allowed_doc_ids = (
                {str(item) for item in metadata_filter.doc_ids}
                if metadata_filter and metadata_filter.doc_ids
                else set()
            )
            if doc_id not in allowed_doc_ids:
                return False

        if metadata_filter is None:
            return True
        if metadata_filter.doc_ids and doc_id not in {str(item) for item in metadata_filter.doc_ids}:
            return False
        if metadata_filter.source_names:
            source_name = str(metadata.get("source_name", ""))
            source_uri_or_path = str(metadata.get("source_uri_or_path", ""))
            if (
                source_name not in metadata_filter.source_names
                and source_uri_or_path not in metadata_filter.source_names
            ):
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
        if metadata_filter.updated_after and (
            updated_at is None or updated_at < metadata_filter.updated_after
        ):
            return False
        if metadata_filter.updated_before and (
            updated_at is None or updated_at > metadata_filter.updated_before
        ):
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
    def _resolve_chunk_id(
        *, document: Document, metadata: dict[str, Any], fallback_prefix: str
    ) -> str:
        chunk_id = metadata.get("chunk_id") or getattr(document, "id", None)
        if isinstance(chunk_id, str) and chunk_id.strip():
            return chunk_id
        doc_id = str(metadata.get("doc_id") or fallback_prefix)
        page = str(metadata.get("page") or metadata.get("page_or_section") or "na")
        section = (
            "-".join(VectorStoreService._coerce_section_path(metadata.get("section_path")))
            or "section"
        )
        return f"{fallback_prefix}-{doc_id}-{page}-{section}"

    @staticmethod
    def _build_source_dedupe_key(
        metadata: dict[str, Any],
    ) -> tuple[str, str | None, tuple[str, ...]]:
        source_uri_or_path = str(
            metadata.get("source_uri_or_path")
            or metadata.get("source_name")
            or metadata.get("doc_id")
            or ""
        )
        page = metadata.get("page") or metadata.get("page_or_section")
        section_path = tuple(VectorStoreService._coerce_section_path(metadata.get("section_path")))
        return (source_uri_or_path, str(page) if page else None, section_path)

    @staticmethod
    def _candidate_block_type(candidate: RetrievedCandidate) -> str:
        metadata = candidate.document.metadata or {}
        return str(
            metadata.get("source_block_type") or metadata.get("block_type") or "paragraph"
        ).lower()

    def _candidate_matches_target(
        self, candidate: RetrievedCandidate, targeted_plan: TargetedRetrievalPlan
    ) -> bool:
        return self._candidate_block_type(candidate) in set(targeted_plan.target_block_types)

    def _candidate_matches_soft_target(
        self, candidate: RetrievedCandidate, targeted_plan: TargetedRetrievalPlan
    ) -> bool:
        return self._candidate_block_type(candidate) in set(targeted_plan.target_block_types_soft)

    def _candidate_matches_section_terms(
        self, candidate: RetrievedCandidate, targeted_plan: TargetedRetrievalPlan
    ) -> bool:
        if not targeted_plan.target_section_terms:
            return False
        metadata = candidate.document.metadata or {}
        fields = [
            str(metadata.get("title") or ""),
            str(metadata.get("caption_text") or ""),
            " ".join(self._coerce_section_path(metadata.get("section_path"))),
        ]
        haystack = " ".join(fields).lower()
        return any(term.lower() in haystack for term in targeted_plan.target_section_terms)

    def _candidate_target_priority(
        self, candidate: RetrievedCandidate, targeted_plan: TargetedRetrievalPlan
    ) -> int:
        block_type = self._candidate_block_type(candidate)
        if self._candidate_matches_target(candidate, targeted_plan):
            if targeted_plan.route_name == "explained_structure" and block_type == "paragraph":
                return 1
            return 0
        if self._candidate_matches_soft_target(candidate, targeted_plan):
            return 1
        if self._candidate_matches_section_terms(candidate, targeted_plan):
            return 2
        return 3

    @staticmethod
    def _replace_candidate_metadata(
        candidate: RetrievedCandidate, metadata_updates: dict[str, Any]
    ) -> RetrievedCandidate:
        metadata = dict(candidate.document.metadata or {})
        metadata.update(metadata_updates)
        document = Document(
            page_content=candidate.document.page_content,
            metadata=metadata,
            id=getattr(candidate.document, "id", None),
        )
        return replace(candidate, document=document)

    @staticmethod
    def _merge_candidate_documents(primary: Document, secondary: Document) -> Document:
        metadata = dict(primary.metadata or {})
        for key, value in (secondary.metadata or {}).items():
            if key not in metadata or metadata[key] in (
                None,
                "",
                [],
                (),
            ):  # prefer richer existing metadata
                metadata[key] = value
        return Document(
            page_content=primary.page_content, metadata=metadata, id=getattr(primary, "id", None)
        )
