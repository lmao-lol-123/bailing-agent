from __future__ import annotations

import math
from dataclasses import dataclass

from backend.src.core.config import Settings
from backend.src.retrieve.query_router import QueryRouteDecision
from backend.src.retrieve.rerank import RetrievedCandidate


@dataclass(frozen=True)
class TargetedRetrievalPlan:
    route_name: str
    target_block_types: tuple[str, ...]
    target_block_types_soft: tuple[str, ...]
    target_section_terms: tuple[str, ...]
    enforce_targeting: bool
    use_hard_targeting: bool
    retry_policy: str
    dense_k: int
    lexical_k: int
    fusion_top_k: int
    soft_bias_weight: float
    hard_min_candidates: int
    attempt: int = 0


@dataclass(frozen=True)
class RetrievalSufficiency:
    is_sufficient: bool
    result_count: int
    min_required_results: int
    top_score: float
    second_score: float
    target_hit_count: int
    context_hit_count: int
    reason: str


@dataclass(frozen=True)
class RetryDecision:
    should_retry: bool
    policy: str
    reason: str
    dense_k: int
    lexical_k: int
    fusion_top_k: int
    allow_keyword_variant: bool
    use_hard_targeting: bool


class RetrievalRetryService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build_targeted_plan(
        self,
        *,
        decision: QueryRouteDecision,
        attempt: int = 0,
        use_hard_targeting: bool | None = None,
        dense_k: int | None = None,
        lexical_k: int | None = None,
        fusion_top_k: int | None = None,
    ) -> TargetedRetrievalPlan:
        hard_targeting = decision.enforce_targeting and decision.route_name in {"structure", "explained_structure"}
        if use_hard_targeting is not None:
            hard_targeting = use_hard_targeting and hard_targeting
        return TargetedRetrievalPlan(
            route_name=decision.route_name,
            target_block_types=tuple(decision.target_block_types),
            target_block_types_soft=tuple(decision.target_block_types_soft),
            target_section_terms=tuple(decision.target_section_terms),
            enforce_targeting=decision.enforce_targeting,
            use_hard_targeting=hard_targeting,
            retry_policy=decision.retry_policy,
            dense_k=dense_k if dense_k is not None else decision.dense_k,
            lexical_k=lexical_k if lexical_k is not None else decision.lexical_k,
            fusion_top_k=fusion_top_k if fusion_top_k is not None else decision.fusion_top_k,
            soft_bias_weight=self._settings.targeted_soft_bias_weight,
            hard_min_candidates=self._settings.targeted_hard_min_candidates,
            attempt=attempt,
        )

    def assess(
        self,
        *,
        candidates: list[RetrievedCandidate],
        top_k: int,
        plan: TargetedRetrievalPlan,
    ) -> RetrievalSufficiency:
        min_required_results = max(1, math.ceil(top_k * self._settings.targeted_min_results_ratio))
        result_count = len(candidates)
        top_score = self._candidate_score(candidates[0]) if candidates else 0.0
        second_score = self._candidate_score(candidates[1]) if len(candidates) > 1 else 0.0
        target_hit_count = sum(1 for candidate in candidates if self._is_primary_target(candidate, plan))
        context_hit_count = sum(1 for candidate in candidates if self._is_context_target(candidate, plan))

        if result_count < min_required_results:
            return RetrievalSufficiency(False, result_count, min_required_results, top_score, second_score, target_hit_count, context_hit_count, "insufficient_count")
        if plan.route_name in {"structure", "explained_structure"} and plan.target_block_types and target_hit_count == 0:
            return RetrievalSufficiency(False, result_count, min_required_results, top_score, second_score, target_hit_count, context_hit_count, "no_target_block_hit")
        if plan.route_name == "explained_structure" and context_hit_count == 0:
            return RetrievalSufficiency(False, result_count, min_required_results, top_score, second_score, target_hit_count, context_hit_count, "missing_explanatory_context")
        if result_count and top_score < 0.05 and second_score < 0.04:
            return RetrievalSufficiency(False, result_count, min_required_results, top_score, second_score, target_hit_count, context_hit_count, "low_rerank_confidence")
        return RetrievalSufficiency(True, result_count, min_required_results, top_score, second_score, target_hit_count, context_hit_count, "sufficient")

    def decide_retry(
        self,
        *,
        sufficiency: RetrievalSufficiency,
        plan: TargetedRetrievalPlan,
        attempt: int,
    ) -> RetryDecision:
        if sufficiency.is_sufficient:
            return RetryDecision(False, plan.retry_policy, sufficiency.reason, plan.dense_k, plan.lexical_k, plan.fusion_top_k, False, plan.use_hard_targeting)
        if not self._settings.targeted_retry_enabled or attempt >= self._settings.targeted_retry_max_attempts:
            return RetryDecision(False, plan.retry_policy, sufficiency.reason, plan.dense_k, plan.lexical_k, plan.fusion_top_k, False, plan.use_hard_targeting)
        if plan.retry_policy == "none":
            return RetryDecision(False, plan.retry_policy, sufficiency.reason, plan.dense_k, plan.lexical_k, plan.fusion_top_k, False, plan.use_hard_targeting)

        dense_k = max(plan.dense_k + 1, int(math.ceil(plan.dense_k * self._settings.targeted_retry_dense_multiplier)))
        lexical_k = max(plan.lexical_k + 1, int(math.ceil(plan.lexical_k * self._settings.targeted_retry_lexical_multiplier)))
        fusion_top_k = max(plan.fusion_top_k, max(dense_k, lexical_k))
        allow_keyword_variant = plan.retry_policy == "widen_then_keyword"
        return RetryDecision(
            should_retry=True,
            policy=plan.retry_policy,
            reason=sufficiency.reason,
            dense_k=dense_k,
            lexical_k=lexical_k,
            fusion_top_k=fusion_top_k,
            allow_keyword_variant=allow_keyword_variant,
            use_hard_targeting=False,
        )

    @staticmethod
    def _candidate_score(candidate: RetrievedCandidate) -> float:
        if candidate.rerank_score is not None:
            return float(candidate.rerank_score)
        return float(candidate.query_fusion_score if candidate.query_fusion_score is not None else candidate.fusion_score)

    @staticmethod
    def _candidate_block_type(candidate: RetrievedCandidate) -> str:
        metadata = candidate.document.metadata or {}
        return str(metadata.get("source_block_type") or metadata.get("block_type") or "paragraph").lower()

    def _is_primary_target(self, candidate: RetrievedCandidate, plan: TargetedRetrievalPlan) -> bool:
        return self._candidate_block_type(candidate) in set(plan.target_block_types)

    def _is_context_target(self, candidate: RetrievedCandidate, plan: TargetedRetrievalPlan) -> bool:
        block_type = self._candidate_block_type(candidate)
        if block_type in set(plan.target_block_types):
            return True
        return block_type in set(plan.target_block_types_soft)
