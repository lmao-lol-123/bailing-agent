from __future__ import annotations

import re
from dataclasses import dataclass

from backend.src.core.config import Settings
from backend.src.core.text import tokenize_search_text

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_STRUCTURE_PATTERN = re.compile(
    r"\b(?:figure|fig\.?|table|section|page|equation|formula)\b|[图表公式]\s*\d+|第\s*\d+\s*[章节页]",
    re.IGNORECASE,
)
_EXPLAIN_PATTERN = re.compile(
    r"\b(?:explain|meaning|tradeoff|result|results|interpret|interpretation)\b|说明|解释|含义|结果|权衡",
    re.IGNORECASE,
)
_ERROR_PATTERN = re.compile(
    r"\b(?:error|exception|traceback|stack trace|failed|failure|timeout|404|500|importerror|keyerror|valueerror)\b",
    re.IGNORECASE,
)
_BROAD_PATTERN = re.compile(
    r"总结|比较|区别|原理|设计|概览|overview|compare|difference|design|architecture|workflow|summary",
    re.IGNORECASE,
)
_FOLLOWUP_PATTERN = re.compile(
    r"\b(?:it|this|that|these|those|they|them|he|she|there)\b|这个|那个|这些|那些|上面|前面|刚才",
    re.IGNORECASE,
)
_REGULATION_PATTERN = re.compile(
    r"目的|宗旨|适用于|适用范围|英文名称|英文名|主要用于|用于检测|确定什么|承载力指标|不应低于|应满足什么要求|多少天|表\s*\d+(?:\.\d+)+|(?:\d+\.){1,3}\d+条?",
    re.IGNORECASE,
)
_TARGET_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "for",
    "and",
    "or",
    "in",
    "on",
    "at",
    "by",
    "how",
    "what",
    "why",
    "when",
    "where",
    "which",
    "does",
    "do",
    "did",
    "can",
    "could",
    "should",
    "would",
    "please",
    "explain",
    "meaning",
    "tradeoff",
    "result",
    "results",
    "interpret",
    "interpretation",
    "请",
    "帮",
    "一下",
    "怎么",
    "如何",
    "什么",
    "哪个",
    "哪些",
    "以及",
    "或者",
    "还有",
    "和",
    "的",
    "与",
    "及",
    "说明",
    "解释",
    "含义",
    "结果",
    "权衡",
    "图",
    "表",
    "公式",
    "页",
    "节",
    "第",
}


@dataclass(frozen=True)
class QueryAnalysis:
    raw_query: str
    normalized_query: str
    tokens: list[str]
    token_count: int
    has_structure_signal: bool
    has_explain_signal: bool
    has_code_signal: bool
    has_error_signal: bool
    has_broad_intent: bool
    is_short_query: bool
    is_followup_style: bool
    literal_ratio: float
    has_regulation_signal: bool


@dataclass(frozen=True)
class QueryRouteDecision:
    route_name: str
    enable_multi_query: bool
    preserve_literals: bool
    prefer_lexical: bool
    dense_k: int
    lexical_k: int
    fusion_top_k: int
    rerank_top_k: int
    max_query_variants: int
    target_block_types: tuple[str, ...]
    target_block_types_soft: tuple[str, ...]
    target_section_terms: tuple[str, ...]
    enforce_targeting: bool
    retry_policy: str


class QueryRouter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def analyze(self, query: str) -> QueryAnalysis:
        normalized_query = " ".join(query.split())
        raw_tokens = _TOKEN_PATTERN.findall(normalized_query)
        tokens = tokenize_search_text(normalized_query)
        token_count = len(tokens)
        literal_tokens = [token for token in raw_tokens if self._is_literal_token(token)]
        literal_ratio = len(literal_tokens) / token_count if token_count else 0.0
        camel_tokens = [token for token in raw_tokens if self._has_internal_upper(token)]
        has_code_signal = (
            bool(self._contains_hard_code_signal(normalized_query)) or len(camel_tokens) >= 2
        )
        return QueryAnalysis(
            raw_query=query,
            normalized_query=normalized_query,
            tokens=tokens,
            token_count=token_count,
            has_structure_signal=bool(
                self._settings.query_route_structure_pattern_enabled
                and _STRUCTURE_PATTERN.search(normalized_query)
            ),
            has_explain_signal=bool(_EXPLAIN_PATTERN.search(normalized_query)),
            has_code_signal=has_code_signal,
            has_error_signal=bool(_ERROR_PATTERN.search(normalized_query)),
            has_broad_intent=bool(_BROAD_PATTERN.search(normalized_query)),
            is_short_query=token_count <= self._settings.query_route_exploration_token_threshold,
            is_followup_style=bool(_FOLLOWUP_PATTERN.search(normalized_query))
            and token_count <= max(6, self._settings.query_route_exploration_token_threshold + 1),
            literal_ratio=literal_ratio,
            has_regulation_signal=bool(_REGULATION_PATTERN.search(normalized_query)),
        )

    def route(self, query: str, *, requested_top_k: int) -> QueryRouteDecision:
        analysis = self.analyze(query)
        if not self._settings.query_routing_enabled:
            return self._build_general_decision(analysis=analysis, requested_top_k=requested_top_k)

        if analysis.has_regulation_signal:
            return self._build_regulation_decision(
                analysis=analysis, requested_top_k=requested_top_k
            )
        if analysis.has_structure_signal and analysis.has_explain_signal:
            return self._build_explained_structure_decision(
                analysis=analysis, requested_top_k=requested_top_k
            )
        if analysis.has_structure_signal:
            return self._build_structure_decision(
                analysis=analysis, requested_top_k=requested_top_k
            )
        if self._is_precision_query(analysis):
            return self._build_precision_decision(
                analysis=analysis, requested_top_k=requested_top_k
            )
        return self._build_general_decision(analysis=analysis, requested_top_k=requested_top_k)

    def _is_precision_query(self, analysis: QueryAnalysis) -> bool:
        return (
            analysis.has_error_signal
            or analysis.has_code_signal
            or analysis.literal_ratio
            >= self._settings.query_route_precision_literal_ratio_threshold
        )

    def _extract_target_section_terms(self, analysis: QueryAnalysis) -> tuple[str, ...]:
        terms: list[str] = []
        for token in analysis.tokens:
            lowered = token.lower()
            if lowered in _TARGET_STOPWORDS:
                continue
            if len(lowered) <= 1:
                continue
            if lowered not in terms:
                terms.append(lowered)
            if len(terms) >= 4:
                break
        return tuple(terms)

    def _build_precision_decision(
        self, *, analysis: QueryAnalysis, requested_top_k: int
    ) -> QueryRouteDecision:
        rerank_top_k = max(
            requested_top_k * self._settings.rerank_candidate_multiplier, requested_top_k
        )
        return QueryRouteDecision(
            route_name="precision",
            enable_multi_query=False,
            preserve_literals=True,
            prefer_lexical=True,
            dense_k=max(
                requested_top_k * 2, min(self._settings.hybrid_dense_k, max(requested_top_k * 2, 8))
            ),
            lexical_k=max(
                requested_top_k * 3,
                min(self._settings.hybrid_lexical_k, max(requested_top_k * 3, 10)),
            ),
            fusion_top_k=max(requested_top_k * 2, requested_top_k),
            rerank_top_k=rerank_top_k,
            max_query_variants=1,
            target_block_types=(),
            target_block_types_soft=(),
            target_section_terms=self._extract_target_section_terms(analysis),
            enforce_targeting=False,
            retry_policy="none",
        )

    def _build_structure_decision(
        self, *, analysis: QueryAnalysis, requested_top_k: int
    ) -> QueryRouteDecision:
        rerank_top_k = max(
            requested_top_k * self._settings.rerank_candidate_multiplier, requested_top_k
        )
        return QueryRouteDecision(
            route_name="structure",
            enable_multi_query=self._settings.query_multi_enabled,
            preserve_literals=True,
            prefer_lexical=True,
            dense_k=max(
                requested_top_k * 2, min(self._settings.hybrid_dense_k, max(requested_top_k * 2, 8))
            ),
            lexical_k=max(
                requested_top_k * 4,
                min(self._settings.hybrid_lexical_k, max(requested_top_k * 4, 12)),
            ),
            fusion_top_k=max(requested_top_k * 2, requested_top_k),
            rerank_top_k=rerank_top_k,
            max_query_variants=min(2, self._settings.query_multi_max_variants),
            target_block_types=("table", "image", "formula", "caption", "section_header"),
            target_block_types_soft=("paragraph",),
            target_section_terms=self._extract_target_section_terms(analysis),
            enforce_targeting=self._settings.targeted_retrieval_enabled,
            retry_policy="widen_then_keyword" if self._settings.targeted_retry_enabled else "none",
        )

    def _build_regulation_decision(
        self, *, analysis: QueryAnalysis, requested_top_k: int
    ) -> QueryRouteDecision:
        rerank_top_k = max(
            requested_top_k * self._settings.rerank_candidate_multiplier, requested_top_k
        )
        return QueryRouteDecision(
            route_name="regulation",
            enable_multi_query=self._settings.query_multi_enabled,
            preserve_literals=True,
            prefer_lexical=True,
            dense_k=max(
                requested_top_k * 2, min(self._settings.hybrid_dense_k, max(requested_top_k * 2, 8))
            ),
            lexical_k=max(
                requested_top_k * 4,
                min(self._settings.hybrid_lexical_k, max(requested_top_k * 4, 12)),
            ),
            fusion_top_k=max(requested_top_k * 2, requested_top_k),
            rerank_top_k=rerank_top_k,
            max_query_variants=min(2, self._settings.query_multi_max_variants),
            target_block_types=(),
            target_block_types_soft=(),
            target_section_terms=self._extract_target_section_terms(analysis),
            enforce_targeting=False,
            retry_policy="widen_then_keyword" if self._settings.targeted_retry_enabled else "none",
        )

    def _build_explained_structure_decision(
        self, *, analysis: QueryAnalysis, requested_top_k: int
    ) -> QueryRouteDecision:
        rerank_top_k = max(
            requested_top_k * self._settings.rerank_candidate_multiplier, requested_top_k
        )
        return QueryRouteDecision(
            route_name="explained_structure",
            enable_multi_query=self._settings.query_multi_enabled,
            preserve_literals=True,
            prefer_lexical=True,
            dense_k=max(
                requested_top_k * 2, min(self._settings.hybrid_dense_k, max(requested_top_k * 2, 8))
            ),
            lexical_k=max(
                requested_top_k * 4,
                min(self._settings.hybrid_lexical_k, max(requested_top_k * 4, 12)),
            ),
            fusion_top_k=max(requested_top_k * 2, requested_top_k),
            rerank_top_k=rerank_top_k,
            max_query_variants=min(3, self._settings.query_multi_max_variants),
            target_block_types=("table", "image", "formula", "caption", "paragraph"),
            target_block_types_soft=("section_header",),
            target_section_terms=self._extract_target_section_terms(analysis),
            enforce_targeting=self._settings.targeted_retrieval_enabled,
            retry_policy="widen_then_keyword" if self._settings.targeted_retry_enabled else "none",
        )

    def _build_general_decision(
        self, *, analysis: QueryAnalysis, requested_top_k: int
    ) -> QueryRouteDecision:
        rerank_top_k = max(
            requested_top_k * self._settings.rerank_candidate_multiplier, requested_top_k
        )
        return QueryRouteDecision(
            route_name="general",
            enable_multi_query=False,
            preserve_literals=False,
            prefer_lexical=False,
            dense_k=max(self._settings.hybrid_dense_k, rerank_top_k),
            lexical_k=max(self._settings.hybrid_lexical_k, rerank_top_k),
            fusion_top_k=max(self._settings.hybrid_fusion_top_k, rerank_top_k),
            rerank_top_k=max(self._settings.hybrid_fusion_top_k, rerank_top_k),
            max_query_variants=1,
            target_block_types=(),
            target_block_types_soft=(),
            target_section_terms=self._extract_target_section_terms(analysis),
            enforce_targeting=False,
            retry_policy="widen_then_keyword" if self._settings.targeted_retry_enabled else "none",
        )

    @staticmethod
    def _is_literal_token(token: str) -> bool:
        return (
            "/" in token
            or "\\" in token
            or "." in token
            or "_" in token
            or any(char.isdigit() for char in token)
            or QueryRouter._has_internal_upper(token)
        )

    @staticmethod
    def _contains_hard_code_signal(text: str) -> bool:
        return bool(
            re.search(
                r"[/\\]|::|->|[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.]+|[A-Za-z_][A-Za-z0-9_]*\(|[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+",
                text,
            )
        )

    @staticmethod
    def _has_internal_upper(token: str) -> bool:
        return any(char.isupper() for char in token[1:]) and any(char.islower() for char in token)
