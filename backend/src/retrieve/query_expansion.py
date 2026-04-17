from __future__ import annotations

import re
from dataclasses import dataclass

from backend.src.core.config import Settings
from backend.src.core.text import extract_regulation_anchors
from backend.src.retrieve.query_router import QueryAnalysis, QueryRouteDecision

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_REGULATION_KEYWORD_PATTERN = re.compile(
    r"表\s*\d+(?:\.\d+)+|(?:\d+\.){1,3}\d+条?|低应变法|高应变法|声波透射法|本规范|适用于|目的|英文名称|休止时间|承载力|桩身完整性|混凝土强度|设计强度|砂土|MPa|天",
    re.IGNORECASE,
)
_STOPWORDS = {
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
    "比较",
    "说明",
    "解释",
    "含义",
    "结果",
    "权衡",
}


@dataclass(frozen=True)
class QueryVariant:
    variant_id: str
    text: str
    kind: str
    weight: float


class DeterministicQueryExpander:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def expand(
        self, *, query: str, analysis: QueryAnalysis, decision: QueryRouteDecision
    ) -> list[QueryVariant]:
        original_query = analysis.normalized_query or query.strip()
        variants: list[QueryVariant] = [
            QueryVariant(variant_id="original", text=original_query, kind="original", weight=1.0)
        ]

        if (
            not self._settings.query_multi_enabled
            or not decision.enable_multi_query
            or decision.max_query_variants <= 1
        ):
            return variants

        keyword_text = self.build_keyword_variant_text(analysis=analysis)
        if keyword_text and keyword_text.lower() != original_query.lower():
            variants.append(
                QueryVariant(
                    variant_id="keyword_focused",
                    text=keyword_text,
                    kind="keyword_focused",
                    weight=0.92,
                )
            )

        if decision.route_name == "explained_structure":
            clarified_text = self._build_clarified(
                original_query=original_query, keyword_text=keyword_text
            )
            if clarified_text and clarified_text.lower() not in {
                variant.text.lower() for variant in variants
            }:
                variants.append(
                    QueryVariant(
                        variant_id="clarified", text=clarified_text, kind="clarified", weight=0.85
                    )
                )

        return variants[: decision.max_query_variants]

    def build_retry_variant(self, *, query: str, analysis: QueryAnalysis) -> QueryVariant | None:
        keyword_text = self.build_keyword_variant_text(analysis=analysis)
        normalized_query = analysis.normalized_query or query.strip()
        if not keyword_text or keyword_text.lower() == normalized_query.lower():
            return None
        return QueryVariant(
            variant_id="retry_keyword", text=keyword_text, kind="retry_keyword", weight=0.84
        )

    def build_keyword_variant_text(self, *, analysis: QueryAnalysis) -> str:
        if not analysis.tokens:
            return ""
        if analysis.has_regulation_signal:
            anchors = extract_regulation_anchors(analysis.normalized_query)
            terms: list[str] = []
            for value in (
                anchors.get("clause_id"),
                anchors.get("table_id"),
                f"表{anchors['table_id']}" if anchors.get("table_id") else None,
            ):
                if value and value not in terms:
                    terms.append(str(value))
            for match in _REGULATION_KEYWORD_PATTERN.finditer(analysis.normalized_query):
                term = match.group(0).strip()
                if term and term not in terms:
                    terms.append(term)
            if terms:
                return " ".join(terms)
        filtered_tokens: list[str] = []
        for token in _TOKEN_PATTERN.findall(analysis.normalized_query):
            lowered = token.lower()
            if lowered in _STOPWORDS:
                continue
            filtered_tokens.append(token)
        if not filtered_tokens:
            filtered_tokens = _TOKEN_PATTERN.findall(analysis.normalized_query)
        return " ".join(filtered_tokens)

    def _build_clarified(self, *, original_query: str, keyword_text: str) -> str:
        if not keyword_text or keyword_text.lower() == original_query.lower():
            return original_query
        return f"{original_query} {keyword_text}".strip()
