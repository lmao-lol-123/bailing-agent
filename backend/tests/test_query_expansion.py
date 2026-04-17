from __future__ import annotations

from backend.src.core.config import Settings
from backend.src.retrieve.query_expansion import DeterministicQueryExpander
from backend.src.retrieve.query_router import QueryRouter


def test_query_expander_keeps_precision_query_as_original_only() -> None:
    settings = Settings(query_multi_enabled=True)
    router = QueryRouter(settings)
    expander = DeterministicQueryExpander(settings)
    query = "StreamingResponse class path backend/src/api/main.py"
    analysis = router.analyze(query)
    decision = router.route(query, requested_top_k=4)

    variants = expander.expand(query=query, analysis=analysis, decision=decision)

    assert [variant.variant_id for variant in variants] == ["original"]


def test_query_expander_generates_structure_variants_conservatively() -> None:
    settings = Settings(query_multi_enabled=True)
    router = QueryRouter(settings)
    expander = DeterministicQueryExpander(settings)
    query = "Figure 2 on page 12"
    analysis = router.analyze(query)
    decision = router.route(query, requested_top_k=4)

    variants = expander.expand(query=query, analysis=analysis, decision=decision)

    assert [variant.variant_id for variant in variants] == ["original", "keyword_focused"]


def test_query_expander_generates_explained_structure_variants() -> None:
    settings = Settings(query_multi_enabled=True, query_multi_max_variants=4)
    router = QueryRouter(settings)
    expander = DeterministicQueryExpander(settings)
    query = "Explain Figure 2 latency tradeoff"
    analysis = router.analyze(query)
    decision = router.route(query, requested_top_k=4)

    variants = expander.expand(query=query, analysis=analysis, decision=decision)

    assert [variant.variant_id for variant in variants] == [
        "original",
        "keyword_focused",
        "clarified",
    ]
    assert variants[0].text == query


def test_query_expander_keeps_general_query_as_original_only() -> None:
    settings = Settings(query_multi_enabled=True, query_multi_max_variants=4)
    router = QueryRouter(settings)
    expander = DeterministicQueryExpander(settings)
    query = "How does FastAPI stream answers"
    analysis = router.analyze(query)
    decision = router.route(query, requested_top_k=4)

    variants = expander.expand(query=query, analysis=analysis, decision=decision)

    assert [variant.variant_id for variant in variants] == ["original"]


def test_query_expander_builds_retry_keyword_variant() -> None:
    settings = Settings(query_multi_enabled=True)
    router = QueryRouter(settings)
    expander = DeterministicQueryExpander(settings)
    query = "Explain Figure 2 latency tradeoff"
    analysis = router.analyze(query)

    variant = expander.build_retry_variant(query=query, analysis=analysis)

    assert variant is not None
    assert variant.variant_id == "retry_keyword"
    assert "Figure" in variant.text


def test_query_expander_generates_regulation_variants() -> None:
    settings = Settings(query_multi_enabled=True, query_multi_max_variants=4)
    router = QueryRouter(settings)
    expander = DeterministicQueryExpander(settings)
    query = "本规范适用于哪些内容的检测与评价？"
    analysis = router.analyze(query)
    decision = router.route(query, requested_top_k=4)

    variants = expander.expand(query=query, analysis=analysis, decision=decision)

    assert [variant.variant_id for variant in variants] == ["original", "keyword_focused"]
    assert "适用" in variants[1].text
