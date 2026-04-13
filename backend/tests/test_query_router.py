from __future__ import annotations

from backend.src.core.config import Settings
from backend.src.retrieve.query_router import QueryRouter


def test_query_router_routes_precision_queries() -> None:
    router = QueryRouter(Settings())
    decision = router.route("StreamingResponse class path backend/src/api/main.py", requested_top_k=4)

    assert decision.route_name == "precision"
    assert decision.enable_multi_query is False
    assert decision.prefer_lexical is True
    assert decision.retry_policy == "none"
    assert decision.target_block_types == ()


def test_query_router_routes_structure_queries() -> None:
    router = QueryRouter(Settings())
    decision = router.route("Figure 2 on page 12", requested_top_k=4)

    assert decision.route_name == "structure"
    assert decision.preserve_literals is True
    assert decision.max_query_variants == 2
    assert "table" in decision.target_block_types
    assert decision.enforce_targeting is True
    assert decision.retry_policy == "widen_then_keyword"


def test_query_router_routes_explained_structure_queries() -> None:
    router = QueryRouter(Settings())
    decision = router.route("Explain Figure 2 latency tradeoff", requested_top_k=4)

    assert decision.route_name == "explained_structure"
    assert decision.max_query_variants == 3
    assert "paragraph" in decision.target_block_types
    assert decision.retry_policy == "widen_then_keyword"


def test_query_router_routes_general_queries() -> None:
    router = QueryRouter(Settings())
    decision = router.route("How does FastAPI stream answers", requested_top_k=4)

    assert decision.route_name == "general"
    assert decision.enable_multi_query is False
    assert decision.retry_policy == "widen_only"
