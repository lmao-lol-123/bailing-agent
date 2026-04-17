from __future__ import annotations

from langchain_core.documents import Document

from backend.scripts.run_ragas_eval import (
    _build_ragas_runtime,
    _compute_anchor_hit,
    _compute_expected_snippet_topk_hit,
    _normalize_openai_compatible_base_url,
    _summarize_route_names,
    compute_citation_correctness,
)
from backend.src.core.config import Settings
from backend.src.eval.dataset import EvalSample


def test_compute_citation_correctness_passes_for_valid_indices_and_doc_mapping() -> None:
    documents = [
        Document(page_content="parent context", metadata={"doc_id": "doc-1"}),
        Document(page_content="parent context 2", metadata={"doc_id": "doc-2"}),
    ]
    citations = [
        {"index": 1, "doc_id": "doc-1"},
        {"index": 2, "doc_id": "doc-2"},
    ]

    score, detail = compute_citation_correctness(
        answer_text="结论A[1]，结论B[2]。",
        citations=citations,
        retrieved_documents=documents,
    )

    assert score == 1.0
    assert detail["invalid_indices"] == []
    assert detail["unmapped_doc_ids"] == []


def test_compute_citation_correctness_fails_for_out_of_range_or_unmapped_docs() -> None:
    documents = [
        Document(page_content="parent context", metadata={"doc_id": "doc-1"}),
    ]
    citations = [
        {"index": 1, "doc_id": "doc-missing"},
    ]

    score, detail = compute_citation_correctness(
        answer_text="结论A[2]。",
        citations=citations,
        retrieved_documents=documents,
    )

    assert score == 0.0
    assert detail["invalid_indices"] == [2]
    assert detail["unmapped_doc_ids"] == ["doc-missing"]


def test_normalize_openai_compatible_base_url_appends_v1_suffix() -> None:
    assert _normalize_openai_compatible_base_url("https://api.deepseek.com") == (
        "https://api.deepseek.com/v1"
    )
    assert _normalize_openai_compatible_base_url("https://api.deepseek.com/") == (
        "https://api.deepseek.com/v1"
    )
    assert _normalize_openai_compatible_base_url("https://api.deepseek.com/v1") == (
        "https://api.deepseek.com/v1"
    )


def test_build_ragas_runtime_uses_deepseek_llm_and_project_embeddings() -> None:
    calls: dict[str, object] = {}

    class FakeChatModel:
        def __init__(self, **kwargs) -> None:
            calls["chat_kwargs"] = kwargs

    class FakeLLMWrapper:
        def __init__(self, llm) -> None:
            calls["wrapped_llm"] = llm

    class FakeEmbeddingsWrapper:
        def __init__(self, embeddings) -> None:
            calls["wrapped_embeddings"] = embeddings

    class FakeMetric:
        def __init__(self, name: str) -> None:
            self.name = name

    settings = Settings(
        deepseek_api_key="deepseek-key",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
    )
    embedding_model = object()

    llm_wrapper, embedding_wrapper, metrics = _build_ragas_runtime(
        settings,
        chat_model_cls=FakeChatModel,
        llm_wrapper_cls=FakeLLMWrapper,
        embeddings_wrapper_cls=FakeEmbeddingsWrapper,
        metric_factories=[
            lambda: FakeMetric("faithfulness"),
            lambda: FakeMetric("answer_relevancy"),
        ],
        embedding_model=embedding_model,
    )

    assert isinstance(llm_wrapper, FakeLLMWrapper)
    assert isinstance(embedding_wrapper, FakeEmbeddingsWrapper)
    assert [metric.name for metric in metrics] == ["faithfulness", "answer_relevancy"]
    assert calls["chat_kwargs"] == {
        "model": "deepseek-chat",
        "api_key": "deepseek-key",
        "base_url": "https://api.deepseek.com/v1",
        "temperature": 0,
    }
    assert calls["wrapped_embeddings"] is embedding_model


def test_ragas_eval_helpers_track_expected_snippet_anchor_and_route_metrics() -> None:
    sample = EvalSample(
        question="本规范适用于哪些内容的检测与评价？",
        expected_source_name="spec.pdf",
        expected_snippet="本规范适用于建筑工程基桩的承载力和桩身完整性的检测与评价。",
        reference_answer="本规范适用于建筑工程基桩承载力和桩身完整性的检测与评价。",
        allow_uncertain=False,
    )
    documents = [
        Document(
            page_content="`1.0.2` 本规范适用于建筑工程基桩的承载力和桩身完整性的检测与评价。",
            metadata={"clause_id": "1.0.2", "is_regulation_anchor": True},
        )
    ]

    assert _compute_expected_snippet_topk_hit(sample=sample, retrieved_documents=documents) is True
    assert _compute_anchor_hit(retrieved_documents=documents) is True
    assert _summarize_route_names(
        [
            {"retrieval": {"route_name": "regulation"}},
            {"retrieval": {"route_name": "regulation"}},
            {"retrieval": {"route_name": "general"}},
        ]
    ) == {"general": 1, "regulation": 2}
