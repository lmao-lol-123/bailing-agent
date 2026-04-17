from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from backend.src.core.config import Settings, get_settings
from backend.src.core.text import normalize_text
from backend.src.eval.dataset import EvalSample, load_eval_samples
from backend.src.retrieve.providers import build_embedding_model

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def compute_citation_correctness(
    *,
    answer_text: str,
    citations: list[dict[str, Any]],
    retrieved_documents: list[Document],
) -> tuple[float, dict[str, Any]]:
    extracted = [int(match.group(1)) for match in _CITATION_PATTERN.finditer(answer_text)]
    invalid_indices = [index for index in extracted if index < 1 or index > len(citations)]
    retrieved_doc_ids = {
        str((document.metadata or {}).get("doc_id") or "") for document in retrieved_documents
    }
    citation_doc_ids = [str(item.get("doc_id") or "") for item in citations]
    unmapped_doc_ids = [
        doc_id for doc_id in citation_doc_ids if doc_id and doc_id not in retrieved_doc_ids
    ]
    score = 1.0 if not invalid_indices and not unmapped_doc_ids else 0.0
    return score, {
        "score": score,
        "invalid_indices": sorted(set(invalid_indices)),
        "citation_count": len(citations),
        "mentioned_citation_count": len(extracted),
        "unmapped_doc_ids": sorted(set(unmapped_doc_ids)),
    }


async def _run_answer(
    *,
    container: Any,
    session_id: str,
    question: str,
    top_k: int,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    answer_parts: list[str] = []
    citations: list[dict[str, Any]] = []
    done_payload: dict[str, Any] = {}
    async for event in container.answer_service.stream_answer(
        session_id=session_id,
        question=question,
        top_k=top_k,
    ):
        if event["event"] == "token":
            answer_parts.append(str(event["data"].get("text", "")))
        elif event["event"] == "sources":
            citations = [dict(item) for item in event["data"].get("citations", [])]
        elif event["event"] == "done":
            done_payload = dict(event["data"])
    return "".join(answer_parts), citations, done_payload


def _build_main_dataset_rows(
    *,
    sample: EvalSample,
    answer_text: str,
    retrieved_documents: list[Document],
) -> dict[str, Any]:
    reference_answer = sample.reference_answer or sample.expected_snippet
    contexts = [document.page_content for document in retrieved_documents]
    return {
        "question": sample.question,
        "answer": answer_text,
        "contexts": contexts,
        "ground_truth": reference_answer,
    }


def _compute_expected_snippet_topk_hit(
    *, sample: EvalSample, retrieved_documents: list[Document]
) -> bool:
    expected_snippet = _normalize_eval_match_text(sample.expected_snippet or "")
    if not expected_snippet:
        return False
    for document in retrieved_documents:
        content_candidates = [
            _normalize_eval_match_text(document.page_content),
            _normalize_eval_match_text(str((document.metadata or {}).get("child_content") or "")),
        ]
        if any(
            expected_snippet in candidate or candidate in expected_snippet
            for candidate in content_candidates
            if candidate
        ):
            return True
    return False


def _compute_anchor_hit(*, retrieved_documents: list[Document]) -> bool:
    for document in retrieved_documents:
        metadata = document.metadata or {}
        if (
            metadata.get("is_regulation_anchor")
            or metadata.get("clause_id")
            or metadata.get("table_id")
            or metadata.get("english_alias")
            or metadata.get("has_numeric_anchor")
        ):
            return True
    return False


def _summarize_route_names(sample_reports: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in sample_reports:
        route_name = str((item.get("retrieval") or {}).get("route_name") or "unknown")
        counts[route_name] = counts.get(route_name, 0) + 1
    return dict(sorted(counts.items()))


def _normalize_eval_match_text(value: str) -> str:
    normalized = normalize_text(value)
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized).lower()


def _run_ragas_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "available": False,
            "error": "no_ok_samples",
            "metrics": {},
        }
    try:
        from datasets import Dataset
        from ragas import evaluate
    except Exception as exc:  # pragma: no cover - dependency/runtime guard
        return {
            "available": False,
            "error": f"ragas_import_failed: {exc}",
            "metrics": {},
        }

    dataset = Dataset.from_dict(
        {
            "question": [item["question"] for item in rows],
            "answer": [item["answer"] for item in rows],
            "contexts": [item["contexts"] for item in rows],
            "ground_truth": [item["ground_truth"] for item in rows],
        }
    )
    try:
        evaluator_llm, evaluator_embeddings, metrics = _build_ragas_runtime(get_settings())
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        return {
            "available": True,
            "error": None,
            "metrics": result.to_pandas().mean(numeric_only=True).to_dict(),
        }
    except Exception as exc:  # pragma: no cover - external model/runtime guard
        return {
            "available": False,
            "error": f"ragas_evaluate_failed: {exc}",
            "metrics": {},
        }


def _summarize_fallback(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for item in rows:
        status = str(item.get("status") or "unknown")
        reason = str(item.get("reason") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "count": len(rows),
        "status_counts": status_counts,
        "reason_counts": reason_counts,
    }


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _normalize_openai_compatible_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _build_ragas_runtime(
    settings: Settings,
    *,
    chat_model_cls: Any | None = None,
    llm_wrapper_cls: Any | None = None,
    embeddings_wrapper_cls: Any | None = None,
    metric_factories: list[Any] | None = None,
    embedding_model: Any | None = None,
) -> tuple[Any, Any, list[Any]]:
    if (
        chat_model_cls is None
        or llm_wrapper_cls is None
        or embeddings_wrapper_cls is None
        or metric_factories is None
    ):
        from langchain_openai import ChatOpenAI
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

        chat_model_cls = chat_model_cls or ChatOpenAI
        llm_wrapper_cls = llm_wrapper_cls or LangchainLLMWrapper
        embeddings_wrapper_cls = embeddings_wrapper_cls or LangchainEmbeddingsWrapper
        metric_factories = metric_factories or [
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
        ]

    resolved_embedding_model = embedding_model or build_embedding_model(settings)
    evaluator_llm = llm_wrapper_cls(
        chat_model_cls(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=_normalize_openai_compatible_base_url(settings.deepseek_base_url),
            temperature=0,
        )
    )
    evaluator_embeddings = embeddings_wrapper_cls(resolved_embedding_model)
    metrics = [factory() for factory in metric_factories]
    return evaluator_llm, evaluator_embeddings, metrics


async def _run_eval(
    *,
    dataset_path: Path,
    output_path: Path,
    top_k: int,
) -> dict[str, Any]:
    from backend.src.core.dependencies import get_container

    container = get_container()
    samples = load_eval_samples(dataset_path)
    ok_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    sample_reports: list[dict[str, Any]] = []
    citation_scores_all: list[float] = []
    citation_scores_ok: list[float] = []
    expected_snippet_hits: list[float] = []
    anchor_hits: list[float] = []

    for index, sample in enumerate(samples):
        session_id = f"ragas-eval-{index + 1}"
        retrieved_documents, retrieval_trace = container.vector_store.similarity_search_with_trace(
            query=sample.question,
            k=top_k,
            metadata_filter=None,
        )
        answer_text, citations, done_payload = await _run_answer(
            container=container,
            session_id=session_id,
            question=sample.question,
            top_k=top_k,
        )
        status = str(done_payload.get("status") or "ok")
        reason = str(done_payload.get("reason") or "normal")

        citation_score, citation_detail = compute_citation_correctness(
            answer_text=answer_text,
            citations=citations,
            retrieved_documents=retrieved_documents,
        )
        citation_scores_all.append(citation_score)
        if status == "ok":
            citation_scores_ok.append(citation_score)
        expected_snippet_top4_hit = _compute_expected_snippet_topk_hit(
            sample=sample, retrieved_documents=retrieved_documents
        )
        anchor_hit = _compute_anchor_hit(retrieved_documents=retrieved_documents)
        expected_snippet_hits.append(1.0 if expected_snippet_top4_hit else 0.0)
        anchor_hits.append(1.0 if anchor_hit else 0.0)

        sample_report = {
            "question": sample.question,
            "status": status,
            "reason": reason,
            "answer_preview": answer_text[:240],
            "answer_length": len(answer_text),
            "answer_hash": hashlib.sha256(answer_text.encode("utf-8")).hexdigest(),
            "citation_correctness": citation_detail,
            "retrieval": {
                "route_name": (retrieval_trace.get("request") or {}).get("route_name"),
                "retry_reason": (retrieval_trace.get("request") or {}).get("retry_reason"),
                "context_refs": [
                    {
                        "chunk_id": str(
                            (document.metadata or {}).get("chunk_id")
                            or getattr(document, "id", "")
                            or ""
                        ),
                        "parent_chunk_id": str(
                            (document.metadata or {}).get("parent_chunk_id") or ""
                        ),
                        "doc_id": str((document.metadata or {}).get("doc_id") or ""),
                        "source_name": str((document.metadata or {}).get("source_name") or ""),
                    }
                        for document in retrieved_documents
                    ],
                "expected_snippet_top4_hit": expected_snippet_top4_hit,
                "anchor_hit": anchor_hit,
            },
        }
        sample_reports.append(sample_report)

        if status == "ok":
            ok_rows.append(
                _build_main_dataset_rows(
                    sample=sample,
                    answer_text=answer_text,
                    retrieved_documents=retrieved_documents,
                )
            )
        else:
            fallback_rows.append(sample_report)

    ragas_result = _run_ragas_metrics(ok_rows)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "sample_count": len(samples),
        "ok_count": len(ok_rows),
        "fallback_count": len(fallback_rows),
        "citation_correctness_avg_all": _safe_mean(citation_scores_all),
        "citation_correctness_avg_ok": _safe_mean(citation_scores_ok),
        "ragas": ragas_result,
        "retrieval_metrics": {
            "expected_snippet_top4_hit_rate": _safe_mean(expected_snippet_hits),
            "anchor_hit_rate": _safe_mean(anchor_hits),
            "route_name_distribution": _summarize_route_names(sample_reports),
        },
        "fallback_summary": _summarize_fallback(fallback_rows),
        "samples": sample_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_to_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline RAGAS evaluation for the current RAG pipeline."
    )
    parser.add_argument(
        "--dataset",
        default="backend/tests/fixtures/eval_dataset.json",
        help="Eval dataset JSON path.",
    )
    parser.add_argument("--out", default=None, help="Output report JSON path.")
    parser.add_argument("--top-k", type=int, default=4, help="Top-k retrieval size for evaluation.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = (
        Path(args.out)
        if args.out
        else Path("storage/eval")
        / f"ragas-report-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
    )
    try:
        report = asyncio.run(
            _run_eval(
                dataset_path=dataset_path,
                output_path=output_path,
                top_k=args.top_k,
            )
        )
    except Exception as exc:
        failure_payload = {
            "run_status": "failed",
            "exit_code": 2,
            "error": str(exc),
            "dataset_path": str(dataset_path),
            "report_path": str(output_path),
        }
        print(json.dumps(failure_payload, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    ragas_available = bool(report["ragas"]["available"])
    if ragas_available and report["ok_count"] == report["sample_count"]:
        run_status = "ok"
        exit_code = 0
    else:
        run_status = "partial"
        exit_code = 1

    print(
        json.dumps(
            {
                "run_status": run_status,
                "exit_code": exit_code,
                "report_path": str(output_path),
                "sample_count": report["sample_count"],
                "ok_count": report["ok_count"],
                "fallback_count": report["fallback_count"],
                "citation_correctness_avg_all": report["citation_correctness_avg_all"],
                "ragas_available": report["ragas"]["available"],
                "ragas_error": report["ragas"]["error"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
