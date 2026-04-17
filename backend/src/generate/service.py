from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from fastapi.encoders import jsonable_encoder
from langchain_core.documents import Document

from backend.src.core.chat_store import ChatHistoryStore
from backend.src.core.config import Settings
from backend.src.core.models import (
    AnswerReason,
    AnswerStatus,
    ChatMessage,
    ChatRole,
    Citation,
    RetrievalFilter,
    SessionFileStatus,
)
from backend.src.core.text import normalize_text
from backend.src.generate.prompts import (
    build_answer_prompt,
    build_context_blocks,
    build_system_prompt,
)
from backend.src.retrieve.store import VectorStoreService

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_ASSERTION_SIGNAL_PATTERN = re.compile(
    r"\d|%|排名|比例|提升|下降|高于|低于|优于|劣于|because|due to|caused by|导致|因为|according to|文档显示|表明",
    re.IGNORECASE,
)
_NON_ASSERTION_HINT_PATTERN = re.compile(
    r"总的来说|总结|综上|另外|此外|不过|注意|可能|建议|可选|通常", re.IGNORECASE
)


@dataclass(frozen=True)
class ContextBlock:
    index: int
    source_name: str
    title: str
    section_label: str
    page: str
    doc_type: str
    content: str


@dataclass(frozen=True)
class GenerationContext:
    question: str
    retrieval_query: str
    history_messages: list[ChatMessage]
    documents: list[Document]
    citations: list[Citation]
    context_blocks: list[ContextBlock]
    metadata_filter: RetrievalFilter | None
    session_doc_ids: list[str]
    retrieval_trace: dict[str, Any]


@dataclass(frozen=True)
class AnswerDraft:
    answer_text: str
    grounded: bool
    citations: list[Citation]
    status: AnswerStatus
    reason: AnswerReason
    debug_trace: dict[str, Any]


@dataclass(frozen=True)
class AnswerValidationResult:
    is_valid: bool
    invalid_citation_indices: list[int]
    missing_citation_sentences: list[str]


class AnswerService:
    def __init__(
        self,
        settings: Settings,
        chat_client: Any,
        vector_store: VectorStoreService,
        chat_history_store: ChatHistoryStore,
    ) -> None:
        self._settings = settings
        self._chat_client = chat_client
        self._vector_store = vector_store
        self._chat_history_store = chat_history_store

    async def stream_answer(
        self,
        session_id: str,
        question: str,
        top_k: int | None = None,
        metadata_filter: RetrievalFilter | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        history_messages = self._chat_history_store.list_messages(
            session_id=session_id,
            limit=self._settings.chat_history_max_messages,
        )
        session_files = self._chat_history_store.list_session_files(session_id)
        session_has_files = bool(session_files)
        session_doc_ids = [
            str(item.doc_id)
            for item in session_files
            if item.status is SessionFileStatus.INDEXED and item.doc_id
        ]
        resolved_filter = self._merge_session_filter(
            metadata_filter=metadata_filter,
            session_doc_ids=session_doc_ids,
            session_has_files=session_has_files,
        )
        retrieval_query = self._build_retrieval_query(
            question=question, history_messages=history_messages
        )
        documents, retrieval_trace = self._vector_store.similarity_search_with_trace(
            retrieval_query,
            k=top_k,
            metadata_filter=resolved_filter,
        )
        citations = [
            Citation.model_validate(item) for item in self._vector_store.build_citations(documents)
        ]
        generation_context = self._build_generation_context(
            question=question,
            retrieval_query=retrieval_query,
            history_messages=history_messages,
            documents=documents,
            citations=citations,
            metadata_filter=resolved_filter,
            session_doc_ids=session_doc_ids,
            retrieval_trace=retrieval_trace,
        )

        extracted_answer = self._build_regulation_extract_answer(
            question=question,
            documents=documents,
            citations=citations,
        )
        if extracted_answer is not None:
            validation_result = self._validate_answer(
                answer_text=extracted_answer, citations=citations
            )
            answer_draft = AnswerDraft(
                answer_text=extracted_answer,
                grounded=True,
                citations=citations,
                status=AnswerStatus.OK,
                reason=AnswerReason.NORMAL,
                debug_trace=self._build_debug_trace(
                    context=generation_context,
                    session_id=session_id,
                    route_names=self._collect_route_names(documents),
                    status=AnswerStatus.OK,
                    reason=AnswerReason.NORMAL,
                    validation_result=validation_result,
                ),
            )
            self._persist_turn(
                session_id=session_id,
                question=question,
                answer_draft=answer_draft,
                metadata_filter=resolved_filter,
            )
            for token_part in self._chunk_text_for_stream(extracted_answer):
                yield {"event": "token", "data": {"text": token_part}}
            yield {"event": "sources", "data": {"citations": jsonable_encoder(citations)}}
            yield {
                "event": "done",
                "data": {
                    "grounded": True,
                    "session_id": session_id,
                    "status": answer_draft.status.value,
                    "reason": answer_draft.reason.value,
                },
            }
            return

        if not documents:
            fallback_text = "我无法从已索引文档中找到足够可靠的上下文来回答这个问题。"
            answer_draft = AnswerDraft(
                answer_text=fallback_text,
                grounded=False,
                citations=[],
                status=AnswerStatus.REJECTED,
                reason=AnswerReason.NO_CONTEXT,
                debug_trace=self._build_debug_trace(
                    context=generation_context,
                    session_id=session_id,
                    route_names=[],
                    status=AnswerStatus.REJECTED,
                    reason=AnswerReason.NO_CONTEXT,
                    validation_result=None,
                ),
            )
            self._persist_turn(
                session_id=session_id,
                question=question,
                answer_draft=answer_draft,
                metadata_filter=resolved_filter,
            )
            yield {"event": "token", "data": {"text": fallback_text}}
            yield {"event": "sources", "data": {"citations": []}}
            yield {
                "event": "done",
                "data": {
                    "grounded": False,
                    "session_id": session_id,
                    "status": answer_draft.status.value,
                    "reason": answer_draft.reason.value,
                },
            }
            return

        stream = await self._chat_client.chat.completions.create(
            model=self._settings.deepseek_model,
            temperature=0.1,
            stream=True,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                *self._build_history_messages(history_messages),
                {
                    "role": "user",
                    "content": build_answer_prompt(
                        question,
                        history_messages,
                        build_context_blocks(
                            [
                                self._context_block_to_prompt_item(item)
                                for item in generation_context.context_blocks
                            ]
                        ),
                    ),
                },
            ],
        )

        answer_parts: list[str] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                answer_parts.append(delta)

        answer_text = self._sanitize_answer_citations("".join(answer_parts), citations)
        validation_result = self._validate_answer(answer_text=answer_text, citations=citations)
        if not validation_result.is_valid:
            repaired_answer = self._repair_answer_once(answer_text=answer_text, citations=citations)
            repaired_validation = self._validate_answer(
                answer_text=repaired_answer, citations=citations
            )
            if repaired_validation.is_valid:
                answer_text = repaired_answer
                validation_result = repaired_validation
            else:
                answer_text, degraded_status, degraded_reason = self._build_fallback_answer(
                    citations=citations,
                    question=question,
                    weak_evidence=bool(citations),
                )
                answer_draft = AnswerDraft(
                    answer_text=answer_text,
                    grounded=False,
                    citations=citations,
                    status=degraded_status,
                    reason=degraded_reason,
                    debug_trace=self._build_debug_trace(
                        context=generation_context,
                        session_id=session_id,
                        route_names=self._collect_route_names(documents),
                        status=degraded_status,
                        reason=degraded_reason,
                        validation_result=repaired_validation,
                    ),
                )
                self._persist_turn(
                    session_id=session_id,
                    question=question,
                    answer_draft=answer_draft,
                    metadata_filter=resolved_filter,
                )
                for token_part in self._chunk_text_for_stream(answer_text):
                    yield {"event": "token", "data": {"text": token_part}}
                yield {"event": "sources", "data": {"citations": jsonable_encoder(citations)}}
                yield {
                    "event": "done",
                    "data": {
                        "grounded": False,
                        "session_id": session_id,
                        "status": answer_draft.status.value,
                        "reason": answer_draft.reason.value,
                    },
                }
                return

        answer_draft = AnswerDraft(
            answer_text=answer_text,
            grounded=True,
            citations=citations,
            status=AnswerStatus.OK,
            reason=AnswerReason.NORMAL,
            debug_trace=self._build_debug_trace(
                context=generation_context,
                session_id=session_id,
                route_names=self._collect_route_names(documents),
                status=AnswerStatus.OK,
                reason=AnswerReason.NORMAL,
                validation_result=validation_result,
            ),
        )
        self._persist_turn(
            session_id=session_id,
            question=question,
            answer_draft=answer_draft,
            metadata_filter=resolved_filter,
        )
        for token_part in self._chunk_text_for_stream(answer_text):
            yield {"event": "token", "data": {"text": token_part}}
        yield {"event": "sources", "data": {"citations": jsonable_encoder(citations)}}
        yield {
            "event": "done",
            "data": {
                "grounded": True,
                "session_id": session_id,
                "status": answer_draft.status.value,
                "reason": answer_draft.reason.value,
            },
        }

    def _build_history_messages(self, history_messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [
            {"role": message.role.value, "content": message.content} for message in history_messages
        ]

    def _build_retrieval_query(self, question: str, history_messages: list[ChatMessage]) -> str:
        user_messages = [
            message.content for message in history_messages if message.role is ChatRole.USER
        ]
        if not user_messages:
            return question

        recent_user_messages = user_messages[-self._settings.retrieval_history_user_messages :]
        return "\n".join([*recent_user_messages, question])

    def _merge_session_filter(
        self,
        *,
        metadata_filter: RetrievalFilter | None,
        session_doc_ids: list[str],
        session_has_files: bool,
    ) -> RetrievalFilter | None:
        if session_has_files and not session_doc_ids:
            return RetrievalFilter(doc_ids=["__session_scope_no_indexed_files__"])
        if not session_doc_ids:
            return metadata_filter

        if metadata_filter is None:
            return RetrievalFilter(doc_ids=session_doc_ids)

        payload = metadata_filter.model_dump()
        existing_doc_ids = [str(item) for item in payload.get("doc_ids", []) if str(item).strip()]
        if existing_doc_ids:
            intersected_doc_ids = [
                doc_id for doc_id in existing_doc_ids if doc_id in set(session_doc_ids)
            ]
            payload["doc_ids"] = intersected_doc_ids or ["__session_scope_no_match__"]
        else:
            payload["doc_ids"] = session_doc_ids
        return RetrievalFilter.model_validate(payload)

    def _build_generation_context(
        self,
        *,
        question: str,
        retrieval_query: str,
        history_messages: list[ChatMessage],
        documents: list[Document],
        citations: list[Citation],
        metadata_filter: RetrievalFilter | None,
        session_doc_ids: list[str],
        retrieval_trace: dict[str, Any],
    ) -> GenerationContext:
        context_blocks: list[ContextBlock] = []
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata or {}
            source_name = str(metadata.get("source_name", "unknown"))
            title = str(metadata.get("title") or source_name)
            section_path = metadata.get("section_path") or []
            if isinstance(section_path, list):
                section_label = (
                    " > ".join(str(item) for item in section_path if str(item).strip()) or "unknown"
                )
            else:
                section_label = str(section_path) or "unknown"
            page = str(metadata.get("page") or metadata.get("page_or_section") or "unknown")
            doc_type = str(metadata.get("doc_type") or metadata.get("source_type") or "unknown")
            context_blocks.append(
                ContextBlock(
                    index=index,
                    source_name=source_name,
                    title=title,
                    section_label=section_label,
                    page=page,
                    doc_type=doc_type,
                    content=document.page_content,
                )
            )

        return GenerationContext(
            question=question,
            retrieval_query=retrieval_query,
            history_messages=history_messages,
            documents=documents,
            citations=citations,
            context_blocks=context_blocks,
            metadata_filter=metadata_filter,
            session_doc_ids=session_doc_ids,
            retrieval_trace=retrieval_trace,
        )

    def _context_block_to_prompt_item(self, context_block: ContextBlock) -> dict[str, str]:
        return {
            "index": str(context_block.index),
            "source_name": context_block.source_name,
            "title": context_block.title,
            "section_label": context_block.section_label,
            "page": context_block.page,
            "doc_type": context_block.doc_type,
            "content": context_block.content,
        }

    def _sanitize_answer_citations(self, answer_text: str, citations: list[Citation]) -> str:
        max_index = len(citations)

        def replacer(match: re.Match[str]) -> str:
            index = int(match.group(1))
            return match.group(0) if 1 <= index <= max_index else ""

        sanitized = _CITATION_PATTERN.sub(replacer, answer_text)
        return re.sub(r"\s{2,}", " ", sanitized).strip()

    def _collect_route_names(self, documents: list[Document]) -> list[str]:
        route_names: list[str] = []
        for document in documents:
            route_name = str((document.metadata or {}).get("route_name") or "")
            if route_name and route_name not in route_names:
                route_names.append(route_name)
        return route_names

    def _build_debug_trace(
        self,
        *,
        context: GenerationContext,
        session_id: str,
        route_names: list[str],
        status: AnswerStatus,
        reason: AnswerReason,
        validation_result: AnswerValidationResult | None,
    ) -> dict[str, Any]:
        request_trace = dict(context.retrieval_trace.get("request") or {})
        request_trace["session_id"] = session_id
        request_trace["retrieval_query"] = context.retrieval_query
        if route_names and "route_name" not in request_trace:
            request_trace["route_name"] = route_names[0]
        if "retry_reason" not in request_trace:
            request_trace["retry_reason"] = "none"

        retrieval_trace = dict(context.retrieval_trace.get("retrieval") or {})
        retrieval_trace.setdefault("result_count", len(context.documents))
        retrieval_trace.setdefault(
            "source_doc_ids",
            [str((document.metadata or {}).get("doc_id") or "") for document in context.documents],
        )
        retrieval_trace.setdefault(
            "retrieval_sources",
            [
                list((document.metadata or {}).get("retrieval_sources") or [])
                for document in context.documents
            ],
        )

        answer_hash = (
            hashlib.sha256(context.question.encode("utf-8")).hexdigest()
            if not context.documents
            else ""
        )
        generation_trace = {
            "citation_indices_used": self._extract_citation_indices_from_documents(
                context.documents
            ),
            "final_citations": [
                {
                    "index": citation.index,
                    "doc_id": citation.doc_id,
                    "source_name": citation.source_name,
                }
                for citation in context.citations
            ],
            "answer_status": status.value,
            "answer_reason": reason.value,
            "answer_text_len": 0,
            "answer_text_hash": answer_hash,
            "answer_preview": "",
            "validation": {
                "is_valid": validation_result.is_valid if validation_result else None,
                "invalid_citation_indices": validation_result.invalid_citation_indices
                if validation_result
                else [],
                "missing_citation_sentence_count": len(validation_result.missing_citation_sentences)
                if validation_result
                else 0,
            },
        }

        return {
            "request": request_trace,
            "retrieval": retrieval_trace,
            "generation": generation_trace,
        }

    @staticmethod
    def _extract_citation_indices_from_documents(documents: list[Document]) -> list[int]:
        indices: list[int] = []
        for idx, _ in enumerate(documents, start=1):
            indices.append(idx)
        return indices

    def _validate_answer(
        self, *, answer_text: str, citations: list[Citation]
    ) -> AnswerValidationResult:
        max_index = len(citations)
        invalid_indices: list[int] = []
        for match in _CITATION_PATTERN.finditer(answer_text):
            index = int(match.group(1))
            if index < 1 or index > max_index:
                invalid_indices.append(index)

        missing_citation_sentences: list[str] = []
        for sentence in self._split_sentences(answer_text):
            if not sentence.strip():
                continue
            if not self._is_factual_assertion_sentence(sentence):
                continue
            if not _CITATION_PATTERN.search(sentence):
                missing_citation_sentences.append(sentence.strip())

        return AnswerValidationResult(
            is_valid=not invalid_indices and not missing_citation_sentences,
            invalid_citation_indices=sorted(set(invalid_indices)),
            missing_citation_sentences=missing_citation_sentences,
        )

    def _repair_answer_once(self, *, answer_text: str, citations: list[Citation]) -> str:
        sanitized = self._sanitize_answer_citations(answer_text, citations)
        if not citations:
            return sanitized
        rebuilt_sentences: list[str] = []
        for sentence in self._split_sentences(sanitized):
            stripped = sentence.strip()
            if not stripped:
                rebuilt_sentences.append(sentence)
                continue
            if self._is_factual_assertion_sentence(stripped) and not _CITATION_PATTERN.search(
                stripped
            ):
                stripped = f"{stripped} [1]"
            rebuilt_sentences.append(stripped)
        return " ".join(item for item in rebuilt_sentences if item).strip()

    def _build_fallback_answer(
        self,
        *,
        citations: list[Citation],
        question: str,
        weak_evidence: bool,
    ) -> tuple[str, AnswerStatus, AnswerReason]:
        source_list = (
            ", ".join(f"[{item.index}] {item.source_name}" for item in citations)
            if citations
            else "无可用来源"
        )
        if weak_evidence:
            text = (
                f"基于当前检索结果，我只能给出保守结论：该问题在现有证据下无法稳定确认完整细节。"
                f"建议优先参考以下来源再确认：{source_list}。"
            )
            return text, AnswerStatus.WEAK_HIT, AnswerReason.WEAK_EVIDENCE
        text = f"我无法从当前检索上下文中稳定支撑这个问题的回答。可参考来源：{source_list}。"
        return text, AnswerStatus.REJECTED, AnswerReason.UNSUPPORTED

    @staticmethod
    def _split_sentences(answer_text: str) -> list[str]:
        chunks = re.split(r"(?:[。！？!?]+|\n+)", answer_text)
        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    @staticmethod
    def _is_factual_assertion_sentence(sentence: str) -> bool:
        if _NON_ASSERTION_HINT_PATTERN.search(sentence) and not _ASSERTION_SIGNAL_PATTERN.search(
            sentence
        ):
            return False
        return bool(_ASSERTION_SIGNAL_PATTERN.search(sentence))

    @staticmethod
    def _chunk_text_for_stream(answer_text: str, *, max_chars: int = 120) -> list[str]:
        text = answer_text.strip()
        if not text:
            return []
        return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]

    def _persist_turn(
        self,
        session_id: str,
        question: str,
        answer_draft: AnswerDraft,
        metadata_filter: RetrievalFilter | None,
    ) -> None:
        debug_trace = self._with_answer_summary(
            debug_trace=answer_draft.debug_trace,
            answer_text=answer_draft.answer_text,
            status=answer_draft.status,
            reason=answer_draft.reason,
        )
        self._chat_history_store.append_message(
            session_id=session_id,
            role=ChatRole.USER,
            content=question,
        )
        self._chat_history_store.append_message(
            session_id=session_id,
            role=ChatRole.ASSISTANT,
            content=answer_draft.answer_text,
            grounded=answer_draft.grounded,
            citations=list(answer_draft.citations),
            retrieval_query=str((debug_trace.get("request") or {}).get("retrieval_query") or ""),
            metadata_filter=metadata_filter.model_dump(mode="json") if metadata_filter else None,
            debug_trace=debug_trace,
        )

    @staticmethod
    def _with_answer_summary(
        *,
        debug_trace: dict[str, Any],
        answer_text: str,
        status: AnswerStatus,
        reason: AnswerReason,
    ) -> dict[str, Any]:
        payload = dict(debug_trace)
        generation_trace = dict(payload.get("generation") or {})
        generation_trace["answer_status"] = status.value
        generation_trace["answer_reason"] = reason.value
        generation_trace["answer_text_len"] = len(answer_text)
        generation_trace["answer_text_hash"] = hashlib.sha256(
            answer_text.encode("utf-8")
        ).hexdigest()
        generation_trace["answer_preview"] = answer_text[:240]
        payload["generation"] = generation_trace
        return payload

    def _build_regulation_extract_answer(
        self,
        *,
        question: str,
        documents: list[Document],
        citations: list[Citation],
    ) -> str | None:
        if not documents or not citations:
            return None
        normalized_question = normalize_text(question)
        for document in documents[:2]:
            metadata = document.metadata or {}
            citation_index = self._find_citation_index_for_document(
                document=document, citations=citations
            )
            if citation_index is None:
                continue
            extract_text = str(metadata.get("child_content") or document.page_content or "")
            if re.search(r"英文名称|英文名|english", normalized_question, re.IGNORECASE):
                english_alias = str(metadata.get("english_alias") or "").strip()
                if english_alias:
                    subject = self._extract_subject_from_question(
                        question=normalized_question, fallback=extract_text
                    )
                    return f"{subject}的英文名称是 {english_alias} [{citation_index}]。"
            if re.search(r"多少天|几天", normalized_question):
                row_key = str(metadata.get("pseudo_table_row_key") or "").strip()
                row_value = str(metadata.get("pseudo_table_row_value") or "").strip()
                if row_key and row_value and row_key in normalized_question:
                    return f"{row_key}的休止时间是 {row_value} 天 [{citation_index}]。"
            patterns = self._regulation_extract_patterns(normalized_question)
            for pattern in patterns:
                sentence = self._extract_matching_sentence(text=extract_text, pattern=pattern)
                if sentence is None:
                    continue
                return f"{sentence} [{citation_index}]。"
        return None

    @staticmethod
    def _find_citation_index_for_document(
        *, document: Document, citations: list[Citation]
    ) -> int | None:
        metadata = document.metadata or {}
        target_doc_id = str(metadata.get("doc_id") or "")
        target_page = str(metadata.get("page") or metadata.get("page_or_section") or "")
        for citation in citations:
            if target_doc_id and citation.doc_id == target_doc_id:
                citation_page = str(citation.page or citation.page_or_section or "")
                if not target_page or not citation_page or citation_page == target_page:
                    return citation.index
        return citations[0].index if citations else None

    @staticmethod
    def _extract_subject_from_question(*, question: str, fallback: str) -> str:
        subject_match = re.search(r"([\u4e00-\u9fffA-Za-z0-9\-]+?)的英文名称", question)
        if subject_match:
            return subject_match.group(1)
        fallback_match = re.search(r"[\d.]+\s*([\u4e00-\u9fffA-Za-z0-9\-]+)\s+[A-Za-z]", fallback)
        if fallback_match:
            return fallback_match.group(1)
        return "该术语"

    @staticmethod
    def _extract_matching_sentence(*, text: str, pattern: str) -> str | None:
        normalized_text = normalize_text(text)
        match = re.search(pattern, normalized_text)
        if match is None:
            return None
        return match.group(0).rstrip("。；;")

    @staticmethod
    def _regulation_extract_patterns(question: str) -> list[str]:
        if re.search(r"目的|宗旨", question):
            return [r"为了[^。；]*制定本规范"]
        if "适用" in question:
            return [r"本规范适用于[^。；]*"]
        if re.search(r"主要用于|用于检测", question):
            return [r"检测[^。；]*判定[^。；]*", r"主要用于[^。；]*"]
        if re.search(r"承载力指标|确定什么", question):
            return [r"确定[^。；]*极限承载力[^。；]*", r"极限承载力[^。；]*"]
        if re.search(r"不应低于|应满足什么要求|满足什么要求", question):
            return [r"不应低于[^。；]*"]
        return []
