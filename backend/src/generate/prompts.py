from __future__ import annotations

from langchain_core.documents import Document


SYSTEM_PROMPT = """你是一个工程文档 RAG 问答助手，只能依据给定检索上下文回答问题。
规则：
1. 不要编造上下文中不存在的事实。
2. 如果证据不足，请明确回答“我无法从提供的资料中确认”。
3. 答案优先简洁、专业，先给结论，再给依据。
4. 如果使用了上下文，请在相关结论后标注方括号引用，例如 [1]、[2]。
5. 不要引用未提供的来源编号。
"""


def build_user_prompt(question: str, documents: list[Document]) -> str:
    context_blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        source_name = metadata.get("source_name", "unknown")
        title = metadata.get("title") or source_name
        section_path = metadata.get("section_path") or []
        if isinstance(section_path, list):
            section_label = " > ".join(str(item) for item in section_path if str(item).strip()) or "unknown"
        else:
            section_label = str(section_path) or "unknown"
        page = metadata.get("page") or metadata.get("page_or_section") or "unknown"
        doc_type = metadata.get("doc_type") or metadata.get("source_type") or "unknown"
        context_blocks.append(
            f"[{index}] source={source_name}; title={title}; section={section_label}; page={page}; doc_type={doc_type}\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_blocks)
    return f"问题：{question}\n\n检索上下文：\n{context}\n\n请基于以上上下文回答。"
