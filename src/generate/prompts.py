from __future__ import annotations

from langchain_core.documents import Document


SYSTEM_PROMPT = """你是一个工程文档 RAG 问答助手。你只能依据给定的检索上下文回答问题。

规则：
1. 不要编造上下文中没有的事实。
2. 如果证据不足，请明确回答“我无法从提供的资料中确认”。
3. 答案应简洁、专业。
4. 如使用上下文，请在相关结论后标注方括号引用，例如 [1]、[2]。
5. 不要引用未提供的来源编号。
"""


def build_user_prompt(question: str, documents: list[Document]) -> str:
    context_blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        source_name = metadata.get("source_name", "unknown")
        section = metadata.get("page_or_section") or "unknown"
        context_blocks.append(
            f"[{index}] source={source_name}; section={section}\n{document.page_content}"
        )

    context = "\n\n".join(context_blocks)
    return f"问题：{question}\n\n检索上下文：\n{context}\n\n请基于以上上下文回答。"

