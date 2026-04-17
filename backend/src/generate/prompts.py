from __future__ import annotations

from backend.src.core.models import ChatMessage


def build_system_prompt() -> str:
    return """你是一个工程文档 RAG 问答助手，只能依据给定检索上下文回答问题。
规则：
1. 不要编造上下文中不存在的事实。
2. 如果证据不足，请明确回答“我无法从提供的资料中确认”。
3. 答案优先简洁、专业，先给结论，再给依据。
4. 事实性断言句必须带引用，例如涉及数值、比较、归因、来源描述的结论。
5. 总结句、过渡句、限定条件句可以不带引用，但不能引入新事实。
6. 不要引用未提供的来源编号。
7. 如果问题是多轮追问，优先结合最近对话理解指代，但结论仍必须来自检索上下文。
"""


def build_context_blocks(documents: list[dict[str, str]]) -> list[str]:
    context_blocks: list[str] = []
    for item in documents:
        context_blocks.append(
            f"[{item['index']}] source={item['source_name']}; title={item['title']}; section={item['section_label']}; "
            f"page={item['page']}; doc_type={item['doc_type']}\n{item['content']}"
        )
    return context_blocks


def build_answer_prompt(
    question: str, history: list[ChatMessage], context_blocks: list[str]
) -> str:
    history_lines: list[str] = []
    for message in history[-4:]:
        history_lines.append(f"{message.role.value}: {message.content}")

    history_block = "\n".join(history_lines) if history_lines else "无"
    context = "\n\n".join(context_blocks)
    return (
        f"当前问题：{question}\n\n"
        f"最近对话：\n{history_block}\n\n"
        f"检索上下文：\n{context}\n\n"
        "请严格依据检索上下文回答。先给结论，再给简短依据。"
        "事实性断言请使用引用编号，引用只能使用上下文中已有编号。"
    )
