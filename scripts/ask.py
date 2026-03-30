from __future__ import annotations

import argparse
import asyncio

from src.core.dependencies import get_container


async def run(question: str) -> None:
    container = get_container()
    async for event in container.answer_service.stream_answer(question):
        if event["event"] == "token":
            print(event["data"]["text"], end="", flush=True)
        elif event["event"] == "sources":
            print("\n\nSources:")
            for citation in event["data"]["citations"]:
                print(f"- [{citation['index']}] {citation['source_name']}: {citation['snippet']}")
        elif event["event"] == "done":
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the default knowledge base.")
    parser.add_argument("question", help="Question to ask.")
    args = parser.parse_args()
    asyncio.run(run(args.question))


if __name__ == "__main__":
    main()
