from __future__ import annotations

from collections.abc import AsyncIterator

import pytest


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def __call__(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        length = float(len(text))
        vowels = float(sum(1 for char in text.lower() if char in "aeiou"))
        spaces = float(text.count(" "))
        return [length, vowels, spaces]


class FakeChatStream:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[object]:
        async def generator() -> AsyncIterator[object]:
            for chunk in self._chunks:
                delta = type("Delta", (), {"content": chunk})()
                choice = type("Choice", (), {"delta": delta})()
                yield type("Chunk", (), {"choices": [choice]})()

        return generator()


class FakeChatCompletions:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> FakeChatStream:
        self.calls.append(kwargs)
        return FakeChatStream(self._chunks)


class FakeChatClient:
    def __init__(self, chunks: list[str] | None = None) -> None:
        self.completions = FakeChatCompletions(chunks or ["Answer"])
        self.chat = type("ChatNamespace", (), {"completions": self.completions})()


@pytest.fixture()
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()
