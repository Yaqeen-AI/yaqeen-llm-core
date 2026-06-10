from __future__ import annotations

from abc import ABC, abstractmethod

from orchestrator.models import RetrievedDocument, SourceRetrievalConfig


class RagAdapter(ABC):
    @abstractmethod
    async def retrieve(self, query: str, config: SourceRetrievalConfig) -> list[RetrievedDocument]:
        raise NotImplementedError

