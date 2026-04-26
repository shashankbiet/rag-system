from typing import Protocol

from langchain_core.documents import Document


class DocumentLoader(Protocol):
    def load(self, path: str) -> list[Document]:
        """Load documents from a file path"""
        ...
