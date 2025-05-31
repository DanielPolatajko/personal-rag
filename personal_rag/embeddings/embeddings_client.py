from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class BaseEmbeddingsClient(ABC):
    """Abstract base class for embeddings clients."""

    @abstractmethod
    def embed_content(self, contents: str) -> List[float]:
        """Embed a single piece of content.

        Args:
            contents: The text content to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """Embed multiple documents.

        Args:
            documents: List of LangChain documents to embed

        Returns:
            List of embedding vectors, one for each document
        """
        pass
