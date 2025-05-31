from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from personal_rag.embeddings.embeddings_client import BaseEmbeddingsClient


class HuggingFaceEmbeddingsClient(BaseEmbeddingsClient):
    """Client for generating embeddings using HuggingFace models."""

    def __init__(self, model_name: str):
        """Initialize the HuggingFace embeddings client.

        Args:
            model_name: Name of the HuggingFace model to use
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # Force CPU usage
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_content(self, contents: str) -> List[float]:
        """Embed a single piece of content.

        Args:
            contents: The text content to embed

        Returns:
            List of floats representing the embedding vector
        """
        return self.embeddings.embed_query(contents)

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """Embed multiple documents.

        Args:
            documents: List of LangChain documents to embed

        Returns:
            List of embedding vectors, one for each document
        """
        texts = [doc.page_content for doc in documents]
        return self.embeddings.embed_documents(texts)
