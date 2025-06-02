from abc import ABC, abstractmethod
from typing import Any
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)


class BaseVectorStoreManager(ABC):
    """Base class for vector store managers that defines the common interface."""

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs that were added
        """
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks from the vector store.

        Args:
            doc_id: ID of the document to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        pass

    @abstractmethod
    def list_documents(self) -> list[dict[str, Any]]:
        """
        List all documents in the vector store with their metadata.

        Returns:
            List of dictionaries containing document metadata
        """
        pass

    @abstractmethod
    def get_document_by_id(self, doc_id: str) -> list[Document]:
        """
        Retrieve all chunks of a specific document.

        Args:
            doc_id: ID of the document to retrieve

        Returns:
            List of Document objects containing the document chunks
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: dict[str, Any] | None = None,
        with_score: bool = False,
    ) -> list[tuple[Document, float]]:
        """
        Search for similar documents using vector similarity.

        Args:
            query: Search query string
            k: Number of results to return
            filter_dict: Optional metadata filters
            with_score: Whether to return the score of the search
        Returns:
            List of matching Document objects optionally with their scores
        """
        pass

    @abstractmethod
    def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the vector store collection.

        Returns:
            Dictionary containing collection statistics
        """
        pass

    @abstractmethod
    def reset_collection(self) -> bool:
        """
        Reset the entire collection (delete all documents).

        Returns:
            True if reset was successful, False otherwise
        """
        pass


class SearchFilters:
    """Helper class for building search filters."""

    @staticmethod
    def by_content_type(content_type: str) -> dict[str, str]:
        """Filter by content type (e.g., 'blog_post', 'paper')."""
        return {"content_type": content_type}

    @staticmethod
    def by_author(author: str) -> dict[str, str]:
        """Filter by author name."""
        return {"authors": {"$contains": author}}

    @staticmethod
    def by_tag(tag: str) -> dict[str, str]:
        """Filter by tag."""
        return {"tags": {"$contains": tag}}

    @staticmethod
    def by_date_range(start_date: str, end_date: str) -> dict:
        """Filter by date range (ISO format dates)."""
        return {"publish_date": {"$gte": start_date, "$lte": end_date}}

    @staticmethod
    def combine_filters(*filters: dict[str, Any]) -> dict[str, Any]:
        """Combine multiple filters with AND logic."""
        if not filters:
            return {}

        if len(filters) == 1:
            return filters[0]
        return {"$and": list(filters)}
