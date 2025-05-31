from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)


class BaseVectorStoreManager(ABC):
    """Base class for vector store managers that defines the common interface."""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
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
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the vector store with their metadata.

        Returns:
            List of dictionaries containing document metadata
        """
        pass

    @abstractmethod
    def get_document_by_id(self, doc_id: str) -> List[Document]:
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
        self, query: str, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents using vector similarity.

        Args:
            query: Search query string
            k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of matching Document objects
        """
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
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
