import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import streamlit as st

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(
        self,
        persist_directory: str,
        embedding_model: str,
        collection_name: str,
    ):
        with st.spinner("Initializing vector store..."):
            self.persist_directory = Path(persist_directory)
            self.collection_name = collection_name

            # Create directory if it doesn't exist
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # Initialize embeddings (CPU-only)
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cpu"},  # Force CPU usage
                encode_kwargs={"normalize_embeddings": True},
            )

            # Initialize Chroma client
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Initialize vector store
            self.vector_store = None
            self._initialize_vector_store()

    def _initialize_vector_store(self):
        try:
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory),
            )
            logger.info(
                f"Vector store initialized with collection: {self.collection_name}"
            )

            # Log existing document count
            try:
                existing_count = len(self.vector_store.get()["ids"])
                logger.info(
                    f"Found {existing_count} existing documents in vector store"
                )
            except Exception as e:
                logger.info(f"Initialized empty vector store because of {e}")

        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise e

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a specific document ID."""
        try:
            # Get all chunk IDs for this document
            results = self.vector_store.get(where={"doc_id": doc_id})

            if not results["ids"]:
                logger.warning(f"No chunks found for document {doc_id}")
                return False

            # Delete all chunks
            self.vector_store.delete(ids=results["ids"])

            logger.info(f"Deleted {len(results['ids'])} chunks for document {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False

    def list_documents(self) -> list[dict[str, Any]]:
        """List all documents in the vector store with metadata."""
        try:
            results = self.vector_store.get()

            # Group by document ID
            documents = {}
            for i, chunk_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                doc_id = metadata.get("doc_id")

                if doc_id not in documents:
                    documents[doc_id] = {
                        "doc_id": doc_id,
                        "title": metadata.get("title", "Unknown"),
                        "source": metadata.get("source", ""),
                        "authors": metadata.get("authors", []),
                        "tags": metadata.get("tags", []),
                        "publish_date": metadata.get("publish_date"),
                        "scraped_at": metadata.get("scraped_at"),
                        "total_chunks": metadata.get("total_chunks", 0),
                        "chunk_count": 0,
                    }

                documents[doc_id]["chunk_count"] += 1

            return list(documents.values())

        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store collection."""
        try:
            results = self.vector_store.get()

            total_chunks = len(results["ids"])
            unique_docs = len(
                set(
                    metadata.get("doc_id")
                    for metadata in results["metadatas"]
                    if metadata
                )
            )

            # Calculate content statistics
            total_chars = sum(len(doc) for doc in results["documents"])
            avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0

            # Get content types
            content_types = {}
            for metadata in results["metadatas"]:
                if metadata:
                    content_type = metadata.get("content_type", "unknown")
                    content_types[content_type] = content_types.get(content_type, 0) + 1

            return {
                "total_chunks": total_chunks,
                "unique_documents": unique_docs,
                "total_characters": total_chars,
                "average_chunk_size": round(avg_chunk_size, 2),
                "content_types": content_types,
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory),
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}

    def reset_collection(self) -> bool:
        """Reset the entire collection (delete all documents)."""
        try:
            # Delete the collection
            self.chroma_client.delete_collection(self.collection_name)

            # Reinitialize
            self._initialize_vector_store()

            logger.info(f"Reset collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            return False

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of document IDs that were added
        """
        if not documents:
            logger.warning("No documents provided for addition")
            return []

        try:
            # Filter out any documents that already exist
            new_documents = self._filter_existing_documents(documents)

            if not new_documents:
                logger.info("All documents already exist in vector store")
                return []

            # Add documents to vector store
            ids = self.vector_store.add_documents(new_documents)

            logger.info(f"Added {len(new_documents)} new documents to vector store")
            return ids

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return []

    def _filter_existing_documents(self, documents: list[Document]) -> list[Document]:
        """Filter out documents that already exist in the vector store."""
        try:
            # Get all existing chunk IDs
            existing_data = self.vector_store.get()
            existing_chunk_ids = set()

            for metadata in existing_data.get("metadatas", []):
                if metadata and "chunk_id" in metadata:
                    existing_chunk_ids.add(metadata["chunk_id"])

            # Filter new documents
            new_documents = []
            for doc in documents:
                chunk_id = doc.metadata.get("chunk_id")
                if chunk_id not in existing_chunk_ids:
                    new_documents.append(doc)

            logger.info(
                f"Filtered {len(documents) - len(new_documents)} existing documents"
            )
            return new_documents

        except Exception as e:
            logger.warning(f"Error filtering existing documents: {str(e)}")
            return documents  # Return all if filtering fails

    def similarity_search(
        self, query: str, k: int = 5, filter_dict: dict[str, Any] | None = None
    ) -> list[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query string
            k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of matching Document objects
        """
        try:
            if filter_dict:
                results = self.vector_store.similarity_search(
                    query=query, k=k, filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(query=query, k=k)

            logger.debug(f"Found {len(results)} results for query: {query[:100]}...")
            return results

        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []

    def get_document_by_id(self, doc_id: str) -> list[Document]:
        """Get all chunks for a specific document ID."""
        try:
            results = self.vector_store.get(where={"doc_id": doc_id})

            documents = []
            if results["ids"]:
                for i, chunk_id in enumerate(results["ids"]):
                    doc = Document(
                        page_content=results["documents"][i],
                        metadata=results["metadatas"][i],
                    )
                    documents.append(doc)

            # Sort by chunk index
            documents.sort(key=lambda x: x.metadata.get("chunk_index", 0))

            logger.debug(f"Retrieved {len(documents)} chunks for document {doc_id}")
            return documents

        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            raise e


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
