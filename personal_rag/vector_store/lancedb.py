import lancedb
import pandas as pd
from typing import Any
import logging
from personal_rag.embeddings.gemini import GeminiEmbeddingClient
from langchain.schema import Document
from personal_rag.vector_store.vector_store_manager import BaseVectorStoreManager
from personal_rag.vector_store.models import Article

logger = logging.getLogger(__name__)


class LanceDBVectorStoreManager(BaseVectorStoreManager):
    def __init__(
        self,
        db_path: str,
        service_account_path: str,
        embedding_client: GeminiEmbeddingClient,
    ):
        self.db_path = db_path
        self.embedding_client = embedding_client

        # Connect to LanceDB
        logger.info(f"Connecting to LanceDB at {db_path}")
        self.db = lancedb.connect(
            db_path,
            storage_options={
                "service_account": service_account_path,
            },
        )
        self.table_name = "documents"

        # Initialize or connect to table
        self._initialize_table()

    def _initialize_table(self):
        """Initialize documents table if it doesn't exist"""
        logger.info(f"Initializing table {self.table_name}")
        self.table = self.db.create_table(
            self.table_name, schema=Article, exist_ok=True
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to LanceDB"""
        if not documents:
            return []

        # Process documents and generate embeddings
        processed_docs: list[Article] = []

        for doc in documents:
            # Generate embedding
            text = doc.page_content
            embedding = self.embedding_client.embed_content(text)[0].values
            processed_doc = Article(
                id=doc.metadata.get("id", ""),
                text=text,
                vector=embedding,
                doc_id=doc.metadata.get("doc_id", ""),
                title=doc.metadata.get("title", ""),
                source=doc.metadata.get("source", ""),
                authors=doc.metadata.get("authors", "").split(","),
                publish_date=doc.metadata.get("publish_date"),
                tags=doc.metadata.get("tags", "").split(","),
                chunk_index=doc.metadata.get("chunk_index"),
                total_chunks=doc.metadata.get("total_chunks"),
                content_type=doc.metadata.get("content_type"),
                metadata=str(doc.metadata),
            )
            processed_docs.append(processed_doc)

        # Add to table
        self.table.add(processed_docs)

        logger.info(f"Added {len(processed_docs)} documents to LanceDB")
        return [doc.id for doc in processed_docs]

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: dict[str, Any] | None = None,
        with_score: bool = False,
    ) -> list[tuple[Document, float]]:
        """Search documents using vector similarity"""
        query_embedding = self.embedding_client.embed_content(query)[0].values

        search_results = self.table.search(query_embedding).limit(k)

        if filter_dict:
            filter_conditions = []
            for key, value in filter_dict.items():
                if isinstance(value, list):
                    filter_conditions.append(f"{key} IN {value}")
                else:
                    filter_conditions.append(f"{key} = '{value}'")
            filters = " AND ".join(filter_conditions)
            search_results = search_results.where(filters)

        results = search_results.to_pandas()
        documents = []

        for _, row in results.iterrows():
            doc = Document(
                page_content=row["text"],
                metadata={
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "source": row["source"],
                    "authors": row["authors"],
                    "publish_date": row["publish_date"],
                    "tags": row["tags"],
                    "chunk_index": row["chunk_index"],
                    "total_chunks": row["total_chunks"],
                    "content_type": row["content_type"],
                },
            )
            documents.append((doc, row["_distance"] if with_score else -1.0))

        return documents

    def get_document_by_id(self, doc_id: str) -> list[Document]:
        """Get all chunks for a specific document"""
        results = self.table.search().where(f"doc_id = '{doc_id}'").to_pandas()

        if results.empty:
            logger.warning(f"No chunks found for document {doc_id}")
            return []

        # Sort by chunk_index
        results = results.sort_values("chunk_index")

        # Convert to Documents
        documents = []
        for _, row in results.iterrows():
            doc = Document(
                page_content=row["text"],
                metadata={
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "source": row["source"],
                    "authors": row["authors"],
                    "publish_date": row["publish_date"],
                    "tags": row["tags"],
                    "chunk_index": row["chunk_index"],
                    "total_chunks": row["total_chunks"],
                    "content_type": row["content_type"],
                },
            )
            documents.append(doc)

        return documents

    def reset_collection(self) -> bool:
        """Reset the entire collection (delete all documents)."""
        self.db.drop_table(self.table_name)
        self._create_table()
        logger.info(f"Reset collection: {self.table_name}")
        return True

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store collection."""
        df = self.table.to_pandas()

        if df.empty:
            return {
                "total_chunks": 0,
                "unique_documents": 0,
                "total_characters": 0,
                "average_chunk_size": 0,
                "content_types": {},
                "collection_name": self.table_name,
                "db_path": self.db_path,
            }

        return {
            "total_chunks": len(df),
            "unique_documents": df["doc_id"].nunique(),
            "total_characters": df["text"].str.len().sum(),
            "average_chunk_size": round(df["text"].str.len().mean(), 2),
            "content_types": df["content_type"].value_counts().to_dict(),
            "collection_name": self.table_name,
            "db_path": self.db_path,
        }

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks from the vector store.

        Args:
            doc_id: ID of the document to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        # Delete all chunks with matching doc_id
        self.table.delete(f"doc_id = '{doc_id}'")
        logger.info(f"Deleted document {doc_id} and all its chunks")
        return True

    def list_documents(self) -> list[dict[str, Any]]:
        """
        List all documents in the vector store with their metadata.

        Returns:
            List of dictionaries containing document metadata
        """
        # Get all unique documents
        df = self.table.to_pandas()
        if df.empty:
            return []

        # Group by doc_id and get first occurrence of metadata
        documents = []
        for doc_id in df["doc_id"].unique():
            doc_chunks = df[df["doc_id"] == doc_id]
            first_chunk = doc_chunks.iloc[0]

            doc_info = {
                "doc_id": doc_id,
                "title": first_chunk["title"],
                "source": first_chunk["source"],
                "authors": first_chunk["authors"],
                "publish_date": first_chunk["publish_date"],
                "tags": first_chunk["tags"],
                "content_type": first_chunk["content_type"],
                "chunk_count": len(doc_chunks),
                "total_chunks": first_chunk["total_chunks"],
            }
            documents.append(doc_info)

        return documents
