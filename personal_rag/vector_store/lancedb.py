import lancedb
import pyarrow as pa
import pandas as pd
from pathlib import Path
import streamlit as st
from typing import List, Dict, Any, Optional
import logging
from personal_rag.embeddings.gemini import GeminiEmbeddingClient
from langchain.schema import Document
from personal_rag.vector_store.vector_store_manager import BaseVectorStoreManager

logger = logging.getLogger(__name__)


class LanceDBVectorStoreManager(BaseVectorStoreManager):
    def __init__(
        self,
        db_path: str,
        embedding_client: GeminiEmbeddingClient,
    ):
        self.db_path = db_path
        self.embedding_client = embedding_client

        # Connect to LanceDB
        self.db = lancedb.connect(db_path)
        self.table_name = "documents"

        # Initialize or connect to table
        self._initialize_table()

    def _initialize_table(self):
        """Initialize documents table if it doesn't exist"""
        try:
            self.table = self.db.open_table(self.table_name)
            logger.info(f"Connected to existing table with {len(self.table)} documents")
        except Exception:
            # Create new table with schema
            logger.info("Creating new documents table")
            self._create_table()

    def _create_table(self):
        """Create new documents table with proper schema"""
        # Create sample data to define schema
        sample_data = {
            "id": ["sample"],
            "text": ["Sample document text"],
            "vector": [self.embedding_model.encode("sample").tolist()],
            "doc_id": ["sample_doc"],
            "title": ["Sample Title"],
            "source": ["https://example.com"],
            "authors": ["Sample Author"],
            "publish_date": ["2024-01-01"],
            "tags": ["sample"],
            "chunk_index": [0],
            "total_chunks": [1],
            "content_type": ["sample"],
            "metadata": ["{}"],
        }

        df = pd.DataFrame(sample_data)
        self.table = self.db.create_table(self.table_name, df)

        # Delete sample data
        self.table.delete("id = 'sample'")

    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to LanceDB"""
        if not documents:
            return []

        try:
            # Process documents and generate embeddings
            processed_docs = []

            for doc in documents:
                # Generate embedding
                text = doc.get("text", "")
                embedding = self.embedding_model.encode(text).tolist()

                processed_doc = {
                    "id": doc.get("id", ""),
                    "text": text,
                    "vector": embedding,
                    "doc_id": doc.get("doc_id", ""),
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "authors": doc.get("authors", ""),
                    "publish_date": doc.get("publish_date", ""),
                    "tags": doc.get("tags", ""),
                    "chunk_index": doc.get("chunk_index", 0),
                    "total_chunks": doc.get("total_chunks", 1),
                    "content_type": doc.get("content_type", ""),
                    "metadata": doc.get("metadata", "{}"),
                }
                processed_docs.append(processed_doc)

            # Add to table
            df = pd.DataFrame(processed_docs)
            self.table.add(df)

            logger.info(f"Added {len(processed_docs)} documents to LanceDB")
            return [doc["id"] for doc in processed_docs]

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return []

    def similarity_search(
        self, query: str, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search documents using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()

            # Perform search
            search_results = self.table.search(query_embedding).limit(k)

            # Apply filters if provided
            if filter_dict:
                filter_conditions = []
                for key, value in filter_dict.items():
                    if isinstance(value, list):
                        filter_conditions.append(f"{key} IN {value}")
                    else:
                        filter_conditions.append(f"{key} = '{value}'")
                filters = " AND ".join(filter_conditions)
                search_results = search_results.where(filters)

            # Execute and convert to Documents
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
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def get_document_by_id(self, doc_id: str) -> List[Document]:
        """Get all chunks for a specific document"""
        try:
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

        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return []

    def reset_collection(self) -> bool:
        """Reset the entire collection (delete all documents)."""
        try:
            # Drop and recreate the table
            self.db.drop_table(self.table_name)
            self._create_table()
            logger.info(f"Reset collection: {self.table_name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        try:
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

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
