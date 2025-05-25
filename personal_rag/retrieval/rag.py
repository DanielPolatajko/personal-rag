from langchain_anthropic import ChatAnthropic
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from typing import Any
import logging
from datetime import datetime
import streamlit as st

from personal_rag.retrieval.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str,
        temperature: float,
        max_tokens: int,
        system_prompt_path: str,
        query_prompt_path: str,
    ):
        with st.spinner("Initializing RAG pipeline..."):
            self.vector_store_manager = vector_store_manager

            self.llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            self.system_prompt = open(system_prompt_path).read()

            self.query_prompt = PromptTemplate(
                input_variables=["question", "context", "metadata"],
                template=open(query_prompt_path).read(),
            )

    def query(
        self,
        question: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """
        Query the RAG system with a question.

        Args:
            question: User's question
            k: Number of documents to retrieve
            filters: Optional search filters
            include_metadata: Whether to include document metadata in response

        Returns:
            Dictionary containing answer, sources, and metadata
        """
        try:
            # Retrieve relevant documents
            results = (
                self.vector_store_manager.vector_store.similarity_search_with_score(
                    query=question, k=k, filter_dict=filters
                )
            )

            if not results:
                return {
                    "answer": "I couldn't find any relevant documents to answer your question. Please try rephrasing or check if documents on this topic have been added to the knowledge base.",
                    "sources": [],
                    "metadata": {
                        "query": question,
                        "retrieved_docs": 0,
                        "timestamp": datetime.now().isoformat(),
                    },
                }

            # Prepare context and metadata
            context = self._format_context(results)
            metadata_str = self._format_metadata(results) if include_metadata else ""

            # Generate prompt
            prompt = self.query_prompt.format(
                question=question, context=context, metadata=metadata_str
            )

            # Get response from Claude
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt),
            ]

            response = self.llm(messages)

            # Extract source information
            sources = self._extract_sources(results)

            return {
                "answer": response.content,
                "sources": sources,
                "metadata": {
                    "query": question,
                    "retrieved_docs": len(results),
                    "model": self.llm.model,
                    "timestamp": datetime.now().isoformat(),
                    "filters_applied": filters,
                },
            }

        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "metadata": {
                    "query": question,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
            }

    def _format_context(self, results: list[tuple[Document, float]]) -> str:
        """Format retrieved documents as context for the LLM."""
        context_parts = []

        for i, (doc, score) in enumerate(results, 1):
            title = doc.metadata.get("title", "Unknown Document")
            source = doc.metadata.get("source", "Unknown Source")
            authors = doc.metadata.get("authors", [])

            header = f"Document {i}: {title}"
            if authors:
                header += f" by {', '.join(authors)}"
            header += f" (Relevance: {score:.3f})"

            context_parts.append(
                f"{header}\nSource: {source}\n\nContent:\n{doc.page_content}\n"
            )

        return "\n" + "=" * 80 + "\n".join(context_parts)

    def _format_metadata(self, results: list[tuple[Document, float]]) -> str:
        """Format document metadata for the LLM."""
        metadata_parts = []

        for i, (doc, score) in enumerate(results, 1):
            metadata = doc.metadata
            meta_info = [
                f"Document {i} Metadata:",
                f"- Doc ID: {metadata.get('doc_id', 'Unknown')}",
                f"- Chunk: {metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}",
                f"- Content Type: {metadata.get('content_type', 'Unknown')}",
                f"- Extraction Method: {metadata.get('extraction_method', 'Unknown')}",
            ]

            if metadata.get("publish_date"):
                meta_info.append(f"- Published: {metadata['publish_date']}")

            if metadata.get("tags"):
                meta_info.append(f"- Tags: {', '.join(metadata['tags'])}")

            metadata_parts.append("\n".join(meta_info))

        return "\n\n".join(metadata_parts)

    def _extract_sources(
        self, results: list[tuple[Document, float]]
    ) -> list[dict[str, Any]]:
        """Extract source information from retrieved documents."""
        sources = []
        seen_docs = set()

        for doc, score in results:
            doc_id = doc.metadata.get("doc_id")
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)

                source_info = {
                    "doc_id": doc_id,
                    "title": doc.metadata.get("title", "Unknown Document"),
                    "source": doc.metadata.get("source", ""),
                    "authors": doc.metadata.get("authors", []),
                    "publish_date": doc.metadata.get("publish_date"),
                    "tags": doc.metadata.get("tags", []),
                    "relevance_score": score,
                    "chunk_count": doc.metadata.get("total_chunks", 1),
                }
                sources.append(source_info)

        return sources

    def search_documents(
        self, query: str, k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Search for documents without generating an answer.
        Useful for exploration and document discovery.
        """
        try:
            results = (
                self.vector_store_manager.vector_store.similarity_search_with_score(
                    query=query, k=k, filter_dict=filters
                )
            )

            search_results = []
            for doc, score in results:
                result = {
                    "doc_id": doc.metadata.get("doc_id"),
                    "title": doc.metadata.get("title", "Unknown"),
                    "source": doc.metadata.get("source", ""),
                    "authors": doc.metadata.get("authors", []),
                    "publish_date": doc.metadata.get("publish_date"),
                    "tags": doc.metadata.get("tags", []),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "total_chunks": doc.metadata.get("total_chunks", 1),
                    "relevance_score": score,
                    "content_preview": doc.page_content[:300] + "..."
                    if len(doc.page_content) > 300
                    else doc.page_content,
                }
                search_results.append(result)

            return search_results

        except Exception as e:
            logger.error(f"Error in document search: {str(e)}")
            return []

    def get_document_content(self, doc_id: str) -> dict[str, Any] | None:
        """Get full content of a specific document."""
        try:
            chunks = self.vector_store_manager.get_document_by_id(doc_id)

            if not chunks:
                return None

            # Reconstruct full document
            full_content = "\n\n".join(chunk.page_content for chunk in chunks)

            # Get metadata from first chunk
            metadata = chunks[0].metadata

            return {
                "doc_id": doc_id,
                "title": metadata.get("title", "Unknown"),
                "source": metadata.get("source", ""),
                "authors": metadata.get("authors", []),
                "publish_date": metadata.get("publish_date"),
                "tags": metadata.get("tags", []),
                "content": full_content,
                "chunk_count": len(chunks),
                "total_characters": len(full_content),
                "scraped_at": metadata.get("scraped_at"),
                "processed_at": metadata.get("processed_at"),
            }

        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None
