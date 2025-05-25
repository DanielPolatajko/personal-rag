from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import Any
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process_blog_post(self, blog_data: dict[str, Any]) -> list[Document]:
        """
        Process a blog post into chunks with metadata.

        Args:
            blog_data: Dictionary containing blog post data from scraper

        Returns:
            List of LangChain Document objects ready for vector storage
        """
        try:
            doc_id = self._generate_doc_id(blog_data["url"])

            content = self._prepare_content(blog_data)

            chunks = self._split_content(content)

            documents = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source": blog_data["url"],
                    "title": blog_data.get("title", ""),
                    "authors": blog_data.get("authors", []),
                    "publish_date": blog_data.get("publish_date"),
                    "tags": blog_data.get("tags", []),
                    "extraction_method": blog_data.get("extraction_method", ""),
                    "scraped_at": blog_data.get("scraped_at", ""),
                    "processed_at": datetime.now().isoformat(),
                    "content_type": "blog_post",
                    "chunk_size": len(chunk),
                }

                for k, v in metadata.items():
                    if isinstance(v, list):
                        metadata[k] = ", ".join(v) if v else ""

                if i == 0:
                    metadata["summary"] = blog_data.get("summary", "")

                documents.append(Document(page_content=chunk, metadata=metadata))

            logger.info(
                f"Processed blog post '{blog_data.get('title', 'Unknown')}' into {len(documents)} chunks"
            )
            return documents

        except Exception as e:
            logger.error(f"Error processing blog post: {str(e)}")
            return []

    def _prepare_content(self, blog_data: dict[str, Any]) -> str:
        """Prepare content for chunking by adding structure."""
        content_parts = []

        if blog_data.get("title"):
            content_parts.append(f"# {blog_data['title']}\n")

        metadata_parts = []
        if blog_data.get("authors"):
            authors_str = ", ".join(blog_data["authors"])
            metadata_parts.append(f"Authors: {authors_str}")

        if blog_data.get("publish_date"):
            metadata_parts.append(f"Published: {blog_data['publish_date']}")

        if blog_data.get("tags"):
            tags_str = ", ".join(blog_data["tags"])
            metadata_parts.append(f"Tags: {tags_str}")

        if metadata_parts:
            content_parts.append("\n".join(metadata_parts) + "\n")

        content_parts.append(blog_data.get("content", ""))

        return "\n\n".join(content_parts)

    def _split_content(self, content: str) -> list[str]:
        """Split content into chunks using the text splitter."""
        return self.text_splitter.split_text(content)

    def _generate_doc_id(self, url: str) -> str:
        """Generate a unique document ID from URL."""
        return hashlib.md5(url.encode()).hexdigest()[:16]

    def get_document_summary(self, documents: list[Document]) -> dict[str, Any]:
        """Get a summary of processed documents."""
        if not documents:
            return {}

        first_doc = documents[0]
        total_chars = sum(len(doc.page_content) for doc in documents)

        return {
            "doc_id": first_doc.metadata.get("doc_id"),
            "title": first_doc.metadata.get("title"),
            "source": first_doc.metadata.get("source"),
            "total_chunks": len(documents),
            "total_characters": total_chars,
            "authors": first_doc.metadata.get("authors", []),
            "tags": first_doc.metadata.get("tags", []),
            "publish_date": first_doc.metadata.get("publish_date"),
            "processed_at": first_doc.metadata.get("processed_at"),
        }


class DocumentValidator:
    """Validate processed documents before storage."""

    @staticmethod
    def validate_document(document: Document) -> bool:
        """Check if a document is valid for storage."""
        if not document.page_content or len(document.page_content.strip()) < 50:
            return False

        required_metadata = ["doc_id", "chunk_id", "source", "content_type"]
        for field in required_metadata:
            if field not in document.metadata:
                return False

        return True

    @staticmethod
    def validate_documents(documents: list[Document]) -> list[Document]:
        """Filter out invalid documents."""
        valid_documents = []
        invalid_count = 0

        for doc in documents:
            if DocumentValidator.validate_document(doc):
                valid_documents.append(doc)
            else:
                invalid_count += 1

        if invalid_count > 0:
            logger.warning(f"Filtered out {invalid_count} invalid document chunks")

        return valid_documents
