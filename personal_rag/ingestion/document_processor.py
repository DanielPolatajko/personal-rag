from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any
import hashlib
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk documents for vector storage."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process_blog_post(self, blog_data: Dict[str, Any]) -> List[Document]:
        """
        Process a blog post into chunks with metadata.

        Args:
            blog_data: Dictionary containing blog post data from scraper

        Returns:
            List of LangChain Document objects ready for vector storage
        """
        try:
            # Generate unique document ID
            doc_id = self._generate_doc_id(blog_data["url"])

            # Prepare content for chunking
            content = self._prepare_content(blog_data)

            # Split into chunks
            chunks = self._split_content(content)

            # Create Document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                # Base metadata for all chunks
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

                # Add summary to first chunk metadata
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

    def _prepare_content(self, blog_data: Dict[str, Any]) -> str:
        """Prepare content for chunking by adding structure."""
        content_parts = []

        # Add title as header if available
        if blog_data.get("title"):
            content_parts.append(f"# {blog_data['title']}\n")

        # Add metadata section
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

        # Add main content
        content_parts.append(blog_data.get("content", ""))

        return "\n\n".join(content_parts)

    def _split_content(self, content: str) -> List[str]:
        """Split content into chunks using the text splitter."""
        return self.text_splitter.split_text(content)

    def _generate_doc_id(self, url: str) -> str:
        """Generate a unique document ID from URL."""
        return hashlib.md5(url.encode()).hexdigest()[:16]

    def get_document_summary(self, documents: List[Document]) -> Dict[str, Any]:
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
        # Check if content exists and is meaningful
        if not document.page_content or len(document.page_content.strip()) < 50:
            return False

        # Check if required metadata exists
        required_metadata = ["doc_id", "chunk_id", "source", "content_type"]
        for field in required_metadata:
            if field not in document.metadata:
                return False

        return True

    @staticmethod
    def validate_documents(documents: List[Document]) -> List[Document]:
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


# Utility functions for document management
def save_document_metadata(documents: List[Document], filepath: str):
    """Save document metadata to JSON file for reference."""
    if not documents:
        return

    processor = DocumentProcessor()
    summary = processor.get_document_summary(documents)

    # Add chunk-level metadata
    summary["chunks"] = []
    for doc in documents:
        chunk_info = {
            "chunk_id": doc.metadata.get("chunk_id"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "chunk_size": len(doc.page_content),
            "preview": doc.page_content[:200] + "..."
            if len(doc.page_content) > 200
            else doc.page_content,
        }
        summary["chunks"].append(chunk_info)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Document metadata saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")


def load_document_metadata(filepath: str) -> Dict[str, Any]:
    """Load document metadata from JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        return {}
