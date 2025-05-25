import streamlit as st
from datetime import datetime
import logging
from typing import Dict, Any, Optional
from dependency_injector.wiring import inject, Provide

from personal_rag.ingestion.blog_post_scraper import BlogScraper
from personal_rag.ingestion.document_processor import (
    DocumentProcessor,
    DocumentValidator,
)
from personal_rag.retrieval.vector_store import VectorStoreManager
from personal_rag.retrieval.rag import RAGPipeline
from personal_rag.utils.config import load_config
from personal_rag.container import Container

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Page config
st.set_page_config(
    page_title="Personal RAG Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


class RAGApp:
    """Main Streamlit RAG application."""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        rag_pipeline: RAGPipeline,
    ):
        self.vector_store_manager = vector_store_manager
        self.rag_pipeline = rag_pipeline
        self.config = load_config()
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state."""

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if "ingestion_status" not in st.session_state:
            st.session_state.ingestion_status = []

    def run(self):
        """Run the main application."""
        st.title("🔬 Personal RAG Research Assistant")
        st.markdown(
            "*Ingest blog posts and papers, then query your personal knowledge base with AI assistance.*"
        )

        # Sidebar
        self.render_sidebar()

        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs(
            ["💬 Chat", "📄 Ingest Documents", "🔍 Browse Documents", "📊 Statistics"]
        )

        with tab1:
            self.render_chat_interface()

        with tab2:
            self.render_ingestion_interface()

        with tab3:
            self.render_browse_interface()

        with tab4:
            self.render_statistics_interface()

    def render_sidebar(self):
        """Render the sidebar with controls."""
        st.sidebar.header("🛠️ Controls")

        # Collection statistics
        stats = self.vector_store_manager.get_collection_stats()
        st.sidebar.metric("Documents", stats.get("unique_documents", 0))
        st.sidebar.metric("Chunks", stats.get("total_chunks", 0))

        st.sidebar.divider()

        # Query settings
        st.sidebar.subheader("Query Settings")
        search_k = st.sidebar.slider("Results to retrieve", 1, 20, 5)
        include_metadata = st.sidebar.checkbox("Include metadata", True)

        # Content filters
        st.sidebar.subheader("Content Filters")
        filter_by_type = st.sidebar.selectbox(
            "Content Type", ["All", "blog_post", "paper"], index=0
        )

        filter_by_author = st.sidebar.text_input("Author (optional)")
        filter_by_tags = st.sidebar.text_input("Tags (comma-separated)")

        # Store settings in session state
        st.session_state.search_settings = {
            "k": search_k,
            "include_metadata": include_metadata,
            "content_type": filter_by_type if filter_by_type != "All" else None,
            "author": filter_by_author if filter_by_author else None,
            "tags": [tag.strip() for tag in filter_by_tags.split(",") if tag.strip()]
            if filter_by_tags
            else None,
        }

        st.sidebar.divider()

        # Admin actions
        st.sidebar.subheader("Admin")
        if st.sidebar.button("🗑️ Reset Collection", type="secondary"):
            if st.sidebar.checkbox("Confirm reset"):
                with st.spinner("Resetting collection..."):
                    success = self.vector_store_manager.reset_collection()
                    if success:
                        st.success("Collection reset successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to reset collection")

    def render_chat_interface(self):
        """Render the main chat interface."""
        st.header("💬 Chat with Your Knowledge Base")

        # Display chat history
        for i, entry in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(entry["question"])

            with st.chat_message("assistant"):
                st.write(entry["answer"])

                # Show sources
                if entry.get("sources"):
                    with st.expander(f"📚 Sources ({len(entry['sources'])})"):
                        for j, source in enumerate(entry["sources"], 1):
                            st.markdown(f"""
                            **{j}. {source["title"]}**  
                            *Authors: {", ".join(source["authors"]) if source["authors"] else "Unknown"}*  
                            *Relevance: {source["relevance_score"]:.3f}*  
                            [Source]({source["source"]})
                            """)

        # Chat input
        question = st.chat_input("Ask a question about your documents...")

        if question:
            # Build filters from sidebar settings
            settings = st.session_state.search_settings
            filters = self._build_filters(settings)

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching and generating answer..."):
                    result = self.rag_pipeline.query(
                        question=question,
                        k=settings["k"],
                        filters=filters,
                        include_metadata=settings["include_metadata"],
                    )

                st.write(result["answer"])

                # Show sources
                if result.get("sources"):
                    with st.expander(f"📚 Sources ({len(result['sources'])})"):
                        for i, source in enumerate(result["sources"], 1):
                            st.markdown(f"""
                            **{i}. {source["title"]}**  
                            *Authors: {", ".join(source["authors"]) if source["authors"] else "Unknown"}*  
                            *Relevance: {source["relevance_score"]:.3f}*  
                            [Source]({source["source"]})
                            """)

                # Add to chat history
                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": result["answer"],
                        "sources": result.get("sources", []),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

    def render_ingestion_interface(self):
        """Render the document ingestion interface."""
        st.header("📄 Ingest New Documents")

        # Blog post ingestion
        st.subheader("🌐 Add Blog Post from URL")

        col1, col2 = st.columns([3, 1])

        with col1:
            blog_url = st.text_input(
                "Blog Post URL", placeholder="https://example.com/blog-post"
            )

        with col2:
            st.write("")  # Spacing
            ingest_button = st.button("📥 Ingest", type="primary")

        if ingest_button and blog_url:
            self._ingest_blog_post(blog_url)

        # Ingestion status
        if st.session_state.ingestion_status:
            st.subheader("📈 Recent Ingestion Status")
            for status in st.session_state.ingestion_status[-5:]:  # Show last 5
                icon = "✅" if status["success"] else "❌"
                st.write(f"{icon} {status['url']} - {status['message']}")

    def render_browse_interface(self):
        """Render the document browsing interface."""
        st.header("🔍 Browse Documents")

        # Search interface
        search_col1, search_col2 = st.columns([3, 1])

        with search_col1:
            search_query = st.text_input(
                "Search documents", placeholder="Enter search terms..."
            )

        with search_col2:
            st.write("")
            search_button = st.button("🔍 Search")

        if search_button and search_query:
            settings = st.session_state.search_settings
            filters = self._build_filters(settings)

            results = self.rag_pipeline.search_documents(
                query=search_query,
                k=settings["k"] * 2,  # Get more results for browsing
                filters=filters,
            )

            if results:
                st.subheader(f"Found {len(results)} results")
                for result in results:
                    with st.expander(
                        f"📄 {result['title']} (Relevance: {result['relevance_score']:.3f})"
                    ):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.write(
                                f"**Authors:** {', '.join(result['authors']) if result['authors'] else 'Unknown'}"
                            )
                            st.write(
                                f"**Published:** {result['publish_date'] or 'Unknown'}"
                            )
                            st.write(
                                f"**Tags:** {', '.join(result['tags']) if result['tags'] else 'None'}"
                            )
                            st.write("**Content Preview:**")
                            st.write(result["content_preview"])

                        with col2:
                            st.write(
                                f"**Chunk:** {result['chunk_index'] + 1}/{result['total_chunks']}"
                            )
                            if st.button(
                                "View Full Document", key=f"view_{result['doc_id']}"
                            ):
                                self._show_full_document(result["doc_id"])
                            st.link_button("🔗 Source", result["source"])
            else:
                st.info("No documents found matching your search.")

        # List all documents
        st.subheader("📚 All Documents")
        documents = self.vector_store_manager.list_documents()

        if documents:
            for doc in documents:
                with st.expander(f"📄 {doc['title']}"):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.write(
                            f"**Authors:** {', '.join(doc['authors']) if doc['authors'] else 'Unknown'}"
                        )
                        st.write(f"**Published:** {doc['publish_date'] or 'Unknown'}")
                        st.write(
                            f"**Tags:** {', '.join(doc['tags']) if doc['tags'] else 'None'}"
                        )
                        st.write(f"**Chunks:** {doc['chunk_count']}")

                    with col2:
                        if st.button("👁️ View", key=f"view_all_{doc['doc_id']}"):
                            self._show_full_document(doc["doc_id"])
                        if st.button("🗑️ Delete", key=f"delete_{doc['doc_id']}"):
                            if self.vector_store_manager.delete_document(doc["doc_id"]):
                                st.success("Document deleted!")
                                st.rerun()
                            else:
                                st.error("Failed to delete document")
                        st.link_button("🔗 Source", doc["source"])
        else:
            st.info(
                "No documents in the knowledge base yet. Add some using the Ingest tab!"
            )

    def render_statistics_interface(self):
        """Render statistics and analytics."""
        st.header("📊 Knowledge Base Statistics")

        stats = self.vector_store_manager.get_collection_stats()

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Documents", stats.get("unique_documents", 0))

        with col2:
            st.metric("Total Chunks", stats.get("total_chunks", 0))

        with col3:
            st.metric("Total Characters", f"{stats.get('total_characters', 0):,}")

        with col4:
            st.metric("Avg Chunk Size", f"{stats.get('average_chunk_size', 0):.0f}")

        # Content type breakdown
        if stats.get("content_types"):
            st.subheader("📋 Content Types")
            content_types = stats["content_types"]

            for content_type, count in content_types.items():
                st.write(f"**{content_type.title()}:** {count} chunks")

        # Collection info
        st.subheader("🗄️ Collection Information")
        st.write(f"**Collection Name:** {stats.get('collection_name', 'Unknown')}")
        st.write(f"**Storage Directory:** {stats.get('persist_directory', 'Unknown')}")

        # Recent activity
        st.subheader("📈 Recent Activity")
        if st.session_state.ingestion_status:
            successful = sum(
                1 for status in st.session_state.ingestion_status if status["success"]
            )
            total = len(st.session_state.ingestion_status)
            st.write(
                f"**Ingestion Success Rate:** {successful}/{total} ({successful / total * 100:.1f}%)"
            )

            # Show recent ingestions
            for status in st.session_state.ingestion_status[-10:]:  # Last 10
                icon = "✅" if status["success"] else "❌"
                timestamp = status.get("timestamp", "Unknown time")
                st.write(f"{icon} {timestamp}: {status['url']} - {status['message']}")
        else:
            st.info("No ingestion activity yet.")

    def _build_filters(self, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build search filters from settings."""
        from retrieval.vector_store import SearchFilters

        filters = []

        if settings.get("content_type"):
            filters.append(SearchFilters.by_content_type(settings["content_type"]))

        if settings.get("author"):
            filters.append(SearchFilters.by_author(settings["author"]))

        if settings.get("tags"):
            for tag in settings["tags"]:
                filters.append(SearchFilters.by_tag(tag))

        return SearchFilters.combine_filters(*filters) if filters else None

    @inject
    def _ingest_blog_post(
        self,
        url: str,
        blog_scraper: BlogScraper = Provide[Container.blog_scraper],
        document_processor: DocumentProcessor = Provide[Container.document_processor],
    ):
        """Ingest a single blog post."""
        with st.spinner(f"Ingesting {url}..."):
            try:
                blog_data = blog_scraper.scrape_blog_post(url)

                if not blog_data:
                    st.error("Failed to scrape blog post. Please check the URL.")
                    self._log_ingestion_status(url, False, "Scraping failed")
                    return

                documents = document_processor.process_blog_post(blog_data)
                valid_documents = DocumentValidator.validate_documents(documents)

                if not valid_documents:
                    st.error("No valid document chunks created.")
                    self._log_ingestion_status(url, False, "No valid chunks")
                    return

                ids = self.vector_store_manager.add_documents(valid_documents)

                if ids:
                    st.success(
                        f"Successfully ingested '{blog_data['title']}' ({len(valid_documents)} chunks)"
                    )
                    self._log_ingestion_status(
                        url, True, f"Added {len(valid_documents)} chunks"
                    )
                else:
                    st.warning("Document may already exist in the knowledge base.")
                    self._log_ingestion_status(url, False, "Already exists")

            except Exception as e:
                st.error(f"Error ingesting document: {str(e)}")
                self._log_ingestion_status(url, False, f"Error: {str(e)}")

    def _show_full_document(self, doc_id: str):
        """Show full document content in a modal."""
        doc_content = self.rag_pipeline.get_document_content(doc_id)

        if doc_content:
            st.subheader(f"📄 {doc_content['title']}")

            # Metadata
            col1, col2 = st.columns(2)
            with col1:
                st.write(
                    f"**Authors:** {', '.join(doc_content['authors']) if doc_content['authors'] else 'Unknown'}"
                )
                st.write(f"**Published:** {doc_content['publish_date'] or 'Unknown'}")
                st.write(
                    f"**Tags:** {', '.join(doc_content['tags']) if doc_content['tags'] else 'None'}"
                )

            with col2:
                st.write(f"**Chunks:** {doc_content['chunk_count']}")
                st.write(f"**Characters:** {doc_content['total_characters']:,}")
                st.link_button("🔗 Original Source", doc_content["source"])

            # Content
            st.subheader("📝 Content")
            st.text_area(
                "Document Content", doc_content["content"], height=400, disabled=True
            )
        else:
            st.error("Could not retrieve document content.")

    def _log_ingestion_status(self, url: str, success: bool, message: str):
        """Log ingestion status."""
        status = {
            "url": url,
            "success": success,
            "message": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.session_state.ingestion_status.append(status)

        # Keep only last 100 entries
        if len(st.session_state.ingestion_status) > 100:
            st.session_state.ingestion_status = st.session_state.ingestion_status[-100:]
