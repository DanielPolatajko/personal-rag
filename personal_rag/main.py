import streamlit as st
import logging
from personal_rag.container import Container
from dotenv import load_dotenv
from personal_rag.ui.app import RAGApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        load_dotenv()
        logger.info("Loading configurations...")
        container = Container()
        container.config.from_yaml("personal_rag/config/config.yaml")
        logger.info("Configurations loaded successfully")
        logger.info("Initializing RAG application...")
        app = RAGApp(
            vector_store_manager=container.lancedb_vector_store_manager(),
            rag_pipeline=container.rag_pipeline(),
            blog_scraper=container.blog_scraper(),
            document_processor=container.document_processor(),
        )
        logger.info("RAG application initialized successfully")
        logger.info("Starting application...")
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
