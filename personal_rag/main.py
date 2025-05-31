from personal_rag.ui.app import RAGApp
import streamlit as st
import logging
import torch
from personal_rag.container import Container
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        load_dotenv()
        torch.classes.__path__ = []  # TODO: why is this necessary?
        logger.info("Loading configurations...")
        container = Container()
        container.config.from_yaml("personal_rag/config/config.yaml")
        logger.info("Configurations loaded successfully")
        logger.info("Initializing RAG application...")
        app = RAGApp(
            vector_store_manager=container.chroma_vector_store_manager(),
            rag_pipeline=container.rag_pipeline(),
        )
        logger.info("RAG application initialized successfully")
        logger.info("Starting application...")
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()
