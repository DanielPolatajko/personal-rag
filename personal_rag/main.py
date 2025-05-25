from personal_rag.ui.app import RAGApp
import streamlit as st
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    try:
        torch.classes.__path__ = []  # TODO: why is this necessary?
        app = RAGApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()
