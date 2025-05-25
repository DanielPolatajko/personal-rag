from dependency_injector import containers, providers
from personal_rag.retrieval.vector_store import VectorStoreManager
from personal_rag.retrieval.rag import RAGPipeline
from personal_rag.ingestion.blog_post_scraper import BlogScraper
from personal_rag.ingestion.document_processor import DocumentProcessor


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    vector_store_manager = providers.Singleton(
        VectorStoreManager,
        persist_directory=config.vector_store.persist_directory,
        embedding_model=config.vector_store.embedding_model,
        collection_name=config.vector_store.collection_name,
    )

    rag_pipeline = providers.Singleton(
        RAGPipeline,
        vector_store_manager=vector_store_manager,
        model_name=config.rag.model_name,
        temperature=config.rag.temperature,
        max_tokens=config.rag.max_tokens,
        system_prompt_path=config.rag.system_prompt_path,
        query_prompt_path=config.rag.query_prompt_path,
    )

    blog_scraper = providers.Singleton(
        BlogScraper,
        timeout=config.blog_scraper.timeout,
        user_agent=config.blog_scraper.user_agent,
    )

    document_processor = providers.Singleton(
        DocumentProcessor,
        chunk_size=config.document_processor.chunk_size,
        chunk_overlap=config.document_processor.chunk_overlap,
    )
