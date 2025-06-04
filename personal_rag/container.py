from dependency_injector import containers, providers
from personal_rag.retrieval.rag import RAGPipeline
from personal_rag.ingestion.blog_post_scraper import BlogScraper
from personal_rag.ingestion.document_processor import DocumentProcessor
from personal_rag.embeddings.gemini import GeminiEmbeddingClient
from personal_rag.vector_store.lancedb import LanceDBVectorStoreManager


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

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

    gemini_embedding = providers.Singleton(
        GeminiEmbeddingClient,
        api_key=config.embeddings.gemini.api_key,
        model=config.embeddings.gemini.model,
    )

    lancedb_vector_store_manager = providers.Singleton(
        LanceDBVectorStoreManager,
        db_path=config.lancedb_vector_store.db_path,
        service_account_path=config.lancedb_vector_store.service_account_path,
        embedding_client=gemini_embedding,
    )

    rag_pipeline = providers.Singleton(
        RAGPipeline,
        vector_store_manager=lancedb_vector_store_manager,
        model_name=config.rag.model_name,
        temperature=config.rag.temperature,
        max_tokens=config.rag.max_tokens,
        system_prompt_path=config.rag.system_prompt_path,
        query_prompt_path=config.rag.query_prompt_path,
    )
