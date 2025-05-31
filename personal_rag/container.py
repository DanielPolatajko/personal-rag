from dependency_injector import containers, providers
from vector_store.chroma import ChromaVectorStoreManager
from personal_rag.retrieval.rag import RAGPipeline
from personal_rag.ingestion.blog_post_scraper import BlogScraper
from personal_rag.ingestion.document_processor import DocumentProcessor
from personal_rag.embeddings.gemini import GeminiEmbeddingClient
from personal_rag.embeddings.huggingface import HuggingFaceEmbeddingsClient
from vector_store.lancedb import LanceDBVectorStoreManager


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    huggingface_embeddings_client = providers.Singleton(
        HuggingFaceEmbeddingsClient,
        model_name=config.embeddings.huggingface.model_name,
    )

    chroma_vector_store_manager = providers.Singleton(
        ChromaVectorStoreManager,
        persist_directory=config.chroma_vector_store.persist_directory,
        embeddings_client=huggingface_embeddings_client,
        collection_name=config.chroma_vector_store.collection_name,
    )

    rag_pipeline = providers.Singleton(
        RAGPipeline,
        vector_store_manager=chroma_vector_store_manager,
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

    gemini_embedding = providers.Singleton(
        GeminiEmbeddingClient,
        api_key=config.embeddings.gemini.api_key,
        model=config.embeddings.gemini.model,
    )

    lancedb_vector_store_manager = providers.Singleton(
        LanceDBVectorStoreManager,
        db_path=config.lancedb_vector_store.db_path,
        embedding_client=gemini_embedding,
    )
