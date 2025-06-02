from google import genai
from google.genai import types
from langchain_core.documents import Document
from personal_rag.embeddings.embeddings_client import BaseEmbeddingsClient


class GeminiEmbeddingClient(BaseEmbeddingsClient):
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed_content(self, contents: str) -> list[types.ContentEmbedding]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=contents,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        return result.embeddings

    def embed_documents(
        self, documents: list[Document]
    ) -> list[list[types.ContentEmbedding]]:
        """Embed multiple documents.

        Args:
            documents: List of LangChain documents to embed

        Returns:
            List of embedding vectors, one for each document
        """
        return [self.embed_content(doc.page_content) for doc in documents]
