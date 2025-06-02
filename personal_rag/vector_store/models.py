from lancedb.pydantic import Vector, LanceModel


class Article(LanceModel):
    id: str
    text: str
    vector: Vector(768)  # type: ignore
    doc_id: str
    title: str
    source: str
    authors: list[str] | None = None
    publish_date: str | None = None
    tags: list[str] | None = None
    chunk_index: int
    total_chunks: int
    content_type: str | None = None
    metadata: str | None = None
