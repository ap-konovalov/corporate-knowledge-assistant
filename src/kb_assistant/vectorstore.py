"""Хранилище векторов в Qdrant: запись чанков с метаданными и поиск."""

import uuid
from dataclasses import asdict, fields

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from kb_assistant.models import Chunk, SearchResult, SourceMeta


class VectorStore:
    """Коллекция чанков в Qdrant."""

    def __init__(self, url: str, collection: str) -> None:
        self._client = QdrantClient(url=url)
        self._collection = collection

    def recreate(self, dimension: int) -> None:
        """Удалить коллекцию, если есть, и создать пустую."""
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Записать чанки с векторами; запись с тем же ID перезаписывается."""
        points = [
            PointStruct(id=_point_id(chunk), vector=vector, payload=_to_payload(chunk))
            # идти по двум спискам одновременно, парами: чанк - вектор чанка. strict=True — если списки разной длины, упасть с ошибкой.
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        """Найти top_k чанков, ближайших к вектору вопроса."""
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchResult(chunk=_from_payload(point.payload), score=point.score)
            for point in response.points
        ]


def _point_id(chunk: Chunk) -> str:
    """Один и тот же чанк всегда получает один и тот же ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.meta.source}#{chunk.chunk_index}"))


def _to_payload(chunk: Chunk) -> dict:
    """Переписываем массив meta потому что Qdrant принимает только простой словарь: пары «ключ: значение» с текстом и числами"""
    return {"text": chunk.text, "chunk_index": chunk.chunk_index, **asdict(chunk.meta)}


def _from_payload(payload: dict) -> Chunk:
    """Собираем meta из Qdrant → обратно в Chunk. Meta было плоски словарь в Qdrant - стало объект"""
    meta = SourceMeta(**{field.name: payload.get(field.name) for field in fields(SourceMeta)})
    return Chunk(text=payload["text"], meta=meta, chunk_index=payload["chunk_index"])