"""Индексация: прочитать документы → нарезать на чанки → векторизовать → записать в Qdrant.

Запуск из корня проекта:
    uv run python scripts/ingest.py
"""

import logging
import time

from kb_assistant.chunking import chunk_sections
from kb_assistant.config import get_settings
from kb_assistant.embeddings import Embedder
from kb_assistant.loaders import load_directory
from kb_assistant.vectorstore import VectorStore

logger = logging.getLogger("ingest")


def main() -> None:
    settings = get_settings()
    started = time.perf_counter()

    sections = load_directory(settings.raw_data_dir)
    chunks = chunk_sections(sections, settings.chunk_size, settings.chunk_overlap)
    logger.info(
        "Разделов: %d, чанков: %d (chunk_size=%d, overlap=%d)",
        len(sections), len(chunks), settings.chunk_size, settings.chunk_overlap,
    )

    embedder = Embedder(settings.embedding_model)
    # chunk.embedding_text - векторизум текс чанка + заголовок раздела к которому он относится
    vectors = embedder.embed_passages([chunk.embedding_text for chunk in chunks])

    store = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    store.recreate(embedder.dimension)
    # в payload.text будем сохранять текст чанка без заголовка раздела
    store.upsert(chunks, vectors)

    logger.info(
        "Готово: %d чанков в коллекции «%s», модель %s, за %.1f с",
        len(chunks), settings.qdrant_collection, settings.embedding_model,
        time.perf_counter() - started,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()