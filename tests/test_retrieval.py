"""Тесты отбора чанков: проверяем что есть лимит на количество чанков из одного источника"""

from kb_assistant.models import Chunk, SearchResult, SourceMeta
from kb_assistant.retrieval import diversify


def make_result(source: str, score: float) -> SearchResult:
    meta = SourceMeta(source=source, title=source, doc_type="html")
    return SearchResult(Chunk("текст", meta, 0), score)


def test_other_sources_get_room_when_available():
    candidates = [
        make_result(source, score)
        for source, score in [("rag", 0.89), ("rag", 0.88), ("rag", 0.87),
                              ("hall", 0.86), ("rag", 0.85), ("llm", 0.84)]
    ]
    result = diversify(candidates, top_k=4, max_per_source=2)
    assert [r.chunk.meta.source for r in result] == ["rag", "rag", "hall", "llm"]


def test_single_source_fills_all_slots():
    candidates = [make_result("rag", score) for score in [0.89, 0.88, 0.87, 0.86, 0.85, 0.84]]
    result = diversify(candidates, top_k=4, max_per_source=2)
    assert [r.score for r in result] == [0.89, 0.88, 0.87, 0.86]