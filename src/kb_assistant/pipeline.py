"""Конвейер ассистента: вопрос пользователя → поиск в базе → составление промпта → запрос в модель → проверка ссылок в ответе."""

import logging
import time
from dataclasses import dataclass

from kb_assistant.citations import CitationReport, check_citations
from kb_assistant.config import Settings
from kb_assistant.embeddings import Embedder
from kb_assistant.llm import LLMClient, create_llm
from kb_assistant.models import SearchResult
from kb_assistant.prompts import NO_ANSWER, SYSTEM_PROMPT, build_user_prompt
from kb_assistant.vectorstore import VectorStore
from kb_assistant.retrieval import diversify

logger = logging.getLogger(__name__)
CANDIDATE_MULTIPLIER = 5  # чанков из Qdrant берём втрое больше top_k, чтобы было из чего выбирать


@dataclass(frozen=True)
class Answer:
    """Ответ ассистента со всем, что нужно для логов и отчёта."""

    question: str
    text: str
    sources: list[SearchResult]
    citations: CitationReport
    llm_called: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_version: str | None = None
    latency_s: float = 0.0


class Assistant:
    """Корпоративный ассистент: отвечает по базе знаний со ссылками на источники."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMClient,
        top_k: int,
        min_score: float,
        max_per_source: int,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._top_k = top_k
        self._min_score = min_score
        self._max_per_source = max_per_source

    @classmethod
    def from_settings(cls, settings: Settings) -> "Assistant":
        """Собрать ассистента из настроек .env."""
        return cls(
            embedder=Embedder(settings.embedding_model),
            store=VectorStore(settings.qdrant_url, settings.qdrant_collection),
            llm=create_llm(settings),
            top_k=settings.top_k,
            min_score=settings.min_score,
            max_per_source=settings.max_per_source,
        )

    def ask(self, question: str) -> Answer:
        """Ответить на вопрос сотрудника."""
        started = time.perf_counter()
        candidates = self._store.search(self._embedder.embed_query(question), self._top_k * CANDIDATE_MULTIPLIER)
        sources = diversify(candidates, self._top_k, self._max_per_source)
        # если RAG база вернула ответ, берем самый первый чанк (Qdrant возвращает результаты от самого похожего к наименее похожему)
        best_score = sources[0].score if sources else 0.0

        # если чанк похож ниже порогового значения - не вызываем LLM, не тратим деньги, а сразу отвечаем что не знаем
        if best_score < self._min_score:
            logger.info("Лучший score %.3f ниже порога %.3f — LLM не вызываем", best_score, self._min_score)
            return Answer(
                question=question,
                text=NO_ANSWER,
                sources=sources,
                citations=check_citations(NO_ANSWER, sources),
                llm_called=False,
                latency_s=time.perf_counter() - started,
            )

        response = self._llm.generate(SYSTEM_PROMPT, build_user_prompt(question, sources))
        return Answer(
            question=question,
            text=response.text,
            sources=sources,
            citations=check_citations(response.text, sources),
            llm_called=True,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model_version=response.model_version,
            latency_s=time.perf_counter() - started,
        )