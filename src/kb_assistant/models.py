"""Модели данных: из чего состоит база знаний."""
# Путь целиком: файл → Section (страница) → Chunk (кусочки с meta) → Qdrant → SearchResult (найденное + оценка) → ответ со ссылкой.

from dataclasses import dataclass
from typing import Literal

DocType = Literal["pdf", "html"]


@dataclass(frozen=True)
class SourceMeta:
    """«Бирка»: откуда взят текст."""

    source: str                     # имя файла: gost_r_59898_2021.pdf
    title: str                      # название документа
    doc_type: DocType               # "pdf" или "html"
    page: int | None = None         # физический номер страницы в PDF, с 1
    page_label: str | None = None   # номер, напечатанный на самой странице
    header: str | None = None       # ближайший заголовок раздела

    @property
    def citation(self) -> str:
        """Ссылка в формате из задания."""
        if self.doc_type == "pdf":
            return f"[{self.source}, стр. {self.page}]"
        return f"[{self.title}]"

@dataclass(frozen=True)
class Section:
    """Фрагмент документа после загрузки, до нарезки на чанки."""

    text: str
    meta: SourceMeta


@dataclass(frozen=True)
class Chunk:
    """Chunk после нарезки — единица поиска."""

    text: str
    meta: SourceMeta
    # chunk_index - порядковый номер chunk в документе , нужен чтобы сгенерировать id чанка 
    # чтобы. Иначе автосгенереный id всегда будет новый при индексации и в базе
    # будуд одинаковые чанки.
    chunk_index: int
    
    @property
    def embedding_text(self) -> str:
        """Добавляем к тексту чанк заголовок если его там нет"""
        # Длинный раздел режется на несколько чанков, и заголовок есть только в первом. 
        # поиск по чанкам может не найти подходящий потому нет заголовка. 
        # при векторизации приклеим к каждому чанку заголовок его раздела. 
        # Сам текст чанка не меняется: заголовок добавляется только в то, что превращается в вектор. 
        # Приём называют контекстными заголовками чанков (contextual chunk headers).
        header = self.meta.header
        if header and not self.text.startswith(header):
            return f"{header}\n{self.text}"
        return self.text                


@dataclass(frozen=True)
class SearchResult:
    """Найденный в RAG chunk и насколько он похож на вопрос."""

    chunk: Chunk
    score: float