"""Проверка ссылок в ответе модели. В ответе должны быть реальные источники откуда взят ответ."""

import re
from dataclasses import dataclass

from kb_assistant.models import SearchResult
from kb_assistant.prompts import NO_ANSWER

# Ссылка: текст в квадратных скобках, внутри которого нет других скобок
CITATION_RE = re.compile(r"\[[^\[\]]+\]")
# Граница предложения: после . ! ? идут пробелы, и следом НЕ открывается ссылка
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?!\[)")


@dataclass(frozen=True)
class CitationReport:
    """Результат проверки ссылок в одном ответе."""

    is_refusal: bool               # модель ответила «Я не знаю»
    valid: list[str]               # ссылки, которые были в контексте
    invalid: list[str]             # ссылки, которых в контексте не было
    uncited_sentences: list[str]   # предложения без единой ссылки

    @property
    def is_ok(self) -> bool:
        """Ответ прошёл проверку."""
        if self.is_refusal:
            return True
        return bool(self.valid) and not self.invalid and not self.uncited_sentences


def check_citations(answer: str, results: list[SearchResult]) -> CitationReport:
    """Сверить ссылки в ответе с источниками, которые реально были в контексте."""
    # если модель отвечает что не знает - проверка пройдена, так как это нормальный ответ
    if answer.strip().rstrip(".") == NO_ANSWER:
        return CitationReport(is_refusal=True, valid=[], invalid=[], uncited_sentences=[])

    # из всех чанков, которые использовались в запросе взять ссылки
    allowed = {result.chunk.meta.citation for result in results}
    # взять все ссылки, которые пришли в ответе языковой модели
    found = CITATION_RE.findall(answer)
    # нарезаем ответ на отдельные предложения
    # Ссылки заменяем меткой без точек, чтобы «стр. 4» не считалось концом предложения
    masked = CITATION_RE.sub("[ref]", answer.strip())
    sentences = [s for s in SENTENCE_SPLIT_RE.split(masked) if s]
    return CitationReport(
        is_refusal=False,
        # если в ответе модели пришла ссылка, котора есть в списке ссылок, полученном из чанков в базе - кладем эту ссылку в список корректных ссылок
        valid=[c for c in found if c in allowed],
        invalid=[c for c in found if c not in allowed],
        # собираем в список предложения, в которых нет ссылок
        uncited_sentences=[s for s in sentences if not CITATION_RE.search(s)],
    )