"""Общий интерфейс для всех загрузчиков документов."""

from abc import ABC, abstractmethod
from pathlib import Path

from kb_assistant.models import Section


class DocumentLoader(ABC):

    # сюда вносятся раширения, которые умеет читать загрузчик 
    extensions: tuple[str, ...] = ()

    def can_load(self, path: Path) -> bool:
        """Подходит ли файл этому загрузчику (по расширению)."""
        return path.suffix.lower() in self.extensions

    @abstractmethod
    def load(self, path: Path) -> list[Section]:
        """Прочитать файл и вернуть список разделов с meta"""