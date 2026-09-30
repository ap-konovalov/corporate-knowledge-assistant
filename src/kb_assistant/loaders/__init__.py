"""Загрузка всех документов из папки: загрузчик выбирается по расширению файла."""

import logging
from pathlib import Path

from kb_assistant.loaders.base import DocumentLoader
from kb_assistant.loaders.html import HtmlLoader
from kb_assistant.loaders.pdf import PdfLoader
from kb_assistant.models import Section

logger = logging.getLogger(__name__)

# создаем экземпляры классов загрузчиков
LOADERS: list[DocumentLoader] = [PdfLoader(), HtmlLoader()]


def load_directory(directory: Path) -> list[Section]:
    """Прочитать все поддерживаемые файлы из папки."""
    sections: list[Section] = []
    for path in sorted(directory.iterdir()):
        if path.name.startswith("."):
            continue  # служебные файлы: .gitkeep, .DS_Store
        loader = _pick_loader(path)
        if loader is None:
            logger.warning("Нет загрузчика для файла, пропущен: %s", path.name)
            continue
        file_sections = loader.load(path)
        logger.info("%s: %d разделов", path.name, len(file_sections))
        # добавляем секции из всех загруженных файлов в один список sections
        sections.extend(file_sections)
    return sections


def _pick_loader(path: Path) -> DocumentLoader | None:
    for loader in LOADERS:
        if loader.can_load(path):
            # перебираем загрузчики возвращаем подходящий если по расширению загрузчик подходит
            return loader
    return None