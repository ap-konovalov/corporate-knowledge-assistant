"""Загрузчик HTML-выгрузок Википедии: делит на секции текст статьи по разделам."""

from pathlib import Path

from bs4 import BeautifulSoup

from kb_assistant.loaders.base import DocumentLoader
from kb_assistant.models import Section, SourceMeta

# Элементы без содержания: сноски [1], «[править]», навигация, скрипты
NOISE_SELECTORS = [
    "script", "style", "sup.reference", ".mw-editsection",
    ".navbox", ".toc", ".metadata", ".ambox", ".mw-references-wrap",
]
# Служебные разделы в конце статьи — дальше читать незачем
STOP_SECTIONS = {"См. также", "Примечания", "Литература", "Ссылки"}
HEADING_TAGS = {"h2", "h3", "h4"}
TITLE_SUFFIX = " — Википедия"


class HtmlLoader(DocumentLoader):
    """Читает сохранённую статью Википедии и режет её на разделы по заголовкам."""

    extensions = (".html", ".htm")

    def load(self, path: Path) -> list[Section]:
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        title = _extract_title(soup, path)

        content = soup.select_one("#mw-content-text") or soup.body
        if content is None:
            raise ValueError(f"{path.name}: не найден текст статьи")
        for noise in content.select(", ".join(NOISE_SELECTORS)):
            noise.decompose()

        sections: list[Section] = []
        header: str | None = None
        buffer: list[str] = []
        for tag in content.find_all(["h2", "h3", "h4", "p", "li"]):
            if tag.name == "li" and tag.find_parent("li"):
                continue  # вложенный пункт уже вошёл в текст родительского
            text = " ".join(tag.get_text().split())
            if not text:
                continue
            if tag.name in HEADING_TAGS:
                if text in STOP_SECTIONS:
                    break
                if buffer:
                    # если дошли до гового заголовка - создаем секцию с текстом из буфера
                    sections.append(_make_section(buffer, header, path.name, title))
                    buffer = []
                header = text
            buffer.append(text)
        if buffer:
            sections.append(_make_section(buffer, header, path.name, title))

        if not sections:
            raise ValueError(f"{path.name}: в статье не найден текст")
        return sections


def _extract_title(soup: BeautifulSoup, path: Path) -> str:
    """Название статьи: из заголовка страницы, иначе из <title>, иначе из имени файла."""
    heading = soup.select_one("#firstHeading")
    if heading:
        return heading.get_text(strip=True)
    if soup.title and soup.title.string:
        return soup.title.string.removesuffix(TITLE_SUFFIX).strip()
    return path.stem


def _make_section(lines: list[str], header: str | None, source: str, title: str) -> Section:
    meta = SourceMeta(source=source, title=title, doc_type="html", header=header)
    return Section(text="\n".join(lines), meta=meta)