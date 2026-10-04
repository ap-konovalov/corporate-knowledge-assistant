"""Загрузчик PDF: текст по страницам, номер страницы, напечатанный номер, заголовок."""

import re
from pathlib import Path
from typing import NamedTuple

import pymupdf

from kb_assistant.loaders.base import DocumentLoader
from kb_assistant.models import Section, SourceMeta

# Заголовок раздела: «4 Общие положения», «5.2 Оценка ...», «Приложение А»
HEADER_RE = re.compile(r"^(?:\d+(?:\.\d+)*|Приложение\s+[А-ЯA-Z])\.?\s+[А-ЯЁA-Z]")
# Номер страницы в колонтитуле: «7» или римское «IV»
PAGE_NUMBER_RE = re.compile(r"^(?:\d{1,3}|[IVXLC]{1,6})$")
TOC_LEADER_RE = re.compile(r"\.{4,}")  # «.......» — точки-заполнители оглавления
MAX_HEADER_LEN = 120
TOP_MARGIN = 0.08     # верхние 8 % высоты страницы — колонтитул (шифр ГОСТа)
BOTTOM_MARGIN = 0.12  # нижние 12 % — номер страницы и служебные надписи
BOLD_FLAG = 16        # признак «жирный шрифт» в PyMuPDF
SOFT_HYPHEN = "\xad"  # мягкий перенос: невидимый знак «слово перенесено»
UNNUMBERED_HEADERS = {"Предисловие", "Введение", "Библиография"}


class Line(NamedTuple):
    """Строка страницы и признак, набрана ли она жирным."""

    text: str
    bold: bool


class PdfLoader(DocumentLoader):
    """Читает PDF по страницам и режет страницы на разделы по заголовкам."""

    extensions = (".pdf",)

    def load(self, path: Path) -> list[Section]:
        sections: list[Section] = []
        header: str | None = None  # заголовок «переезжает» на следующие страницы

        with pymupdf.open(path) as doc:
            for page in doc:
                body, margins = _split_lines(page)
                meta = {
                    "source": path.name,
                    "title": path.stem,
                    "doc_type": "pdf",
                    "page": page.number + 1,
                    "page_label": page.get_label() or _find_printed_number(margins),
                }
                buffer: list[str] = []
                for line in body:
                    if _is_header(line):
                        if buffer:
                            # если дошли до заголовка и буфер не пуст - создаем новую секцию
                            sections.append(_make_section(buffer, header, meta))
                            buffer = []
                        header = line.text
                    buffer.append(line.text)
                if buffer:
                    sections.append(_make_section(buffer, header, meta))

        if not sections:
            raise ValueError(f"{path.name}: в PDF нет текстового слоя — возможно, это скан")
        return sections


def _split_lines(page: pymupdf.Page) -> tuple[list[Line], list[str]]:
    """Разделить строки страницы на основной текст и колонтитулы."""
    height = page.rect.height
    top_edge, bottom_edge = height * TOP_MARGIN, height * (1 - BOTTOM_MARGIN)
    body: list[Line] = []
    margins: list[str] = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        for raw_line in block.get("lines", []):
            spans = [span for span in raw_line["spans"] if span["text"].strip()]
            if not spans:
                continue
            text = "".join(span["text"] for span in spans).strip()
            y0, y1 = raw_line["bbox"][1], raw_line["bbox"][3]
            if y1 < top_edge or y0 > bottom_edge:
                margins.append(text)
            else:
                body.append(Line(text=text, bold=bool(spans[0]["flags"] & BOLD_FLAG)))
    return body, margins


def _find_printed_number(margin_lines: list[str]) -> str | None:
    """Найти номер, напечатанный на странице: ищем снизу вверх."""
    for line in reversed(margin_lines):
        if PAGE_NUMBER_RE.match(line):
            return line
    return None


def _is_header(line: Line) -> bool:
    """Заголовок = жирная строка: «Введение» и т. п. либо номер раздела не из оглавления."""
    if not line.bold:
        return False
    if line.text in UNNUMBERED_HEADERS:
        return True
    return (
        len(line.text) <= MAX_HEADER_LEN
        and not TOC_LEADER_RE.search(line.text)
        and HEADER_RE.match(line.text) is not None
    )


def _make_section(lines: list[str], header: str | None, meta: dict) -> Section:
    return Section(text=_join_lines(lines), meta=SourceMeta(header=header, **meta))


def _join_lines(lines: list[str]) -> str:
    """Склеить строки; слова, разорванные мягким переносом, собрать обратно."""
    text = "\n".join(lines)
    return text.replace(SOFT_HYPHEN + "\n", "").replace(SOFT_HYPHEN, "")