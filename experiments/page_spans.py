"""Эксперимент: какая доля чанков захватывает 2+ страницы,
если резать PDF целиком, без учёта границ страниц.

Запуск из корня проекта:
    uv run python experiments/page_spans.py
"""

from bisect import bisect_right
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from kb_assistant.chunking import SEPARATORS
from kb_assistant.config import get_settings
from kb_assistant.loaders.pdf import PdfLoader

CHUNK_SIZES = [300, 800, 1500, 3000]
OVERLAP_SHARE = 0.125  # нахлёст = 1/8 размера, как 100 из 800 в основной настройке
REPORT_FILE = Path("reports/experiment_page_spans.md")


def page_texts(path: Path) -> list[str]:
    """Текст каждой страницы PDF по порядку (разделы одной страницы склеены)."""
    pages: dict[int, list[str]] = {}
    for section in PdfLoader().load(path):
        pages.setdefault(section.meta.page, []).append(section.text)
    return ["\n".join(parts) for _, parts in sorted(pages.items())]


def count_cross_page(pages: list[str], chunk_size: int) -> tuple[int, int]:
    """Нарезать документ целиком; вернуть (всего чанков, чанков на стыке страниц)."""
    page_starts: list[int] = []
    position = 0
    for text in pages:
        page_starts.append(position)
        position += len(text) + 1  # +1 — перевод строки между страницами
    full_text = "\n".join(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * OVERLAP_SHARE),
        separators=SEPARATORS,
        keep_separator="end",
        add_start_index=True,
    )
    chunks = splitter.create_documents([full_text])

    crossing = 0
    for chunk in chunks:
        start = chunk.metadata["start_index"]
        if start < 0:
            continue  # нарезчик не смог определить позицию — такой чанк не учитываем
        end = start + len(chunk.page_content) - 1
        first_page = bisect_right(page_starts, start) - 1
        last_page = bisect_right(page_starts, end) - 1
        if last_page > first_page:
            crossing += 1
    return len(chunks), crossing


def main() -> None:
    settings = get_settings()
    documents = {pdf: page_texts(pdf) for pdf in sorted(settings.raw_data_dir.glob("*.pdf"))}

    lines = [
        "# Эксперимент: чанки на стыке страниц",
        "",
        "PDF нарезан целиком, без учёта границ страниц. Чанк «на стыке» захватывает 2+ страницы,",
        "и одного номера страницы для него недостаточно.",
        "",
        "| chunk_size | Чанков | На стыке страниц | Доля |",
        "|---|---|---|---|",
    ]
    for size in CHUNK_SIZES:
        total = crossing = 0
        for pages in documents.values():
            doc_total, doc_crossing = count_cross_page(pages, size)
            total += doc_total
            crossing += doc_crossing
        lines.append(f"| {size} | {total} | {crossing} | {crossing / total:.0%} |")
    lines += ["", "В проекте чанки режутся внутри страницы: доля на стыке — 0 % при любом chunk_size."]

    report = "\n".join(lines) + "\n"
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()