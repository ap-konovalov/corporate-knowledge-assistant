"""Нарезка разделов(section) на чанки: каждый chunk получает meta своего раздела."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from kb_assistant.models import Chunk, Section

# Где резать, в порядке предпочтения: абзац → предложение → строка → слово → символ
SEPARATORS = ["\n\n", ". ", "\n", " ", ""]


def chunk_sections(sections: list[Section], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Нарезать разделы на чанки, сохранив метаданные и нумерацию внутри файла."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )
    chunks: list[Chunk] = []
    next_index: dict[str, int] = {}  # словарь «имя файла → номер следующего чанка», для каждого файла чанки нумеруем 0,1..
    for section in sections:
        if _is_header_only(section):
            continue
        for piece in splitter.split_text(section.text):
            # взять номер следующего чанка по имени файла, а если файла ещё нет в словаре, вернуть 0
            index = next_index.get(section.meta.source, 0)
            chunks.append(Chunk(text=piece, meta=section.meta, chunk_index=index))
            next_index[section.meta.source] = index + 1
    return chunks


def _is_header_only(section: Section) -> bool:
    """Раздел, в котором нет ничего, кроме строки заголовка."""
    # есть header и текст раздела после обрезки == header
    return section.meta.header is not None and section.text.strip() == section.meta.header