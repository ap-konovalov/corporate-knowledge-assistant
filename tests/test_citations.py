"""Тесты проверки ссылок: типовые случаи и регрессия на точку внутри «стр. 4»."""

from kb_assistant.citations import check_citations
from kb_assistant.models import Chunk, SearchResult, SourceMeta

META = SourceMeta(source="gost.pdf", title="ГОСТ", doc_type="pdf", page=4)
RESULTS = [SearchResult(Chunk("текст", META, 0), 0.8)]


def test_invented_page_is_invalid():
    report = check_citations("Факт один [gost.pdf, стр. 4]. Факт два [gost.pdf, стр. 9].", RESULTS)
    assert report.invalid == ["[gost.pdf, стр. 9]"]
    assert not report.is_ok


def test_sentence_without_citation_is_reported():
    report = check_citations("Факт без ссылки. Факт со ссылкой [gost.pdf, стр. 4].", RESULTS)
    assert report.uncited_sentences == ["Факт без ссылки."]
    assert not report.is_ok


def test_dot_inside_citation_does_not_split_sentence():
    report = check_citations("Факт [gost.pdf, стр. 4].", RESULTS)
    assert report.uncited_sentences == []
    assert report.is_ok


def test_citation_after_period_belongs_to_sentence():
    report = check_citations("Факт один. [gost.pdf, стр. 4]", RESULTS)
    assert report.uncited_sentences == []
    assert report.is_ok


def test_refusal_is_ok():
    report = check_citations("Я не знаю.", RESULTS)
    assert report.is_refusal
    assert report.is_ok