"""Прогон тестовых вопросов: ответы ассистента + автоматические проверки → отчёт в Markdown.

Запуск из корня проекта:
    uv run python scripts/run_eval.py
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from kb_assistant.citations import CITATION_RE
from kb_assistant.config import Settings, get_settings
from kb_assistant.pipeline import Answer, Assistant

QUESTIONS_FILE = Path("eval/questions.yaml")
REPORTS_DIR = Path("reports")

logger = logging.getLogger("eval")


@dataclass(frozen=True)
class EvalCase:
    """Один тестовый вопрос и ожидания по нему."""

    id: str
    type: str
    question: str
    expect_refusal: bool
    expected_sources: list[str]


@dataclass(frozen=True)
class EvalResult:
    """Ответ ассистента на тестовый вопрос и результаты проверок."""

    case: EvalCase
    answer: Answer
    cited_sources: set[str]  # файлы, на которые в ответе есть верные ссылки

    @property
    def refusal_ok(self) -> bool:
        return self.answer.citations.is_refusal == self.case.expect_refusal

    @property
    def sources_ok(self) -> bool:
        return set(self.case.expected_sources) <= self.cited_sources

    @property
    def passed(self) -> bool:
        return self.refusal_ok and self.answer.citations.is_ok and self.sources_ok


def load_cases(path: Path) -> list[EvalCase]:
    with path.open(encoding="utf-8") as file:
        return [EvalCase(**item) for item in yaml.safe_load(file)]


def run_case(assistant: Assistant, case: EvalCase) -> EvalResult:
    answer = assistant.ask(case.question)
    source_by_citation = {r.chunk.meta.citation: r.chunk.meta.source for r in answer.sources}
    cited = {source_by_citation[citation] for citation in answer.citations.valid}
    return EvalResult(case=case, answer=answer, cited_sources=cited)


def mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def highlight_citations(text: str) -> str:
    """Выделить ссылки в ответе жирным, чтобы их было видно в отчёте."""
    return CITATION_RE.sub(r"**\g<0>**", text)


def render_report(results: list[EvalResult], settings: Settings) -> str:
    passed = sum(result.passed for result in results)
    versions = sorted({r.answer.model_version for r in results if r.answer.model_version})
    lines = [
        "# Логи тестирования корпоративного ассистента",
        "",
        f"- Дата прогона: {datetime.now():%Y-%m-%d %H:%M}",
        f"- LLM: `{settings.llm_model}` (версия {', '.join(versions) or '—'}), temperature={settings.llm_temperature}",
        f"- Эмбеддинги: `{settings.embedding_model}`",
        f"- Нарезка и поиск: chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}, "
        f"top_k={settings.top_k}, min_score={settings.min_score}, max_per_source={settings.max_per_source}",
        "",
        f"**Итог: пройдено {passed} из {len(results)}.**",
        "",
        "| # | Тип | Отказ как ожидалось | Ссылки корректны | Ожидаемые источники | Итог |",
        "|---|---|---|---|---|---|",
    ]
    for number, r in enumerate(results, start=1):
        lines.append(
            f"| {number} | {r.case.type} | {mark(r.refusal_ok)} | {mark(r.answer.citations.is_ok)} "
            f"| {mark(r.sources_ok)} | {mark(r.passed)} |"
        )
    for number, result in enumerate(results, start=1):
        lines += render_case(number, result)
    return "\n".join(lines) + "\n"


def render_case(number: int, result: EvalResult) -> list[str]:
    case, answer, report = result.case, result.answer, result.answer.citations
    lines = [
        "",
        f"## {number}. {case.type}",
        "",
        f"**Вопрос:** {case.question}",
        "",
        "**Ответ ассистента:**",
        "",
        *[f"> {line}" for line in highlight_citations(answer.text).splitlines()],
        "",
        "**Найденные фрагменты (top-K):**",
        "",
        "| score | Источник | Напечатанная стр. | Раздел |",
        "|---|---|---|---|",
        *[
            f"| {s.score:.3f} | `{s.chunk.meta.citation}` | {s.chunk.meta.page_label or '—'} "
            f"| {s.chunk.meta.header or '—'} |"
            for s in answer.sources
        ],
        "",
        "**Проверки:**",
        "",
        f"- Ожидался отказ: {'да' if case.expect_refusal else 'нет'}; "
        f"получен: {'да' if report.is_refusal else 'нет'} {mark(result.refusal_ok)}",
        f"- Ссылки в ответе: {', '.join(f'`{c}`' for c in report.valid) or '—'} {mark(report.is_ok)}",
    ]
    if report.invalid:
        lines.append(f"- Выдуманные ссылки: {', '.join(f'`{c}`' for c in report.invalid)}")
    for sentence in report.uncited_sentences:
        lines.append(f"- Факт без ссылки: «{sentence}»")
    lines += [
        f"- Ожидаемые источники: {', '.join(case.expected_sources) or '—'}; "
        f"процитированы: {', '.join(sorted(result.cited_sources)) or '—'} {mark(result.sources_ok)}",
        f"- Модель вызвана: {'да' if answer.llm_called else 'нет'}; "
        f"токены: {answer.input_tokens} / {answer.output_tokens}; время: {answer.latency_s:.1f} с",
    ]
    return lines


def main() -> None:
    settings = get_settings()
    assistant = Assistant.from_settings(settings)
    results = []
    for case in load_cases(QUESTIONS_FILE):
        logger.info("Вопрос «%s»: %s", case.id, case.question)
        results.append(run_case(assistant, case))

    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M}.md"
    path.write_text(render_report(results, settings), encoding="utf-8")
    passed = sum(result.passed for result in results)
    print(f"Пройдено {passed} из {len(results)}. Отчёт: {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    main()