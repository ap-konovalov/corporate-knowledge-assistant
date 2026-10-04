"""Задать вопрос ассистенту из терминала.

Запуск из корня проекта:
    uv run python scripts/ask.py "Ваш вопрос"
"""

import argparse
import logging

from kb_assistant.config import get_settings
from kb_assistant.pipeline import Answer, Assistant


def print_answer(answer: Answer) -> None:
    """Напечатать ответ, источники и результат проверки ссылок."""
    print(f"\nВопрос: {answer.question}\n")
    print(answer.text)

    print("\nНайденные источники:")
    for result in answer.sources:
        meta = result.chunk.meta
        print(
            f"  {result.score:.3f}  {meta.citation}"
            f"  (напечатано: {meta.page_label or '—'}; раздел: {meta.header or '—'})"
        )

    report = answer.citations
    print(f"\nПроверка ссылок: {'OK' if report.is_ok else 'ПРОБЛЕМА'}")
    if report.invalid:
        print("  Выдуманные ссылки:", ", ".join(report.invalid))
    for sentence in report.uncited_sentences:
        print("  Факт без ссылки:", sentence)

    print(
        f"\nМодель вызвана: {answer.llm_called} | токены: {answer.input_tokens} / {answer.output_tokens}"
        f" | версия: {answer.model_version} | время: {answer.latency_s:.1f} с"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Вопрос корпоративному ассистенту")
    parser.add_argument("question", help="текст вопроса")
    args = parser.parse_args()

    assistant = Assistant.from_settings(get_settings())
    print_answer(assistant.ask(args.question))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    main()