"""Общий контракт для языковых моделей: на вход промпты, на выход ответ."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    """Ответ модели и расход токенов (для оценки стоимости)."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_version: str | None = None


class LLMClient(ABC):
    """Любая языковая модель, которую умеет вызывать ассистент."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Отправить системный и пользовательский промпты, вернуть ответ модели."""