"""Выбор языковой модели по настройке LLM_PROVIDER."""

from kb_assistant.config import Settings
from kb_assistant.llm.base import LLMClient, LLMResponse
from kb_assistant.llm.yandexgpt import YandexGPTClient

__all__ = ["LLMClient", "LLMResponse", "create_llm"]


def create_llm(settings: Settings) -> LLMClient:
    """Создать клиент модели, указанной в настройках."""
    if settings.llm_provider == "yandexgpt":
        return YandexGPTClient(
            api_key=settings.yc_api_key.get_secret_value(),
            folder_id=settings.yc_folder_id,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )
    raise ValueError(f"Неизвестный LLM_PROVIDER: {settings.llm_provider}")